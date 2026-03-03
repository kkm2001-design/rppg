import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import Delaunay
from torch import Tensor

class GPUPiecewiseAffineTransform:
    """
    High-Performance GPU Batch Warper.
    
    [작동 원리]
    1. Pre-computation (__init__):
       - UV 맵은 변하지 않으므로, UV 맵의 각 픽셀이 '어떤 삼각형'에 속하는지 미리 찾습니다.
       - 해당 픽셀을 만들기 위한 삼각형 꼭지점들의 가중치(Barycentric weights)를 미리 계산합니다.
       
    2. Batch Execution (__call__):
       - 루프 없이 (B, V, 2) 랜드마크와 미리 구해둔 가중치를 행렬 곱(Matmul)합니다.
       - 즉시 (B, H, W, 2) 샘플링 그리드가 생성되며, F.grid_sample로 일괄 처리합니다.
    """

    def __init__(self, keypoints_uv, output_size: int = 64 , device: str = "cuda"):
        """
        Args:
            keypoints_uv: (V, 2) Reference UV coordinates (0~1).
            output_size: Output spatial resolution (H=W).
            device: 'cuda' or 'cpu'.
        """
        self.device = torch.device(device)
        self.output_size = output_size
        
        # 1. UV 키포인트 준비
        if torch.is_tensor(keypoints_uv):
            uv_np = keypoints_uv.detach().cpu().numpy()
        else:
            uv_np = np.asarray(keypoints_uv, dtype=np.float32)
        
        # 픽셀 단위라면 정규화 (0~1 범위로)
        if uv_np.max() > 1.5:
            uv_np = uv_np / uv_np.max()

        self.keypoints_uv = torch.tensor(uv_np, dtype=torch.float32, device=self.device)
        
        # 2. 핵심: 그리드 가중치 사전 계산
        self._precompute_grid_weights(uv_np, output_size)

    def _precompute_grid_weights(self, uv_points: np.ndarray, size: int):
        """UV 그리드 픽셀별 삼각형 인덱스 및 가중치 계산"""
        # Destination Grid (UV space)
        y, x = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing='ij')
        grid_points = np.stack([x.ravel(), y.ravel()], axis=-1) # (HW, 2)

        # Delaunay Triangulation
        tri = Delaunay(uv_points)
        
        # 각 픽셀이 어떤 삼각형에 속하는지 찾기 (-1이면 Convex Hull 외부)
        simplex_indices = tri.find_simplex(grid_points)
        
        # 유효한(삼각형 내부) 픽셀만 마스킹
        valid_mask = simplex_indices != -1
        valid_simplices = simplex_indices[valid_mask]       # (Valid_Px,)
        valid_grid_points = grid_points[valid_mask]         # (Valid_Px, 2)

        # Barycentric coordinates 계산을 위한 행렬 준비
        # T * (p - r) -> (u, v)
        transform = tri.transform[valid_simplices, :2]      # (Valid_Px, 2, 2)
        offset = tri.transform[valid_simplices, 2]          # (Valid_Px, 2)
        diff = valid_grid_points - offset                   # (Valid_Px, 2)
        
        # 행렬 곱 (Einstein Summation)
        c = np.einsum('nij,nj->ni', transform, diff)        # (Valid_Px, 2)
        
        # 가중치 3개 생성 (1-u-v, u, v)
        weights = np.c_[1 - c[:, 0] - c[:, 1], c[:, 0], c[:, 1]]
        
        # --- GPU Tensor로 저장 (재사용) ---
        
        # 1. 픽셀에 영향을 주는 3개의 랜드마크 인덱스
        self.pixel_v_indices = torch.tensor(
            tri.simplices[valid_simplices], dtype=torch.long, device=self.device
        ) # (Valid_Px, 3)

        # 2. 각 랜드마크의 가중치 (Matmul을 위해 unsqueeze)
        self.pixel_weights = torch.tensor(
            weights, dtype=torch.float32, device=self.device
        ).unsqueeze(1) # (Valid_Px, 1, 3) 

        # 3. 유효 픽셀 마스크
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device) # (HW,)
        
        self.H, self.W = size, size

    def __call__(self, frames: Tensor, landmarks: Tensor) -> Tensor:
        """
        Batch Warp.
        
        Args:
            frames: (B, C, H, W) Source images.
            landmarks: (B, V, 2) Image-space landmarks matching keypoints_uv.
        
        Returns:
            warped: (B, H_out, W_out, C)
        """
        # 입력 차원 정리
        if frames.ndim == 3: frames = frames.unsqueeze(0)
        if landmarks.ndim == 2: landmarks = landmarks.unsqueeze(0)
        
        # (B, H, W, C) -> (B, C, H, W) 변환 (grid_sample용)
        permute_back = False
        if frames.shape[-1] == 3: 
            frames = frames.permute(0, 3, 1, 2)
            permute_back = True
            
        B, C, img_H, img_W = frames.shape
        frames = frames.float()
        if frames.max() > 1.1: frames /= 255.0

        # --- 고속 워핑 로직 ---
        
        # 1. 유효 픽셀들의 랜드마크 좌표 가져오기 (Gather)
        # landmarks: (B, V, 2)
        # indices: (Valid_Px, 3) -> batch_tri_verts: (B, Valid_Px, 3, 2)
        batch_tri_verts = landmarks[:, self.pixel_v_indices, :] 

        # 2. 가중치 적용하여 좌표 계산 (Matmul)
        # (1, Valid_Px, 1, 3) @ (B, Valid_Px, 3, 2) -> (B, Valid_Px, 1, 2)
        sampling_coords_valid = torch.matmul(self.pixel_weights, batch_tri_verts).squeeze(2)

        # 3. 전체 그리드 생성 및 채우기
        grid = torch.zeros((B, self.H * self.W, 2), dtype=torch.float32, device=self.device)
        
        # 마스크 확장 (B, HW)
        batch_mask = self.valid_mask.unsqueeze(0).expand(B, -1)
        grid[batch_mask] = sampling_coords_valid.reshape(-1, 2)

        # 4. 좌표 정규화 (-1 ~ 1)
        grid_norm = grid.view(B, self.H, self.W, 2).clone()
        grid_norm[..., 0] = 2.0 * grid_norm[..., 0] / (img_W - 1) - 1.0
        grid_norm[..., 1] = 2.0 * grid_norm[..., 1] / (img_H - 1) - 1.0

        # 5. 배경 처리 (Invalid pixels -> -2.0)
        # grid_sample의 padding_mode='zeros'가 -1~1 범위를 벗어난 좌표를 0(검정)으로 처리함
        grid_norm[~batch_mask.view(B, self.H, self.W)] = -2.0 

        # 6. Sampling
        warped = F.grid_sample(frames, grid_norm, mode='bilinear', padding_mode='zeros', align_corners=True)

        if permute_back:
            warped = warped.permute(0, 2, 3, 1) # (B, H, W, C)

        return warped
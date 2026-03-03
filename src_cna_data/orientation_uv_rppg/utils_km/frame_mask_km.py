import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from scipy.spatial import Delaunay
from typing import Tuple

class BatchUVAngleMasker:
    """
    Batch-optimized Surface Orientation Masker.
    
    [핵심]
    - Scipy Loop 제거: 느린 griddata 대신 GPU 행렬 연산으로 보간합니다.
    - Face Normal -> Vertex Normal -> Pixel Normal 보간 과정을 배치로 처리합니다.
    """

    def __init__(
        self,
        faces: np.ndarray,
        landmarks_uv: np.ndarray, 
        output_size: int = 64 ,
        camera_normal: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        device: str = "cuda",
        eps: float = 1e-12,
    ) -> None:
        self.device = torch.device(device)
        self.faces = torch.tensor(faces, dtype=torch.long, device=self.device) # (F, 3)
        self.output_size = output_size
        self.eps = eps
        
        # 카메라 벡터 정규화
        cam = torch.tensor(camera_normal, dtype=torch.float32, device=self.device)
        self.cam = cam / torch.clamp(torch.norm(cam), min=eps)

        # Interpolation 가중치 미리 계산 (Warper와 동일)
        self._precompute_interpolation(landmarks_uv, output_size)

    def _precompute_interpolation(self, uv_points: np.ndarray, size: int):
        """Vertices의 값을 UV 픽셀로 보간하기 위한 가중치 계산"""
        if torch.is_tensor(uv_points):
            uv_points = uv_points.detach().cpu().numpy()
            
        y, x = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing='ij')
        grid_points = np.stack([x.ravel(), y.ravel()], axis=-1)

        tri = Delaunay(uv_points)
        simplex_indices = tri.find_simplex(grid_points)
        
        valid_mask = simplex_indices != -1
        valid_simplices = simplex_indices[valid_mask]
        valid_grid_points = grid_points[valid_mask]

        transform = tri.transform[valid_simplices, :2]
        offset = tri.transform[valid_simplices, 2]
        diff = valid_grid_points - offset
        
        c = np.einsum('nij,nj->ni', transform, diff)
        weights = np.c_[1 - c[:, 0] - c[:, 1], c[:, 0], c[:, 1]]

        # GPU Tensor 저장
        self.pixel_v_indices = torch.tensor(tri.simplices[valid_simplices], dtype=torch.long, device=self.device)
        self.pixel_weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1) # (Valid, 1, 3)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        self.H, self.W = size, size

    def __call__(
        self,
        frames_uv: Tensor,            # (B, H, W, C)
        landmarks_xyz: Tensor,        # (B, V, 3) - Valid Only
        degree_threshold: float = 60.0,
    ) -> Tensor:
        """
        Args:
            frames_uv: Warped UV frames.
            landmarks_xyz: 3D landmarks corresponding to the frames.
        """
        B = landmarks_xyz.shape[0]
        if landmarks_xyz.device != self.device:
            landmarks_xyz = landmarks_xyz.to(self.device)

        # 1. Face Normal 계산 (Batch Cross Product)
        v = landmarks_xyz[:, self.faces, :] # (B, F, 3, 3)
        e1 = v[:, :, 1] - v[:, :, 0]
        e2 = v[:, :, 2] - v[:, :, 0]
        
        face_normals = torch.cross(e1, e2, dim=-1) # (B, F, 3)
        
        # 2. Vertex Normal 누적 (Batch Scatter Add)
        vertex_normals = torch.zeros_like(landmarks_xyz)
        
        # 각 Face의 Normal을 구성 Vertex에 더해줌
        for i in range(3):
            indices = self.faces[:, i] # (F,)
            # (B, F, 3) -> (B, V, 3) 로 인덱스에 맞춰 더함
            vertex_normals.index_add_(1, indices, face_normals)
            
        vertex_normals = F.normalize(vertex_normals, p=2, dim=-1, eps=self.eps)

        # 3. 카메라와의 각도(Cosine) 계산
        # (B, V, 3) @ (3,) -> (B, V)
        cos_vals_verts = torch.matmul(vertex_normals, self.cam) 

        # 4. 픽셀로 값 보간 (Matrix Multiplication)
        # Vertex 값 Gather: (B, V) -> (B, Valid_Px, 3)
        batch_vals = cos_vals_verts[:, self.pixel_v_indices]
        
        # 가중치 적용: (1, Valid_Px, 1, 3) @ (B, Valid_Px, 3, 1) -> (B, Valid_Px)
        pixel_vals = torch.matmul(
            self.pixel_weights, 
            batch_vals.unsqueeze(-1)
        ).squeeze(-1).squeeze(-1)

        # 5. Mask 생성 (Thresholding)
        cos_threshold = np.cos(np.deg2rad(degree_threshold))
        
        # Cosine > Threshold (덜 기울어짐) -> True (유지)
        keep_mask_pixels = (pixel_vals >= cos_threshold)
        
        # 전체 이미지로 확장
        batch_valid_mask = self.valid_mask.unsqueeze(0).expand(B, -1) # (B, HW)
        final_mask_flat = torch.zeros((B, self.H * self.W), dtype=torch.bool, device=self.device)
        
        # 유효 영역에 값 채우기
        final_mask_flat[batch_valid_mask] = keep_mask_pixels.reshape(-1)
        
        # (B, H, W, 1) 브로드캐스팅용 shape
        final_mask = final_mask_flat.view(B, self.H, self.W).unsqueeze(-1)

        # 6. 마스킹 적용 (각도 안 맞으면 0)
        return frames_uv * final_mask.float()
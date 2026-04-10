import torch


def calculate_laplacian_with_self_loop(matrix: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized Laplacian: D^{-1/2}(A+I)D^{-1/2}"""
    matrix = matrix + torch.eye(matrix.size(0), device=matrix.device)
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)

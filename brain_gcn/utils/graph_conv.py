import torch


def calculate_laplacian_with_self_loop(matrix: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized adjacency: D^{-1/2}(A+I)D^{-1/2}.

    Accepts a single adjacency matrix ``(N, N)`` or a batch ``(B, N, N)``.
    """
    if matrix.dim() == 2:
        eye = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
        matrix = matrix + eye
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)

    if matrix.dim() == 4:
        batch_size, num_windows, num_nodes, _ = matrix.shape
        matrix = matrix.reshape(batch_size * num_windows, num_nodes, num_nodes)
        norm = calculate_laplacian_with_self_loop(matrix)
        return norm.reshape(batch_size, num_windows, num_nodes, num_nodes)

    if matrix.dim() != 3:
        raise ValueError(
            "Expected adjacency shape (N, N), (B, N, N), or (B, W, N, N), "
            f"got {tuple(matrix.shape)}"
        )

    num_nodes = matrix.size(-1)
    eye = torch.eye(num_nodes, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
    matrix = matrix + eye
    row_sum = matrix.sum(-1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt = d_inv_sqrt.view_as(row_sum)
    return d_inv_sqrt.unsqueeze(-1) * matrix * d_inv_sqrt.unsqueeze(-2)

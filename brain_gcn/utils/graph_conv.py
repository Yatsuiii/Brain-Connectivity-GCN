"""
Graph convolution utilities.

v2 additions:
  - drop_edge(): DropEdge regularisation (Rong et al. 2020)
"""

import torch


def calculate_laplacian_with_self_loop(matrix: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized adjacency: D^{-1/2}(A+I)D^{-1/2}.

    Accepts a single adjacency matrix ``(N, N)``, a batch ``(B, N, N)``,
    or a dynamic sequence ``(B, W, N, N)``.
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


def drop_edge(
    adj: torch.Tensor,
    p: float = 0.1,
    training: bool = True,
) -> torch.Tensor:
    """Randomly zero out edges during training (DropEdge, Rong et al. 2020).

    Works for (N,N), (B,N,N), and (B,W,N,N) adjacency shapes.
    Self-loops are NOT dropped (they are added after normalisation anyway).

    Density-aware: scales drop probability inversely with graph sparsity.
    After fc_threshold, sparse graphs need careful regularisation to avoid
    removing too much signal. p_eff = min(p, 0.5 * density) ensures we never
    drop more than 50% of actual edges.

    Parameters
    ----------
    adj      : adjacency tensor (any supported shape)
    p        : base probability of dropping each edge
    training : no-op when False (eval / inference)

    Returns
    -------
    Masked adjacency with the same shape as input.
    """
    if not training or p == 0.0:
        return adj
    
    # Compute graph density: proportion of non-zero edges
    density = (adj > 0).float().mean().item()
    
    # Scale drop probability inversely with density
    # Never drop more than 50% of actual edges
    p_eff = min(p, density * 0.5)
    
    # bernoulli mask: 1 = keep, 0 = drop
    mask = torch.bernoulli(torch.full_like(adj, 1.0 - p_eff))
    return adj * mask

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization (RMSNorm), a variant of layer normalization
    that normalizes inputs based on their root mean square rather than mean and variance.
    
    RMSNorm is often used in models like LLaMA for improved numerical stability and efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        """
        Args:
            dim (int): The size of the last dimension of the input tensor (i.e., embedding dimension).
            eps (float): A small constant added to the denominator to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

        # Learnable scale parameter (like gamma in LayerNorm)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (Tensor): Input tensor of shape (..., dim), where `dim` is the last dimension.
        
        Returns:
            Tensor: Normalized tensor with same shape as input.
        """
        # Compute the root mean square (RMS) across the last dimension
        rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)

        # Normalize the input and apply learned scaling
        x_normed = x / (rms + self.eps)

        return self.weight * x_normed

# Example: normalize a batch of embeddings
x = torch.randn(4, 10, 512)  # (batch_size, seq_len, embed_dim)

rmsnorm = RMSNorm(dim=512)
out = rmsnorm(x)

print(f"output:", {out.shape})  # Output: torch.Size([4, 10, 512])

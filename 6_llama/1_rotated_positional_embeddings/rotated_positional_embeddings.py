import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------
# 🔁 Rotary Positional Embedding (RoPE)
# ------------------------------------------------------
class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE), which injects position information 
    into token representations using trigonometric rotations. Used in models like LLaMA.
    
    The main idea is to rotate hidden dimensions of embeddings based on position,
    improving extrapolation to longer sequences.

    Summary of Key Components:
        RoPE - Encodes positional information without adding trainable parameters. 
               Rotates token representations in embedding space using sine/cosine patterns.
        rotate_half - Simulates a 90-degree rotation used in RoPE math.
    """
    def __init__(self, dim: int):
        """
        Args:
            dim (int): The embedding dimension. Should be even for proper rotation.
        """
        super().__init__()

        # Compute inverse frequency for sinusoidal rotation
        # Shape: (dim/2,)
        # Formula: 1 / (10000^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        # Register buffer so that `inv_freq` is part of the state but not trainable
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to apply rotary positional embedding.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Tensor: Positionally rotated embedding with same shape as x
        """
        # Sequence length
        seq_len = x.size(1)

        # Create a time vector [0, 1, 2, ..., seq_len-1] and match it with the inv_freq shape
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)

        # Use einsum to compute outer product of time steps with inverse frequencies
        # Resulting shape: (seq_len, dim/2)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)

        # Concatenate the frequencies to match the full hidden dimension (dim)
        # Final shape: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Compute cosine and sine embeddings for each position and unsqueeze for broadcasting
        cos_emb = emb.cos().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, dim)
        sin_emb = emb.sin().unsqueeze(0).unsqueeze(0)

        # Apply RoPE: rotate_half applies 90-degree rotation to half of hidden dim
        return (x * cos_emb) + (rotate_half(x) * sin_emb)


# ------------------------------------------------------
# 🔁 Rotation Helper Function
# ------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper function to rotate half of the hidden dimension by 90 degrees.
    
    Args:
        x (Tensor): Input tensor with shape (..., dim)

    Returns:
        Tensor: Rotated tensor with same shape.
    """
    # Split the last dimension into even and odd parts
    x1 = x[..., :x.size(-1) // 2]  # First half
    x2 = x[..., x.size(-1) // 2:]  # Second half

    # Rotate: [-x2, x1] simulates complex multiplication (used in RoPE)
    return torch.cat((-x2, x1), dim=-1)


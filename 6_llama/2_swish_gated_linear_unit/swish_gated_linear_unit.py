import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------
# 🔂 SwiGLU Activation Function
# ------------------------------------------------------
class SwiGLU(nn.Module):
    """
    Implements SwiGLU: Swish-Gated Linear Unit.
    A fast and stable activation function introduced in transformer architectures like PaLM.

    Formula:
        SwiGLU(x) = silu(a) * b, where (a, b) = chunk(x, 2)

    Summary of Key Component:
        SwiGLU - A gated activation function (silu(a) * b) that improves efficiency and performance compared to GeLU or ReLU.
    """
    def __init__(self):
        """
        Constructor for SwiGLU. No learnable parameters needed.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SwiGLU activation.

        Args:
            x (Tensor): Input tensor of shape (..., 2 * hidden_dim)

        Returns:
            Tensor: Output tensor of shape (..., hidden_dim)
        """
        # Split the last dimension into two equal parts
        a, b = x.chunk(2, dim=-1)

        # Apply SiLU (Swish) to a, then gate with b
        return F.silu(a) * b
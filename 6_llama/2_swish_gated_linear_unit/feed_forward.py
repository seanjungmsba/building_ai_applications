import torch
import torch.nn as nn
import torch.nn.functional as F
from swish_gated_linear_unit import SwiGLU

class FeedForward(nn.Module):
    """
    FeedForward Layer with SwiGLU activation as used in LLaMA and other transformer models.

    Consists of:
    - Linear layer projecting from `embed_dim` to `2 * ffn_dim` (required for SwiGLU chunking)
    - SwiGLU activation (gated non-linearity)
    - Linear projection back to `embed_dim`
    - Dropout for regularization

    Why Use SwiGLU in FeedForward?
        - SwiGLU enhances the transformer’s non-linearity while remaining faster and more memory efficient than GeLU. 
        - By chunking the hidden states and applying a sigmoid-gated mechanism, 
          it enables richer token-level transformations and is used in models like LLaMA, GPT-4, and PaLM.
    """

    def __init__(self, embed_dim, ffn_dim):
        super().__init__()

        # First linear layer expands input dimension to twice the FFN dim
        # Needed because SwiGLU splits it into two parts
        self.linear1 = nn.Linear(embed_dim, ffn_dim * 2)

        # SwiGLU activation (gated activation with SiLU)
        self.swiglu = SwiGLU()

        # Second linear layer projects back to embedding dimension
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass of the FeedForward block

        Args:
            x (Tensor): Input tensor of shape (B, S, E)

        Returns:
            Tensor: Output tensor of shape (B, S, E)
        """
        x = self.linear1(x)       # (B, S, 2*ffn_dim)
        x = self.swiglu(x)        # (B, S, ffn_dim) after gated activation
        x = self.linear2(x)       # (B, S, embed_dim)
        x = self.dropout(x)       # Apply dropout
        return x
    
if __name__ == "__main__":
    # Hyperparameters
    embed_dim = 64
    ffn_dim = 128
    batch_size = 2
    seq_len = 10

    # Dummy input: shape (batch_size, seq_len, embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Instantiate and run FeedForward block
    ff = FeedForward(embed_dim, ffn_dim)
    ff.eval()

    with torch.no_grad():
        out = ff(x)

    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)

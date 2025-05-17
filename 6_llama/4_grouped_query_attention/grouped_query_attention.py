import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)

    Implements a memory- and compute-efficient variant of multi-head attention used in models like LLaMA.
    Instead of assigning a unique key and value head to each query head, GQA shares key/value projections 
    across groups of query heads.

    Args:
        embed_dim (int): Total embedding dimension of the model.
        num_heads (int): Total number of query heads.
        num_kv_heads (int): Number of shared key/value heads (must divide num_heads).
        dropout (float): Dropout rate applied to attention weights.
    """

    def __init__(self, embed_dim, num_heads, num_kv_heads, dropout):
        super().__init__()

        # === Sanity checks ===
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        # === Save core attributes ===
        self.embed_dim = embed_dim                # Total embedding size
        self.num_heads = num_heads                # Total query heads
        self.num_kv_heads = num_kv_heads          # Shared key/value heads
        self.head_dim = embed_dim // num_heads    # Dimension of each query head
        self.kv_group_size = num_heads // num_kv_heads  # Query heads per KV head

        # === Linear projections ===
        # Full Q projection: output shape [B, S, E]
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # K and V projections use fewer parameters:
        # Output shape is smaller: [B, S, E / kv_group_size]
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.kv_group_size)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.kv_group_size)

        # === Output projection ===
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # === Dropout on attention weights ===
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of Grouped Query Attention.

        Args:
            x (Tensor): Input tensor of shape (B, S, E) where:
                - B = batch size
                - S = sequence length
                - E = embedding dimension

        Returns:
            Tensor: Output tensor of shape (B, S, E)
        """

        B, S, E = x.size()  # Unpack input dimensions

        # === Compute Query ===
        # Shape after projection: (B, S, E)
        # Then reshape to (B, num_heads, S, head_dim)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # === Compute Key and Value ===
        # We use fewer key/value heads → reduced dimension: E / kv_group_size
        kv_embed_dim = self.embed_dim // self.kv_group_size
        kv_head_dim = kv_embed_dim // self.num_kv_heads  # dim per kv head

        # Shape after projection and reshaping: (B, num_kv_heads, S, kv_head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, kv_head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, kv_head_dim).transpose(1, 2)

        # === Align KV with Q ===
        # Repeat K and V so that each query head has a corresponding key/value
        # Final shape: (B, num_heads, S, head_dim)
        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)

        # === Scaled Dot-Product Attention ===
        # Attention score: (B, num_heads, S, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # === Causal Mask ===
        # Prevent attending to future positions (for autoregressive decoding)
        causal_mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # === Softmax normalization ===
        attn_weights = F.softmax(attn_scores, dim=-1)

        # === Dropout on attention weights ===
        attn_weights = self.dropout(attn_weights)

        # === Weighted sum over value vectors ===
        # Output: (B, num_heads, S, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # === Merge heads ===
        # Transpose and reshape: (B, S, E)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, E)

        # === Final output projection ===
        return self.out_proj(attn_output)

# === Example Usage ===
if __name__ == "__main__":
    embed_dim = 64
    num_heads = 8
    num_kv_heads = 2
    dropout = 0.1

    gqa = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, dropout)
    gqa.eval()

    dummy_input = torch.randn(2, 5, embed_dim)
    output = gqa(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

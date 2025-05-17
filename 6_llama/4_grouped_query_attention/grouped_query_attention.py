import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA), where queries are projected per-head,
    but keys and values are shared across groups of heads.
    """

    def __init__(self, embed_dim: int, num_query_heads: int, num_kv_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_dim (int): Dimension of input and output embeddings
            num_query_heads (int): Number of query heads (higher than kv heads)
            num_kv_heads (int): Number of key/value heads (shared among query groups)
            dropout (float): Dropout probability for attention weights
        """
        super().__init__()
        assert embed_dim % num_query_heads == 0, "Embedding dimension must be divisible by number of query heads"
        assert num_query_heads % num_kv_heads == 0, "Query heads must be divisible by key/value heads"

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.q_per_kv = num_query_heads // num_kv_heads

        # Query projection: one per query head
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Key and Value projections: fewer heads, shared across groups
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute grouped query attention for the input sequence.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            mask (Tensor): Optional mask tensor of shape (batch_size, 1, 1, seq_len)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, T, _ = x.size()  # Batch size, sequence length

        # Project input to queries, keys, values
        q = self.q_proj(x).view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)  # (B, QH, T, D)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, KVH, T, D)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)     # (B, KVH, T, D)

        # Expand k/v to match q: duplicate each kv head q_per_kv times
        k = k.unsqueeze(2).repeat(1, 1, self.q_per_kv, 1, 1).view(B, self.num_query_heads, T, self.head_dim)
        v = v.unsqueeze(2).repeat(1, 1, self.q_per_kv, 1, 1).view(B, self.num_query_heads, T, self.head_dim)

        # Attention score computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, QH, T, T)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to value
        attn_output = torch.matmul(attn_weights, v)  # (B, QH, T, D)

        # Reshape back to (B, T, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)

        return self.out_proj(attn_output)

gqa = GroupedQueryAttention(embed_dim=512, num_query_heads=16, num_kv_heads=4)
x = torch.randn(8, 128, 512)  # (batch, seq_len, embed_dim)
out = gqa(x)
print(f"Output:", {out.shape})  # Output: (8, 128, 512)
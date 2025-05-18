import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Llama(nn.Module):
    """
    Llama Language Model Architecture (simplified)

    This class defines a decoder-only transformer model similar to LLaMA. It includes:
    - Token and rotary position embeddings
    - Stacked decoder blocks (with grouped query attention)
    - Final layer normalization and output projection to vocabulary logits

    Args:
        vocab_size (int): Size of the vocabulary (for output tokens).
        max_seq_len (int): Maximum input sequence length supported.
        embed_dim (int): Dimensionality of token embeddings.
        num_heads (int): Number of attention heads (for query projections).
        num_layers (int): Number of decoder blocks.
        num_kv_heads (int): Number of shared key/value heads (GQA).
        dropout (float): Dropout rate.
    """
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, num_layers, num_kv_heads, dropout=0.1):
        super().__init__()

        # === Embedding layers ===
        # Embeds token indices into dense vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Applies rotary positional encodings (RoPE) to inject position info directly into attention
        self.position_embedding = RotaryPositionalEmbedding(embed_dim)

        # === Decoder Blocks ===
        # Stack of `num_layers` decoder blocks, each with attention + feedforward layers
        self.layers = nn.ModuleList([
            Decoder(embed_dim, num_heads, num_kv_heads, dropout) for _ in range(num_layers)
        ])

        # === Output Layers ===
        # RMS normalization before logits (like LayerNorm but more efficient)
        self.layer_norm = RMSNorm(embed_dim)

        # Final linear layer to project to vocabulary size (for logits/predictions)
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)

        # Track maximum sequence length (not used internally here, but for external safety)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        """
        Forward pass for the Llama model.

        Args:
            x (Tensor): Token indices of shape (B, S), where:
                        B = batch size, S = sequence length

        Returns:
            Tensor: Logits of shape (B, S, vocab_size)
        """
        seq_len = x.size(1)  # Get sequence length from input

        # 1. Convert token indices to embeddings → (B, S, E)
        token_emb = self.token_embedding(x)

        # 2. Apply rotary positional embedding → injects position-aware info
        x = self.position_embedding(token_emb)

        # 3. Pass input through all decoder blocks (each does attention + feedforward)
        for layer in self.layers:
            x = layer(x)

        # 4. Apply RMSNorm to final hidden states
        x = self.layer_norm(x)

        # 5. Project to logits over vocab → used for language modeling
        logits = self.output_layer(x)

        return logits

class Decoder(nn.Module):
    """
    A single decoder block from the LLaMA architecture.

    Consists of:
    - Grouped Query Attention (GQA)
    - SwiGLU FeedForward Network
    - Two RMSNorm layers
    - Residual connections and dropout
    """

    def __init__(self, embed_dim, num_heads, num_kv_heads, dropout):
        super().__init__()

        # === Attention Layer ===
        # Grouped Query Attention (fewer key/value heads for efficiency)
        self.attention = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, dropout)

        # === Feedforward Layer ===
        # SwiGLU FeedForward block: uses 4x embedding size (standard in transformers)
        self.feed_forward = FeedForward(embed_dim, 4 * embed_dim)

        # === Normalization Layers ===
        self.layer_norm1 = RMSNorm(embed_dim)  # Before attention
        self.layer_norm2 = RMSNorm(embed_dim)  # Before feedforward

        # === Dropout ===
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the decoder block.

        Args:
            x (Tensor): Input of shape (B, S, E)

        Returns:
            Tensor: Output of shape (B, S, E)
        """
        # === Attention + Residual + Norm ===
        # 1. Apply grouped self-attention
        attn_output = self.attention(x)

        # 2. Add residual and normalize
        x = self.layer_norm1(x + self.dropout(attn_output))

        # === Feedforward + Residual + Norm ===
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x

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
        cos_emb = emb.cos().unsqueeze(0)  # Shape: (1, seq_len, dim)
        sin_emb = emb.sin().unsqueeze(0)

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

# Class for SwiGLU
class SwiGLU(nn.Module):
    # Forward Pass without a constructor
    def forward(self, x: torch.Tensor):
       
        # Two chunks of the x input
        a, b = x.chunk(2, dim=-1)

        # Sigmoid Linear Unit
        return F.silu(a) * b
    
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
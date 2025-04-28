"""
GPT Architecture - Full Detailed Version

This module defines a simplified GPT model using PyTorch:

- GPT (stacked decoder-only transformer blocks)
- DecoderBlock (masked self-attention + feedforward sublayers)
- MaskedMultiHeadSelfAttention (with causal masking)

Key ideas:
- Causal masking prevents attending to future tokens.
- Token embeddings + learned positional embeddings.
- Layer Normalization and residual connections.
- GeLU activation and large intermediate dimension in feedforward networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================
# GPT Model
# ================================
class GPT(nn.Module):
    """
    GPT Model

    - Input: token IDs
    - Output: logits over vocabulary for next-token prediction
    - Stacks multiple decoder blocks (masked attention + feedforward)
    """

    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, num_layers, dropout=0.1):
        """
        Initialize GPT architecture.

        Args:
            vocab_size (int): Number of unique tokens (words, subwords) in vocabulary.
            max_seq_len (int): Maximum input length (for learned positional embeddings).
            embed_dim (int): Embedding dimension for tokens and positions.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of decoder blocks to stack.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()

        # Embed tokens (word embeddings): [vocab_size, embed_dim]
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Embed positions: [max_seq_len, embed_dim]
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Stack multiple Decoder Blocks
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Final normalization layer after transformer blocks
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Final output layer projecting to vocabulary size
        # Maps from hidden dimension -> vocab logits
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)

        # Store maximum sequence length for later
        self.max_seq_len = max_seq_len

    def forward(self, x):
        """
        Forward pass through the GPT model.

        Args:
            x (torch.Tensor): Input tensor of token IDs [batch_size, seq_len]

        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size]
        """
        seq_len = x.size(1)  # Get sequence length from input

        # Embed tokens
        token_emb = self.token_embedding(x)  # Shape: [batch_size, seq_len, embed_dim]

        # Embed positions (use arange on device for batching)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))  # [seq_len, embed_dim]

        # Combine token embeddings + positional embeddings (broadcasted)
        x = token_emb + pos_emb.unsqueeze(0)  # [batch_size, seq_len, embed_dim]

        # Pass through each stacked decoder block
        for layer in self.layers:
            x = layer(x)

        # Apply final normalization
        x = self.layer_norm(x)

        # Project to logits over vocabulary
        logits = self.output_layer(x)

        return logits


# ================================
# Decoder Block
# ================================
class DecoderBlock(nn.Module):
    """
    Single Decoder Block = (Causal Self-Attention + Feedforward) + Residual connections

    Structure:
    1. Masked Multi-Head Self-Attention (+ residual + layer norm)
    2. Feedforward Network (+ residual + layer norm)
    """

    def __init__(self, embed_dim, num_heads, dropout):
        """
        Initialize Decoder Block.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()

        # Causal masked self-attention layer
        self.attention = MaskedMultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # Feedforward sublayer:
        # Expand dimension 4x, apply GeLU, shrink back to embed_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # Expand [embed_dim -> 4 * embed_dim]
            nn.GELU(),                            # Nonlinear activation (smooth ReLU alternative)
            nn.Linear(4 * embed_dim, embed_dim),   # Contract back to original dimension
            nn.Dropout(dropout)                    # Dropout for regularization
        )

        # Layer normalization after attention and feedforward sublayers
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        # Dropout for residual paths
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through decoder block.

        Args:
            x (torch.Tensor): Input [batch_size, seq_len, embed_dim]

        Returns:
            torch.Tensor: Output [batch_size, seq_len, embed_dim]
        """
        # 1️⃣ Self-attention with residual connection
        attn_output = self.attention(x)
        x = self.layer_norm1(x + self.dropout(attn_output))  # Residual + Norm

        # 2️⃣ Feedforward network with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))    # Residual + Norm

        return x


# ================================
# Masked Multi-Head Self-Attention
# ================================
class MaskedMultiHeadSelfAttention(nn.Module):
    """
    Masked Multi-Head Self Attention

    - Projects inputs to Q, K, V.
    - Applies scaled dot-product attention with causal mask.
    - Concatenates and projects heads back to original dimension.
    """

    def __init__(self, embed_dim, num_heads, dropout):
        """
        Initialize Masked Attention.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()

        # Split hidden dimension across heads
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for queries, keys, and values
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Dropout after softmax attention weights
        self.dropout = nn.Dropout(dropout)

        # Final output projection after concatenating all heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass through masked self-attention.

        Args:
            x (torch.Tensor): [batch_size, seq_len, embed_dim]

        Returns:
            torch.Tensor: Output after attention [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.size()

        # Project input embeddings to Q, K, V and reshape into heads
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask to prevent attention to future tokens
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Softmax over the last dimension (sequence length)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Dropout on attention weights
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of value vectors
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Final linear projection
        output = self.out_proj(attn_output)

        return output

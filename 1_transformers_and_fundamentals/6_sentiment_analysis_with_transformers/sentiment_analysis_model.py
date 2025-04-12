import torch
import torch.nn as nn

# -----------------------------------------------------
# Class: SelfAttention
# -----------------------------------------------------
# Implements multi-head self-attention mechanism.
# This layer is used to allow each token in a sequence
# to attend to other tokens and build contextualized embeddings.
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        """
        Initializes the SelfAttention module.

        Args:
            embed_size (int): Total dimensionality of token embeddings.
            heads (int): Number of parallel attention heads.
        """
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size            # Full embedding size (e.g., 300)
        self.heads = heads                      # Number of attention heads (e.g., 6)
        self.head_dim = embed_size // heads     # Size per attention head (must divide evenly)

        # Linear layers for projecting input embeddings to Q, K, V representations
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Output projection layer to merge all heads back to embed_size
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """
        Performs the forward pass of the self-attention layer.

        Args:
            values (Tensor): Value tensor of shape [batch_size, seq_len, embed_size]
            keys (Tensor): Key tensor of shape [batch_size, seq_len, embed_size]
            query (Tensor): Query tensor of shape [batch_size, seq_len, embed_size]
            mask (Tensor or None): Optional attention mask

        Returns:
            Tensor: Output of self-attention layer of shape [batch_size, seq_len, embed_size]
        """
        N = query.shape[0]  # Batch size

        # Sequence lengths for each tensor
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Reshape into [batch_size, seq_len, heads, head_dim]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Project inputs into Q, K, V subspaces using learned linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Compute attention scores: dot product between Q and K^T
        # Output shape: [batch_size, heads, query_len, key_len]
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply mask (if provided) to ignore certain positions (e.g., padding)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize attention scores and convert to probabilities
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Use attention scores to weight the value vectors
        # Output shape: [batch_size, query_len, heads, head_dim]
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values])

        # Concatenate heads and reshape to [batch_size, query_len, embed_size]
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # Final projection layer
        out = self.fc_out(out)

        return out

# -----------------------------------------------------
# Class: SentimentAnalysisModel
# -----------------------------------------------------
# A simple classification model that applies self-attention
# followed by mean pooling and a classification head.
class SentimentAnalysisModel(nn.Module):
    def __init__(self, embed_size: int, heads: int, num_classes: int):
        """
        Initializes the sentiment analysis model.

        Args:
            embed_size (int): Size of the input word embeddings.
            heads (int): Number of attention heads to use.
            num_classes (int): Number of output sentiment classes (e.g., 2 for binary).
        """
        super(SentimentAnalysisModel, self).__init__()

        # Self-attention block to capture contextual dependencies
        self.attention = SelfAttention(embed_size, heads)

        # Final linear layer to map pooled features to class logits
        self.fc_out = nn.Linear(embed_size, num_classes)

    def forward(self, x, mask=None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, embed_size]
            mask (Tensor or None): Optional mask tensor for attention

        Returns:
            Tensor: Logits of shape [batch_size, num_classes]
        """
        # Apply self-attention
        attention_out: torch.Tensor = self.attention(x, x, x, mask)

        # Perform mean pooling across the sequence dimension
        pooled_out = attention_out.mean(dim=1)

        # Pass the pooled output to the classifier
        return self.fc_out(pooled_out)

"""
learnable_embedding.py

This module defines a PyTorch class for learned positional embeddings.
In transformer models, positional embeddings are crucial because transformers
process tokens in parallel, which means they lack a built-in understanding of sequence order. 
This implementation combines token embeddings (what the word is) with position embeddings (where the word is).
"""

import torch
import torch.nn as nn

class LearnedPositionEmbedding(nn.Module):
    """
    A PyTorch module that combines learnable token embeddings and learnable position embeddings.
    Each token in the input gets represented not only by its identity (via token embedding)
    but also by its position in the sequence (via position embedding).
    """

    def __init__(self, vocab_size, embed_dim, max_seq_len):
        """
        Initializes the embedding layers.

        Args:
            vocab_size (int): Total number of unique tokens in your vocabulary.
            embed_dim (int): The dimensionality of each embedding vector (e.g., 512).
            max_seq_len (int): Maximum length of sequences this model expects (e.g., 100).
        """
        # Call the parent constructor to properly initialize nn.Module internals
        super(LearnedPositionEmbedding, self).__init__()

        # What: This creates an embedding matrix for tokens.
        # Why: We want to convert each token ID into a meaningful dense vector.
        # How: PyTorch creates a (vocab_size x embed_dim) learnable matrix internally.
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # What: This creates an embedding matrix for positions in the sequence.
        # Why: Transformers need explicit position info since they process tokens all at once.
        # How: This layer will learn a unique vector for each position [0 to max_seq_len - 1].
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        """
        Performs the forward pass and returns the combined embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length]
                              where each element is a token ID.

        Returns:
            torch.Tensor: Combined token and position embeddings with shape
                          [batch_size, sequence_length, embed_dim].
        """

        # What: Extract the sequence length from the input tensor.
        # Why: We need to know how many position embeddings to retrieve.
        # How: x.shape is [batch_size, seq_len], so x.size(1) gives seq_len.
        seq_len = x.size(1)

        # What: Create position indices like [0, 1, 2, ..., seq_len - 1]
        # Why: Each token needs a corresponding position ID for embedding lookup.
        # How:
        # - torch.arange(seq_len) creates a 1D tensor of positions.
        # - unsqueeze(0) adds a batch dimension → shape becomes [1, seq_len].
        # - expand_as(x) replicates this row across the batch to match shape of input x.
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device) \
                              .unsqueeze(0).expand_as(x)

        # What: Convert input token IDs to dense vectors.
        # Why: Models can't work with raw token IDs — embeddings provide semantic meaning.
        # How: Looks up token embeddings from the `self.embedding` table.
        token_embeddings = self.embedding(x)  # shape: [batch_size, seq_len, embed_dim]

        # What: Convert position indices to dense vectors.
        # Why: This provides the model with knowledge of where each token is in the sequence.
        # How: Looks up position embeddings from the `self.position_embedding` table.
        position_embeddings = self.position_embedding(position_ids)  # same shape as token_embeddings

        # What: Combine token meaning with position context.
        # Why: The model needs both — what the token is and where it is.
        # How: Element-wise addition of two tensors of same shape.
        embeddings = token_embeddings + position_embeddings

        # Return the final embeddings that encode both token identity and sequence order.
        return embeddings


# --- Example Usage ---

if __name__ == "__main__":
    vocab_size = 1000     # Total number of tokens in the vocabulary
    embed_dim = 512       # Dimensionality of each embedding vector
    max_seq_len = 100     # Maximum expected length of an input sequence

    # What: Instantiate the embedding model.
    # Why: We want to create a module that can map token IDs + positions to rich vector embeddings.
    model = LearnedPositionEmbedding(vocab_size=vocab_size,
                                     embed_dim=embed_dim,
                                     max_seq_len=max_seq_len)

    # What: Create a batch of 2 dummy sequences, each with 10 tokens.
    # Why: This simulates input to the model.
    # How: torch.randint creates random token IDs between 0 and vocab_size - 1.
    dummy_input = torch.randint(0, vocab_size, (2, 10))  # shape: [batch_size=2, seq_len=10]

    # What: Pass the dummy input through the model to generate embeddings.
    # Why: To test and see the combined token + position embeddings.
    output = model(dummy_input)  # shape: [2, 10, 512]

    # What: Print the shape of the output tensor.
    # Why: To verify that we get an embedding per token per sequence.
    print(output.shape)  # Result: torch.Size([2, 10, 512])

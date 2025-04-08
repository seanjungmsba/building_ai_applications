"""
sinusoidal_embeddings.py

This module defines a PyTorch class for fixed positional encodings using sinusoidal functions.
Unlike learnable embeddings, these encodings are computed using a mathematical formula and
are not updated during training.

They allow the model to infer both absolute and relative positions in a sequence without
needing to learn them from data, which improves generalization to longer sequences.
"""

import math
import torch
import torch.nn as nn

class FixedPositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in the original Transformer paper.
    These encodings are fixed (not learned) and encode token positions using sine and cosine
    functions at different frequencies.
    """

    def __init__(self, embed_dim, max_seq_len):
        """
        Initializes the positional encoding matrix using sine and cosine waves.

        Args:
            embed_dim (int): Dimensionality of each positional embedding vector (same as token embeddings).
            max_seq_len (int): Maximum number of positions supported (e.g., longest sentence expected).
        """
        # Initialize the nn.Module base class
        super(FixedPositionalEncoding, self).__init__()

        # What: Create a zero-filled matrix of shape [max_seq_len, embed_dim]
        # Why: This will be filled with sinusoidal values for each position/dimension.
        # How: Each row corresponds to a position, each column to a dimension of the embedding.
        position_enc = torch.zeros(max_seq_len, embed_dim)

        # What: Create a tensor of positions [0, 1, ..., max_seq_len - 1] shaped as a column vector.
        # Why: We want to apply sine and cosine functions to each position across multiple dimensions.
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # shape: [max_seq_len, 1]

        # What: Compute the denominator for the frequency of sine/cosine.
        # Why: The exponential term controls how quickly the wave oscillates in each dimension.
        # How:
        #   - torch.arange(0, embed_dim, 2): only even dimensions
        #   - math.log(10000.0) / embed_dim: ensures the waves span a range of frequencies
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        # What: Fill even-numbered dimensions with sine values.
        # Why: Sine is used to encode one half of the dimensions.
        # How: Broadcasting multiplies [pos, 1] x [1, d] → [pos, d]
        position_enc[:, 0::2] = torch.sin(position * div_term)

        # What: Fill odd-numbered dimensions with cosine values.
        # Why: Cosine encodes the other half of the dimensions.
        # How: Same broadcasting mechanism as above.
        position_enc[:, 1::2] = torch.cos(position * div_term)

        # What: Register the resulting tensor as a buffer.
        # Why:
        #   - Buffers are persistent and part of the model state.
        #   - But unlike parameters, they are not trainable (won't get gradients).
        # How: We add a batch dimension (unsqueeze(0)) → shape becomes [1, max_seq_len, embed_dim]
        self.register_buffer('positional_encoding', position_enc.unsqueeze(0))

    def forward(self, x):
        """
        Retrieves the positional encodings that match the length of the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].

        Returns:
            torch.Tensor: Positional encoding of shape [1, seq_len, embed_dim], to be added to input.
        """
        # What: Extract sequence length from input
        # Why: We only need positional encodings for the actual input length, not full max_seq_len.
        seq_len = x.size(1)

        # What: Return the corresponding portion of the fixed encoding
        # Why: Prevents using more positional encodings than needed
        # How: Slicing the precomputed matrix to match the input length
        return self.positional_encoding[:, :seq_len, :]


# --- Example Usage ---

if __name__ == "__main__":
    embed_dim = 512     # Number of embedding dimensions (must match token embeddings)
    max_seq_len = 100   # Maximum expected sequence length

    # What: Create the fixed positional encoding module
    # Why: Used to provide sequence position information to a transformer model
    pos_encoding = FixedPositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)

    # What: Create a dummy input to simulate transformer input
    # Why: In practice, this would be the output of a token embedding layer
    dummy_input = torch.zeros(2, 10, embed_dim)  # shape: [batch_size=2, seq_len=10, embed_dim=512]

    # What: Add positional encodings to the dummy input
    # Why: This is how position information is injected into transformer input
    output: torch.Tensor = dummy_input + pos_encoding(dummy_input)

    # What: Print the shape to verify it matches input
    # Why: To confirm the output is ready to be passed into the transformer encoder
    print(output.shape)  # Result: torch.Size([2, 10, 512])

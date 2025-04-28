"""
GPT Inference Script - Detailed Version

This script:
- Initializes a small GPT model.
- Prepares a simple input sequence (like "the brown fox").
- Runs inference to predict the next token.
- Displays top-k most likely next tokens based on model output.

Focus:
- Loading a GPT model (decoder-only transformer).
- Embedding input text.
- Computing next-token probability distribution via softmax.
"""

import torch
import torch.nn.functional as F

# ================================
# 🔧 Custom Module
# ================================
# Import the GPT model architecture (defined separately)
from gpt_model import GPT

# ================================
# ⚙️ Define vocabulary and model parameters
# ================================

# Vocabulary list (toy example)
# Each word has a corresponding index.
vocab = ['<pad>', '<unk>', 'the', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']

# Size of the vocabulary (number of unique tokens)
vocab_size = len(vocab)

# Transformer configuration
max_seq_len = 10   # Maximum number of tokens per input sequence
embed_dim = 32     # Embedding size (dimension of token vectors)
num_heads = 2      # Number of self-attention heads (divides embedding space)
num_layers = 2     # Number of stacked decoder layers
dropout = 0.1      # Dropout rate for regularization

# ================================
# 📚 Create word-to-index mapping
# ================================
# Create a dictionary mapping each word to a unique integer ID
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# ================================
# 🧠 Initialize the GPT model
# ================================
# Instantiate the GPT model with the defined parameters
model = GPT(vocab_size, max_seq_len, embed_dim, num_heads, num_layers, dropout)

# ================================
# ✏️ Prepare the input sequence
# ================================

# Example input text
input_sequence = "the brown fox"

# Split the sentence into tokens (words)
input_tokens = input_sequence.split()

# Convert tokens to corresponding integer indices (default to <unk> if missing)
input_indices = [word_to_idx.get(token, word_to_idx['<unk>']) for token in input_tokens]

# Convert list to tensor and add batch dimension (batch_size = 1)
input_tensor = torch.tensor(input_indices).unsqueeze(0)  # Shape: [1, seq_len]

# ================================
# 🧪 Inference
# ================================

# Put the model in evaluation mode (disables dropout and uses running stats for LayerNorm)
model.eval()

# Disable gradient computation for faster inference
with torch.no_grad():
    # Forward pass the input tensor through GPT
    logits = model(input_tensor)  # Shape: [batch_size, seq_len, vocab_size]

    # Focus only on the logits for the **last token** (predict next word)
    last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]

    # Apply softmax to transform logits into probabilities
    probabilities = F.softmax(last_token_logits, dim=-1)

# ================================
# 📝 Post-processing: Show top-k predictions
# ================================

# Create reverse mapping: index to word
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Choose top-k most probable next words
top_k = 5
top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

# Convert indices to word strings
top_k_words = [idx_to_word[idx.item()] for idx in top_k_indices]

# ================================
# 📢 Display Results
# ================================

# Show the input sequence
print(f"Input Sequence: '{input_sequence}'")

# Show the top predicted next tokens
print("Top predictions for the next word:")
for word, prob in zip(top_k_words, top_k_probs):
    print(f"  {word}: {prob.item():.4f}")

"""
Example Output:
------------------
Input Sequence: 'the brown fox'
Top predictions for the next word:
  over: 0.1800
  the: 0.1549
  dog: 0.1430
  <pad>: 0.1244
  lazy: 0.1119
"""

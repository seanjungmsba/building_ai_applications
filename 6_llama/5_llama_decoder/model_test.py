import torch
import torch.nn.functional as F
from model import Llama  # Import the Llama model defined in model.py

# ----------------------------
# 🔧 Model & Token Configuration
# ----------------------------

# Vocabulary of known tokens used by the tokenizer and model
vocab = ['<pad>', '<unk>', 'The', 'brown', 'fox', 'crosses', 'the', 'road']
vocab_size = len(vocab)         # Number of unique tokens
max_seq_len = 10                # Maximum sequence length model will support
embed_dim = 32                  # Dimensionality of each token embedding
num_heads = 4                   # Number of attention heads in self-attention
num_layers = 2                  # Number of decoder layers (stacked blocks)
num_kv_heads = 4                # Number of key-value heads for Grouped Query Attention (GQA)
dropout = 0.1                   # Dropout rate for regularization

# ----------------------------
# 🔠 Vocabulary Mapping
# ----------------------------

# Create dictionaries to convert between words and indices
word_to_idx = {word: idx for idx, word in enumerate(vocab)}  # word → index
idx_to_word = {idx: word for word, idx in word_to_idx.items()}  # index → word

# ----------------------------
# 🧠 Initialize LLaMA-like Model
# ----------------------------

# Instantiate the model using specified configuration
model = Llama(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    num_kv_heads=num_kv_heads,
    dropout=dropout
)

model.eval()  # Set model to evaluation mode (turns off dropout)

# ----------------------------
# ✍️ Prepare Input Tokens
# ----------------------------

# Define a natural language input sentence
input_sequence = "The brown fox"

# Tokenize the sentence (split into words)
input_tokens = input_sequence.split()  # → ['The', 'brown', 'fox']

# Convert tokens to indices using vocabulary map
# Use <unk> index for any word not in vocab
input_indices = [word_to_idx.get(token, word_to_idx['<unk>']) for token in input_tokens]

# Convert to PyTorch tensor and add batch dimension → shape: (1, seq_len)
input_tensor = torch.tensor(input_indices).unsqueeze(0)

# ----------------------------
# 🔄 Forward Pass through Model
# ----------------------------

with torch.no_grad():  # Disable gradient tracking for inference
    logits = model(input_tensor)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Extract logits (pre-softmax scores) for the last token in the sequence
    last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

    # Apply softmax to get predicted probability distribution over vocabulary
    probabilities = F.softmax(last_token_logits, dim=-1)

# ----------------------------
# 🔍 Top-K Predictions
# ----------------------------

top_k = 5  # Number of most probable tokens to return

# Extract the top K token indices and their corresponding probabilities
top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

# Convert top-k token indices back to readable words
top_k_words = [idx_to_word[idx.item()] for idx in top_k_indices]

# ----------------------------
# 🖨️ Print Results
# ----------------------------
print(f"Input Sequence: '{input_sequence}'")
'''
Input Sequence: 'The brown fox'
'''

print("Top predictions for the next word:")
for word, prob in zip(top_k_words, top_k_probs):
    print(f"  {word}: {prob:.4f}")
'''
Top predictions for the next word:
  The: 0.1762
  crosses: 0.1696
  <unk>: 0.1524
  the: 0.1426
  <pad>: 0.1249
'''

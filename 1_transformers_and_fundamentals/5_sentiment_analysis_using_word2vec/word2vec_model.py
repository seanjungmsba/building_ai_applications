import numpy as np
from os import getcwd
from torch import float32, tensor, Tensor
from nltk import word_tokenize
from gensim.models import KeyedVectors

# ---------------------------------------------
# Load the pretrained Word2Vec model using Gensim
# ---------------------------------------------
# - This model was trained on Google News (100B words)
# - It contains 3M words mapped to 300-dimensional vectors
# - The binary file must be downloaded and present in the given path
model_path = getcwd() + "\\1_transformers_and_fundamentals\\resources\\googlenews-vectors-negative300.bin"

# Load the model in binary format using Gensim
word2vec = KeyedVectors.load_word2vec_format(fname=model_path, binary=True)

# -----------------------------------------------------------
# Function: preprocess_sentence
# -----------------------------------------------------------
# Description:
# Given a sentence, tokenize it and convert each token into
# its corresponding Word2Vec embedding (300 dimensions).
# If a word is not found in the vocabulary, use a zero vector.
# Returns a 3D tensor of shape [1, sequence_len, embed_dim].

def preprocess_sentence(sentence: str, word2vec: KeyedVectors, embed_size: int = 300) -> Tensor:

    # Step 1: Tokenize the sentence into individual words
    # Example: "I love coffee" -> ['I', 'love', 'coffee']
    tokens = word_tokenize(text=sentence)

    # Step 2: For each token, retrieve its embedding from Word2Vec
    # If the token is not found in the vocab, use a zero vector
    embeddings = [
        word2vec[token] if token in word2vec else np.zeros(embed_size)
        for token in tokens
    ]

    # Step 3: Convert the list of embeddings into a PyTorch tensor
    # Shape will be [sequence_len, embed_dim] initially
    # unsqueeze(0) adds a batch dimension -> [1, sequence_len, embed_dim]
    return tensor(embeddings, dtype=float32).unsqueeze(0)

# Example usage
sentence = "I love coffee"
output_tensor = preprocess_sentence(sentence, word2vec)
print(output_tensor.shape)  # torch.Size([1, 3, 300])

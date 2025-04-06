import torch
from nltk import word_tokenize

# Function - tokenize_sentence
def tokenize_sentence(sentence: str) -> list[str]:
    # Control Flow
    if '.' in sentence:
        sentence = sentence.replace('.', '')
    else:
        raise ValueError("The input text is not a sentence.")

    return word_tokenize(sentence)

# Function - Create a vocabulary of words
def create_word_vocab(tokens: list[str]) -> dict:
    # Create a dictionary
    return {token : idx for idx, token in enumerate(set(tokens))}

# Function - Generate an embedding
def generate_embedding(vocab: dict, tokens: list[str], embedding_dim: int = 5) -> torch.Tensor:
    # Get the token indices from the tokens
    token_indices = [vocab[token] for token in tokens]

    # Build an input tensor based on the token indices
    input_tensor = torch.tensor(token_indices)

    # Build the embedding layer
    embedding_layer = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)

    # Create the encodings
    embedded_output: torch.Tensor = embedding_layer(input_tensor)

    print(f"Word Embeddings: {embedded_output}")
    print(f"Shape of the embeddings: {embedded_output.shape}")
    
    '''
    Word Embeddings: tensor([[-0.5711,  0.0602,  0.1258, -1.5879,  1.2770],
        [ 0.3306,  1.8695, -0.1848, -0.1870,  0.8795],
        [-2.7947, -0.3038,  0.5011, -0.2378,  1.2127],
        [-0.5069,  0.8572,  0.9240,  0.3406, -0.1319],
        [ 2.7054, -2.0263,  0.9181, -0.9775,  1.5310]],
       grad_fn=<EmbeddingBackward0>)
    
    Shape of the embeddings: torch.Size([5, 5])
    '''
    return embedded_output

# Define a sentence
sentence = 'Deep learning models are powerful.'

# Get our tokens
tokens = tokenize_sentence(sentence=sentence)

# Create a vocabulary
vocab = create_word_vocab(tokens=tokens)

# Generate the embeddings
embedding_output = generate_embedding(
    vocab=vocab,
    tokens=tokens,
    embedding_dim=5
)
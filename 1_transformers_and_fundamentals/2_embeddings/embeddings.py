import torch
from nltk import word_tokenize

def tokenize_sentence(sentence: str) -> list[str]:
    """
    Tokenizes a given sentence into words using NLTK's word_tokenize.

    Args:
        sentence (str): A string containing a single sentence. Must end with a period.

    Returns:
        list[str]: A list of tokens (words) extracted from the sentence.

    Raises:
        ValueError: If the input string does not contain a period, indicating it may not be a full sentence.
    """
    # Check that the input is likely a sentence
    if '.' in sentence:
        sentence = sentence.replace('.', '')  # Remove the period
    else:
        raise ValueError("The input text is not a sentence.")

    return word_tokenize(sentence)


def create_word_vocab(tokens: list[str]) -> dict:
    """
    Creates a vocabulary dictionary that maps each unique word to a unique integer index.

    Args:
        tokens (list[str]): A list of tokenized words.

    Returns:
        dict: A dictionary where keys are unique words and values are integer indices.
    """
    return {token: idx for idx, token in enumerate(set(tokens))}


def generate_embedding(vocab: dict, tokens: list[str], embedding_dim: int = 5) -> torch.Tensor:
    """
    Generates a PyTorch embedding tensor for the input tokens using a randomly initialized embedding layer.

    Args:
        vocab (dict): A word-to-index vocabulary mapping.
        tokens (list[str]): A list of tokenized words from the input sentence.
        embedding_dim (int): The dimensionality of the embedding vector for each token.

    Returns:
        torch.Tensor: A tensor of shape (len(tokens), embedding_dim) representing word embeddings.
    
    Notes:
        - The embeddings are randomly initialized and not trained.
        - Useful for testing or illustrating how embeddings work.
    """
    # Convert each token into its corresponding index
    token_indices = [vocab[token] for token in tokens]

    # Convert to a PyTorch tensor
    input_tensor = torch.tensor(token_indices)

    # Create an embedding layer with random weights
    embedding_layer = torch.nn.Embedding(
        num_embeddings=len(vocab), 
        embedding_dim=embedding_dim
    )

    # Generate embeddings
    embedded_output: torch.Tensor = embedding_layer(input_tensor)

    print(f"Word Embeddings: {embedded_output}")
    '''
    Word Embeddings: tensor([[-0.9760, -0.2332, -0.1486, -0.0752, -0.5735],
        [-2.4092, -0.2145,  0.7495,  0.4592, -0.0860],
        [ 0.1322, -0.6489, -0.2078,  0.6670,  0.9516],
        [-0.4097,  0.6479,  1.2487, -0.0555, -1.0920],
        [ 0.1869, -1.6508, -0.3017, -2.1539,  0.4914]],
       grad_fn=<EmbeddingBackward0>)
    '''
    print(f"Shape of the embeddings: {embedded_output.shape}") # Shape of the embeddings: torch.Size([5, 5])
    
    return embedded_output


# Example usage:
if __name__ == "__main__":
    # Define a sample sentence
    sentence = 'Deep learning models are powerful.'

    # Step 1: Tokenize the sentence
    tokens = tokenize_sentence(sentence=sentence)

    # Step 2: Create a vocabulary mapping
    vocab = create_word_vocab(tokens=tokens)

    # Step 3: Generate embeddings for the tokens
    embedding_output = generate_embedding(
        vocab=vocab,
        tokens=tokens,
        embedding_dim=5
    )

import json
import re
import unicodedata
from requests import get
from collections import defaultdict
from nltk import sent_tokenize

# -----------------------------------------
# Trie Node - Used in WordPiece Token Matching
# -----------------------------------------
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)  # Dictionary to store child characters
        self.is_end = False                    # Flag to indicate end of a valid token
        self.token = None                      # Store full token when it's a valid word


# -----------------------------------------
# Trie Data Structure
# -----------------------------------------
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token: str):
        """
        Insert a token into the trie.
        """
        node = self.root
        for char in token:
            node = node.children[char]
        node.is_end = True
        node.token = token

    def longest_prefix_match(self, word: str):
        """
        Finds the longest prefix match for a given word in the trie.
        Used to identify valid subwords during WordPiece tokenization.
        """
        node = self.root
        longest_match = None

        for i, char in enumerate(word):
            if char in node.children:
                node = node.children[char]
                if node.is_end:
                    longest_match = node.token
            else:
                break

        return longest_match


# -----------------------------------------
# Custom BERT Tokenizer
# -----------------------------------------
class BERTTokenizer:
    def __init__(self, vocab_file_url: str):
        """
        Initializes the tokenizer by downloading the vocabulary file
        and building lookup dictionaries and trie for subword matching.
        """
        try:
            # Download vocabulary from the given URL
            response: str = get(vocab_file_url).text
            self.vocab: list[str] = json.loads(response)  # Load vocab as list of strings
        except Exception as e:
            raise ValueError(f"Error encountered: {e}. \n Reminder: Vocabulary file must be a JSON file.")

        # Create two dictionaries: token -> ID and ID -> token
        self.token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token: dict[int, str] = {idx: token for token, idx in self.token_to_id.items()}

        # Build trie for WordPiece matching
        self.trie = Trie()
        for token in self.vocab:
            self.trie.insert(token)

    def clean_text(self, text: str) -> str:
        """
        Normalize and clean text for consistent tokenization.
        Applies unicode normalization and whitespace collapsing.
        """
        text = unicodedata.normalize("NFKC", text).lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def wordpiece_tokenization(self, word: str) -> list[str]:
        """
        Tokenize a word into subwords using the longest prefix match strategy.
        Falls back to [UNK] if no valid subword found.
        """
        subwords = []
        while len(word) > 0:
            subword = self.trie.longest_prefix_match(word)
            if subword:
                subwords.append(subword)
                word = word[len(subword):]
            else:
                subwords.append("[UNK]")
                break
        return subwords

    def sentence_tokenize(self, text: str) -> list[str]:
        """
        Tokenizes a full paragraph into BERT-friendly tokens.
        Includes [CLS] and [SEP] tokens for each sentence.
        """
        text = self.clean_text(text)
        sentences = sent_tokenize(text)

        # Start token list with [CLS]
        tokens = ["[CLS]"]

        for sentence in sentences:
            words = sentence.split()
            subwords = [sub for word in words for sub in self.wordpiece_tokenization(word)]
            tokens.extend(subwords + ["[SEP]"])

        return tokens

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """
        Converts a list of tokens into their corresponding vocabulary IDs.
        Unknown tokens are mapped to [UNK].
        """
        return [self.token_to_id.get(token, self.token_to_id["[UNK]"]) for token in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """
        Converts a list of IDs back into tokens.
        Unknown IDs are mapped to [UNK].
        """
        return [self.id_to_token.get(id, "[UNK]") for id in ids]
    
# -----------------------------------------
# Example Usage
# -----------------------------------------
if __name__ == "__main__":
    # URL to JSON vocabulary file (replace with actual link if needed)
    vocab_url = "https://data-engineer-academy.s3.us-east-1.amazonaws.com/ai-course/assets/section-two/bert_vocab.json"

    # Initialize tokenizer
    tokenizer = BERTTokenizer(vocab_file_url=vocab_url)

    # Sample input text
    text = "Transformers are revolutionizing NLP applications."

    # Tokenize the input text into BERT tokens
    tokens = tokenizer.sentence_tokenize(text)
    print("Tokens:", tokens) # ['[CLS]', 'transformers', 'are', 'revolution', 'i', 'z', 'ing', 'nl', 'p', 'applications', '.', '[SEP]']

    # Convert tokens to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Token IDs:", token_ids) # [101, 19081, 2024, 4329, 1045, 1062, 13749, 17953, 1052, 5097, 1012, 102]

    # Convert IDs back to tokens
    recovered_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print("Recovered Tokens:", recovered_tokens) # ['[CLS]', 'transformers', 'are', 'revolution', 'i', 'z', 'ing', 'nl', 'p', 'applications', '.', '[SEP]']

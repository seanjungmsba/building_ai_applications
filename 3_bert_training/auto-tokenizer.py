from transformers import AutoTokenizer

# -------------------------------------------------------
# 📦 Load Pretrained BERT Tokenizer
# -------------------------------------------------------

# Load the tokenizer associated with 'bert-base-uncased'.
# This tokenizer:
# - Uses WordPiece tokenization
# - Lowercases input (because it's 'uncased')
# - Adds special tokens like [CLS] and [SEP]
# - Has a fixed vocabulary of ~30k tokens

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# -------------------------------------------------------
# ✍️ Input Sentences
# -------------------------------------------------------

# Define a list of sentences to tokenize. These could be part of a document or batch input.
sentences = [
    "Transformers are revolutionizing NLP.",
    "BERT is one of the most popular transformer models.",
    "Tokenization helps break down text into meaningful units."
]

# -------------------------------------------------------
# 🔄 Tokenization with Batch Encoding
# -------------------------------------------------------

# Tokenize the list of sentences using the tokenizer.
# The arguments include:
# - padding=True: Pads all sentences to the same length
# - truncation=True: Truncates longer sentences to the model's max length (default 512)
# - return_tensors='pt': Returns results as PyTorch tensors

tokenized_sentences = tokenizer(
    text=sentences,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# -------------------------------------------------------
# 📊 Inspect Tokenized Output
# -------------------------------------------------------

# The output is a dictionary with:
# - input_ids: token IDs padded to same length
# - attention_mask: 1 for real tokens, 0 for padding
# - token_type_ids (optional for single sentence tasks)

print("\n📦 Tokenized Tensor Dictionary:")
print(tokenized_sentences)

'''
{'input_ids': 
        tensor([[  101, 19081,  2024,  4329,  6026, 17953,  2361,  1012,   102,     0,
             0,     0,     0],
        [  101, 14324,  2003,  2028,  1997,  1996,  2087,  2759, 10938,  2121,
          4275,  1012,   102],
        [  101, 19204,  3989,  7126,  3338,  2091,  3793,  2046, 15902,  3197,
          1012,   102,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [  101, 14324,  2003,  2028,  1997,  1996,  2087,  2759, 10938,  2121,
          4275,  1012,   102],
        [  101, 19204,  3989,  7126,  3338,  2091,  3793,  2046, 15902,  3197,
          1012,   102,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [  101, 19204,  3989,  7126,  3338,  2091,  3793,  2046, 15902,  3197,
          1012,   102,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])}
'''

# -------------------------------------------------------
# 🔍 Convert IDs Back to Human-Readable Tokens
# -------------------------------------------------------

# For demonstration, decode the first tokenized sentence back to readable tokens
# This helps you inspect how the sentence was split into subwords

first_input_ids = tokenized_sentences['input_ids'][0]  # Select first sentence

tokens = tokenizer.convert_ids_to_tokens(first_input_ids)

print("\n🔤 Tokens for First Sentence:")
print(tokens) # ['[CLS]', 'transformers', 'are', 'revolution', '##izing', 'nl', '##p', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

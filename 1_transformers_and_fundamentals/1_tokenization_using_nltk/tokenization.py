import nltk

# This downloads the correct tokenizer models
nltk.download('punkt', quiet=True)

# Example sentence
sentence = "The brown fox crosses the road."

# Leverage a method from nltk called word_tokenize()
tokens = nltk.word_tokenize(sentence)

print("Word Tokenizer Output: ")
print(tokens) # ['The', 'brown', 'fox', 'crosses', 'the', 'road', '.']

# Group of sentences
sentences = "The brown fox crossed the road. A man saw the fox from the other side of a bridge and started approaching it."

tokens = nltk.sent_tokenize(sentences)

print("Sentence Tokenizer Output: ") # ['The brown fox crossed the road.', 'A man saw the fox from the other side of a bridge and started approaching it.']
print(tokens)
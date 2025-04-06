import nltk

# Example sentence
sentence = "The brown fox crosses the road."

# Leverage a method from nltk called word_tokenize()
tokens = nltk.word_tokenize(sentence)

print("Word Tokenizer Output: ")
print(tokens)

# Group of sentences
sentences = "The brown fox crossed the road. A man saw the fox from the other side of a bridge and started approaching it."

tokens = nltk.sent_tokenize(sentences)

print("Sentence Tokenizer Output: ")
print(tokens)
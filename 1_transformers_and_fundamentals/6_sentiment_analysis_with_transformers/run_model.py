import torch
import os
import sys

# Add the path to the Word2Vec model directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '5_sentiment_analysis_using_word2vec')))

# Import the transformer-based sentiment model
from sentiment_analysis_model import SentimentAnalysisModel

# Import:
# - `word2vec`: the pretrained Google News model loaded with Gensim
# - `preprocess_sentence`: converts raw text into Word2Vec-based tensor
from word2vec_model import word2vec, preprocess_sentence

# ---------------------------------------------------------
# 🧠 Model Initialization
# ---------------------------------------------------------

# Initialize the sentiment classifier model
# - embed_size: 300 (dimensionality of Word2Vec vectors)
# - heads: 6 (number of attention heads)
# - num_classes: 2 (binary classification: positive vs negative)
model = SentimentAnalysisModel(embed_size=300, heads=6, num_classes=2)

# ---------------------------------------------------------
# ✍️ Input Sentence
# ---------------------------------------------------------

# Provide an input sentence for inference
sentence = "I really like this coffee. I think it is great!"

# ---------------------------------------------------------
# 🔄 Preprocess Sentence into Embedding Tensor
# ---------------------------------------------------------

# Step 1: Tokenize and vectorize the sentence using Word2Vec
# - Output shape: [1, sequence_length, 300]
# - Handles out-of-vocabulary words using zero vectors
input_tensor = preprocess_sentence(sentence=sentence, word2vec=word2vec)

# ---------------------------------------------------------
# 🤖 Run Inference Through the Model
# ---------------------------------------------------------

# Step 2: Pass the sentence tensor through the sentiment model
# - Model applies self-attention and pooling
# - Output is a logit vector with shape [1, 2] (for binary classification)
output = model(input_tensor)

# ---------------------------------------------------------
# 📊 Get Prediction
# ---------------------------------------------------------

# Step 3: Select the class with the highest score
# - dim=1 selects from the two class scores
# - .item() extracts the integer class index from the tensor
predicted_class = torch.argmax(output, dim=1).item()

# ---------------------------------------------------------
# 📝 Print Result
# ---------------------------------------------------------

# Step 4: Convert prediction to human-readable label
# - Class 1 → Positive, Class 0 → Negative
print("Predicted sentiment:", "Positive" if predicted_class == 1 else "Negative")

from flask import Flask, request

# 🔁 Custom summarization utility function
from summarizer_util import generate_summary

'''
📄 Flask Backend API for RoBERTa Text Summarizer

This script launches a simple RESTful API with one endpoint `/summary`
that accepts POST requests and returns a summarized version of input text.
'''

# Initialize Flask app instance
app = Flask(__name__)

# Home route — basic connectivity check
@app.route("/")
def home():
    return "Text Summarization RoBERTa API"

# Main summarization endpoint
@app.route("/summary", methods=['POST'])
def summary():
    """
    Endpoint to receive an article and return its summary.

    Expected Input: JSON object with key 'text'
    Output: JSON response with summary and metadata
    """

    # Extract the request body JSON
    data = request.get_json()

    # Get the article content
    text = data['text']

    # Run the summarization
    summary = generate_summary(article=text)

    # Build the response dictionary
    response = {
        'input_text' : text,
        'summary' : summary,
        'model_name' : 'roberta2roberta',
        'model_provider' : 'huggingface'
    }

    return response

# Entry point — launch the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

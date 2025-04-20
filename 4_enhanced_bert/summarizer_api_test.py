from requests import get, post

'''
📄 Test script to validate Flask API endpoint functionality.

This sends a POST request to `/summary` and prints both raw and summarized text.
'''

# Sample article hosted in S3
url = "https://data-engineer-academy.s3.us-east-1.amazonaws.com/ai-course/assets/section-two/sample_article.txt"

# Download article content
article = get(url).text

# Print full article to verify
print(article)

# API port and endpoint
API_PORT = 8080
url = f'http://127.0.0.1:{API_PORT}/summary'

# Send POST request to summarization endpoint
api_call = post(url=url, json={'text' : article})

# Print raw HTTP response object
print(api_call)

# Convert to JSON and print response summary
response = api_call.json()
print(response)

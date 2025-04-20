from requests import get

# 🔁 Local summarization logic
from summarizer_util import generate_summary

'''
📄 Standalone script to fetch a sample article and summarize it using a locally loaded RoBERTa model.
'''

# URL to a sample article hosted on S3
url = "https://data-engineer-academy.s3.us-east-1.amazonaws.com/ai-course/assets/section-two/sample_article.txt"

# Fetch the article from the cloud
article = get(url).text

# Print full article for context
print(article)
print('-' * 50)

# Generate summary using local transformer model
summary = generate_summary(article=article)

# Print summary to console
print(summary)

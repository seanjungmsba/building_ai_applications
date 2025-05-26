
"""
qdrant_index.py

This module sets up a local QDrant instance and initializes a collection for vector similarity search.
It assumes QDrant is running locally on the default ports.

Environment variable:
    - OPENAI_API_KEY: for embedding purposes, loaded externally when required

Dependencies:
    - qdrant-client
    - python-dotenv
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models

# QDrant configuration
collection_name = 'openai-vector-collection'

# Connect to QDrant running locally
client = QdrantClient(host="localhost", port=6333)

# Fetch all existing collections
collections = client.get_collections().collections

# Check if the collection already exists
collection_exists = any(collection.name == collection_name for collection in collections)

# Create the collection if it doesn't exist
if not collection_exists:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=3072,  # Dimensionality of vectors (OpenAI's embedding size)
            distance=models.Distance.COSINE  # Use cosine distance for similarity
        )
    )

# Optional: Print collection info for verification
collection_info = client.get_collection(collection_name=collection_name)
print(f"✅ Collection '{collection_name}' is ready.")


"""
chroma_index.py

This module initializes a ChromaDB collection running on localhost using the Chroma Python SDK.
It creates a collection if one doesn't exist and connects to the Chroma HTTP server.

Environment Variable:
    - OPENAI_API_KEY: For embeddings (set in `.env`)

Dependencies:
    - chromadb
    - python-dotenv
"""

import chromadb
from os import environ
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv('.env')

# Connect to Chroma HTTP server
client = chromadb.HttpClient(host="localhost", port=8000)

# Collection name to use
collection_name = 'openai-vector-collection'

# Try to retrieve an existing collection or create a new one
try:
    collection = client.get_collection(name=collection_name)
    print(f"✅ Collection '{collection_name}' already exists")
except Exception:
    # Create new collection with cosine distance
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"🆕 Created new collection '{collection_name}'")

# Optional: Print all collections
print("📦 Available collections:")
print(client.list_collections())

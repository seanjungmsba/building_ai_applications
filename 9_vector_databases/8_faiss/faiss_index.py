"""
faiss_index.py

Creates a custom FAISS index using random 3072-dimensional vectors and stores associated
metadata externally. Demonstrates adding vectors and performing filtered similarity search.

Dependencies:
    - faiss-cpu
    - numpy
    - python-dotenv
"""

import os
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv

# Load .env variables (optional for embedding generation)
load_dotenv()

DIM = 3072
INDEX_DIR = "./faiss_data"
os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

# Create or load FAISS index and metadata
if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    print(f"✅ Loaded existing index with {index.ntotal} vectors")
else:
    index = faiss.IndexFlatIP(DIM)
    metadata = {"texts": [], "sources": [], "ids": []}
    print(f"🆕 Created new index for dimension {DIM}")

# Sample documents (in place of embeddings from OpenAI)
texts = [
    "This is a sample document for Faiss",
    "Vector databases are efficient for similarity search",
    "Faiss provides fast approximate nearest neighbor search",
    "Embedding models convert text to vectors",
    "This example demonstrates using Faiss as a vector store",
]
sources = ["demo", "demo", "tutorial", "tutorial", "demo"]
ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

vectors = np.random.random((len(texts), DIM)).astype("float32")
faiss.normalize_L2(vectors)

# Add to index and update metadata
index.add(vectors)
metadata["texts"].extend(texts)
metadata["sources"].extend(sources)
metadata["ids"].extend(ids)

# Save to disk
faiss.write_index(index, INDEX_FILE)
with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f"✅ Added {len(vectors)} vectors. Total: {index.ntotal}")

# Perform a search with optional filter
def search(query_vector, k=2, filter_source="demo"):
    faiss.normalize_L2(query_vector.reshape(1, -1))
    scores, indices = index.search(query_vector.reshape(1, -1), index.ntotal)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata["texts"]):
            continue
        if metadata["sources"][idx] == filter_source:
            results.append((metadata["texts"][idx], metadata["sources"][idx], float(score)))
            if len(results) >= k:
                break
    return results

# Run a query
query_vec = np.random.random(DIM).astype("float32")
print("\n🔍 Filtered Search Results:")
for text, src, sim in search(query_vec):
    print(f"* [SIM={sim:.3f}] {text} [{src}]")

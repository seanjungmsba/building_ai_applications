"""
milvus_index.py

Sets up a Milvus collection locally using PyMilvus. The collection stores OpenAI text embeddings
and supports similarity search via IVF_FLAT index using cosine distance.

Dependencies:
    - pymilvus
    - python-dotenv
"""

from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('.env')

# Connect to local Milvus instance
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# Define collection name
collection_name = 'openai-vector-collection'

# Check if collection exists
if utility.has_collection(collection_name):
    print(f"✅ Collection '{collection_name}' already exists.")
    collection = Collection(collection_name)
    collection.load()
else:
    # Define fields for schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=3072)
    ]

    schema = CollectionSchema(fields=fields, description="OpenAI Embedding Vector Collection")

    # Create collection
    collection = Collection(name=collection_name, schema=schema)

    # Define index for vector field
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    collection.load()
    print(f"🆕 Created new collection '{collection_name}'")

# Print collection stats
stats = collection.stats()
print("📊 Collection Stats:")
print(stats)

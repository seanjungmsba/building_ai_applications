
"""
lancedb_index.py

Initializes a LanceDB vector table for OpenAI embeddings. LanceDB is embedded,
so it does not require a separate server and stores data locally.

Environment Variable:
    - OPENAI_API_KEY: For accessing OpenAI embeddings

Dependencies:
    - lancedb
    - python-dotenv
"""

import lancedb
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv('.env')

# Create a database directory if it doesn't exist
db_path = "./lancedb_data"
os.makedirs(db_path, exist_ok=True)

# Connect to the LanceDB database
db = lancedb.connect(db_path)

# Define table name
table_name = 'openai-vector-collection'

# Check if table exists
if table_name not in db.table_names():
    # Define schema for the new table
    schema = {
        "vector": lancedb.vector(3072),  # OpenAI embedding dimension
        "text": lancedb.types.string,
        "source": lancedb.types.string,
    }

    # Create the table
    table = db.create_table(table_name, schema=schema)
    print(f"✅ Created new table '{table_name}'")
else:
    # Open existing table
    table = db.open_table(table_name)
    print(f"📁 Table '{table_name}' already exists")

# Print schema
print("📐 Table Schema:")
print(table.schema)

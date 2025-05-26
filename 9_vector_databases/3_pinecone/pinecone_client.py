
"""
pinecone_client.py

This module provides a PineconeClient class to handle initialization and setup
of a Pinecone vector database index using the Pinecone Python SDK.

Ensure you have the following in your `.env` file before using this script:

    PINECONE_API_KEY="your-pinecone-key"

Dependencies:
    - pinecone-client
    - python-dotenv
"""

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from os import environ

class PineconeClient:
    """
    A wrapper class to manage the initialization and configuration of a Pinecone index.
    """

    def __init__(self, index_name="openai-vector-index-v2", dimension=3072):
        """
        Load environment variables and initialize Pinecone client.

        Args:
            index_name (str): Name of the index to use or create.
            dimension (int): Dimensionality of the embedding vectors.
        """
        load_dotenv(".env")  # Load environment variables from .env file
        self.api_key = environ.get("PINECONE_API_KEY")
        self.index_name = index_name
        self.dimension = dimension
        self.pc = Pinecone(api_key=self.api_key)  # Initialize Pinecone client
        self.index = None  # Placeholder for the connected Pinecone index

    def initialize_index(self):
        """
        Create the index if it doesn't exist, or connect to it if it does.

        Returns:
            pinecone.Index: A connected Pinecone index instance.
        """
        # Create the index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",  # Distance metric used
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        # Connect to the index
        self.index = self.pc.Index(name=self.index_name)
        return self.index

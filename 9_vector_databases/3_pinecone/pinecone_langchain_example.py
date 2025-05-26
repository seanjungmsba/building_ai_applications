
"""
pinecone_langchain_example.py

This module demonstrates how to use Pinecone with LangChain for vector-based
document indexing, similarity search, and retrieval.

Dependencies (install via pip install -r requirements.txt):
    - pinecone-client
    - python-dotenv
    - langchain
    - langchain-openai
    - langchain-pinecone
"""

from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone_client import PineconeClient

class PineconeLangChainDemo:
    """
    A demonstration class that shows how to use Pinecone + LangChain to:
    - Add documents to a vector store
    - Perform similarity searches
    - Use the vector store as a retriever
    """

    def __init__(self):
        """
        Initialize the Pinecone index and embedding model, then create a LangChain vector store.
        """
        # Connect to Pinecone index via helper class
        self.index = PineconeClient().initialize_index()

        # Use OpenAI's embedding model for vector encoding
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Wrap the Pinecone index and embedding model into a LangChain vector store
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)

    def add_documents(self):
        """
        Adds a sample list of documents to the Pinecone vector store with different metadata types.
        """
        # Sample documents with labeled metadata sources
        docs = [
            ("I had chocolate chip pancakes and scrambled eggs for breakfast this morning.", "tweet"),
            ("The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.", "news"),
            ("Building an exciting new project with LangChain - come check it out!", "tweet"),
            ("Robbers broke into the city bank and stole $1 million in cash.", "news"),
            ("Wow! That was an amazing movie. I can't wait to see it again.", "tweet"),
            ("Is the new iPhone worth the price? Read this review to find out.", "website"),
            ("The top 10 soccer players in the world right now.", "website"),
            ("LangGraph is the best framework for building stateful, agentic applications!", "tweet"),
            ("The stock market is down 500 points today due to fears of a recession.", "news"),
            ("I have a bad feeling I am going to get deleted :(", "tweet"),
        ]

        # Create LangChain Document objects with metadata
        documents = [Document(page_content=text, metadata={"source": source}) for text, source in docs]
        uuids = [str(uuid4()) for _ in documents]

        # Add documents to the Pinecone vector index
        self.vector_store.add_documents(documents=documents, ids=uuids)

    def run_queries(self):
        """
        Performs similarity search, similarity search with score, and retrieval.
        """
        print("\n🔍 Similarity Search with Filter:")
        results = self.vector_store.similarity_search(
            "LangChain provides abstractions to make working with LLMs easy", k=2, filter={"source": "tweet"}
        )
        for res in results:
            print(f"* {res.page_content} [{res.metadata}]")

        print("\n🧠 Similarity Search with Scores:")
        results = self.vector_store.similarity_search_with_score(
            "Will it be hot tomorrow?", k=1, filter={"source": "news"}
        )
        for res, score in results:
            print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")

        print("\n🔁 Using Retriever:")
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.5},
        )
        retrieved = retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
        print(f"* {retrieved[0].page_content} [{retrieved[0].metadata}]")

def main():
    """
    Entry point to run the Pinecone + LangChain demo.
    """
    demo = PineconeLangChainDemo()
    demo.add_documents()
    demo.run_queries()

if __name__ == "__main__":
    main()

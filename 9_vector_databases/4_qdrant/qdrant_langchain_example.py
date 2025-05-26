
"""
qdrant_langchain_example.py

This script demonstrates how to use QDrant with LangChain:
- Connect to a QDrant collection
- Add documents with OpenAI embeddings
- Perform similarity search and retrieve results

Dependencies:
    - langchain-qdrant
    - langchain-openai
    - langchain-core
    - qdrant-client
"""

from uuid import uuid4
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv  # ✅ NEW
import os                      # ✅ NEW

# Load environment variables from .env before any use of them
load_dotenv()  # ✅ This must come before OpenAIEmbeddings is initialized

# Custom module for QDrant client and collection name
from qdrant_index import client, collection_name

class QdrantLangChainDemo:
    """
    Demonstrates document indexing, similarity search, and retrieval using QDrant + LangChain.
    """

    def __init__(self):
        """
        Initialize the QDrant vector store with OpenAI embeddings.
        """
        print("Loaded OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set. Please check your .env file and load it correctly.")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings
        )

    def add_documents(self):
        """
        Adds a predefined set of documents with source metadata to the vector store.
        """
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
        documents = [Document(page_content=text, metadata={"source": source}) for text, source in docs]
        uuids = [str(uuid4()) for _ in documents]
        self.vector_store.add_documents(documents=documents, ids=uuids)

    def run_queries(self):
        """
        Executes similarity searches with and without scoring, and demonstrates retrieval.
        """
        print("\n🔍 Similarity Search (Top 2):")
        results = self.vector_store.similarity_search(
            "LangChain provides abstractions to make working with LLMs easy", k=2
        )
        for res in results:
            print(f"* {res.page_content} [{res.metadata}]")

        print("\n🧠 Similarity Search with Score:")
        results = self.vector_store.similarity_search_with_score(
            "Will it be hot tomorrow?", k=1
        )
        for res, score in results:
            print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")

        print("\n🔁 Retrieval:")
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.5}
        )
        retrieved = retriever.invoke("Stealing from the bank is a crime")
        print(f"* {retrieved[0].page_content} [{retrieved[0].metadata}]")

def main():
    """
    Entry point for running the demo.
    """
    demo = QdrantLangChainDemo()
    demo.add_documents()
    demo.run_queries()

if __name__ == "__main__":
    main()

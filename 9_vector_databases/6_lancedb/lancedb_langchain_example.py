
"""
lancedb_langchain_example.py

Demonstrates using LanceDB with LangChain for vector search:
- Uses OpenAI embeddings
- Adds documents
- Performs similarity search and retrieval

Dependencies:
    - langchain-community
    - langchain-openai
    - lancedb
"""

from uuid import uuid4
from dotenv import load_dotenv
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Custom LanceDB connection
from lancedb_index import db, table_name

# Load environment
load_dotenv()

class LanceLangChainDemo:
    """
    Demonstration class for adding, searching, and retrieving documents using LanceDB.
    """

    def __init__(self):
        """
        Initialize LangChain LanceDB vector store with OpenAI embeddings.
        """
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = LanceDB.from_documents(
            documents=[],  # Start with empty state
            embedding=self.embeddings,
            connection=db,
            table_name=table_name
        )

    def add_documents(self):
        """
        Insert sample documents with metadata for vector storage.
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
        self.vector_store.add_documents(documents=documents)

    def run_queries(self):
        """
        Run similarity search and score-based document retrieval.
        """
        print("\n🔍 Similarity Search (Top 2):")
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

        print("\n🔁 Retrieval:")
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.5}
        )
        retrieved = retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
        print(f"* {retrieved[0].page_content} [{retrieved[0].metadata}]")

def main():
    """
    Run the LanceDB + LangChain demo
    """
    demo = LanceLangChainDemo()
    demo.add_documents()
    demo.run_queries()

if __name__ == "__main__":
    main()


"""
chroma_langchain_example.py

Demonstrates how to use ChromaDB with LangChain:
- Load OpenAI embeddings
- Add documents to Chroma
- Perform similarity search
- Use as a retriever

Dependencies:
    - langchain-chroma
    - langchain-openai
    - chromadb
"""

from uuid import uuid4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Custom module to get Chroma client and collection
from chroma_index import client, collection_name

# Load .env for OpenAI key
load_dotenv()

class ChromaLangChainDemo:
    """
    Class for demonstrating ChromaDB + LangChain search and retrieval
    """

    def __init__(self):
        """
        Initialize OpenAI embedding model and vector store
        """
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

    def add_documents(self):
        """
        Add example documents with source metadata to the Chroma collection
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
        Perform similarity search, similarity scoring, and retrieval
        """
        print("\n🔍 Similarity Search (Top 2):")
        results = self.vector_store.similarity_search(
            "LangChain provides abstractions to make working with LLMs easy",
            k=2,
            filter={"source": "tweet"}
        )
        for res in results:
            print(f"* {res.page_content} [{res.metadata}]")

        print("\n🧠 Similarity Search with Scores:")
        results = self.vector_store.similarity_search_with_score(
            "Will it be hot tomorrow?", k=1, filter={"source": "news"}
        )
        for res, score in results:
            print(f"* [SIM={score:.3f}] {res.page_content} [{res.metadata}]")

        print("\n🔁 Retrieval via Retriever:")
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.5}
        )
        retrieved = retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
        print(f"* {retrieved[0].page_content} [{retrieved[0].metadata}]")

def main():
    """
    Run the Chroma + LangChain demo
    """
    demo = ChromaLangChainDemo()
    demo.add_documents()
    demo.run_queries()

if __name__ == "__main__":
    main()

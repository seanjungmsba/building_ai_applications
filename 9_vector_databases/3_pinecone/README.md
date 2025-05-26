
# 🌲 Pinecone + LangChain Integration

Pinecone is a fully managed, cloud-native vector database optimized for real-time similarity search. It allows developers to scale **vector search** for millions to billions of embeddings, making it a go-to solution for Retrieval-Augmented Generation (RAG), recommendation systems, and LLM applications.

---

## 🔐 API Key Setup

### Pinecone

1. Sign up at [pinecone.io](https://pinecone.io)
2. Go to your dashboard and access "API Keys"
3. Create a new key, give it a name and permissions
4. Copy and store your key safely

### OpenAI

1. Sign up at [OpenAI Platform](https://platform.openai.com)
2. Navigate to "API Keys"
3. Create and store a new key

### 🔐 .env File

Create a `.env` file in your root directory with:

```env
OPENAI_API_KEY="your-openai-key"
PINECONE_API_KEY="your-pinecone-key"
```

---

## 📦 Installation

```bash
pip install pinecone-client python-dotenv langchain langchain-openai langchain-pinecone
```

---

## 🧠 Pinecone Python SDK

This script connects to Pinecone, creates an index, and initializes it for use.

```python
# file: pinecone_client.py
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from os import environ

class PineconeClient:
    def __init__(self, index_name="openai-vector-index-v2", dimension=3072):
        load_dotenv(".env")
        self.api_key = environ.get("PINECONE_API_KEY")
        self.index_name = index_name
        self.dimension = dimension
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None

    def initialize_index(self):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.pc.Index(name=self.index_name)
        return self.index
```

---

## 🔗 LangChain + Pinecone Vector Store

```python
# file: pinecone_langchain_example.py
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone_client import PineconeClient

class PineconeLangChainDemo:
    def __init__(self):
        self.index = PineconeClient().initialize_index()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)

    def add_documents(self):
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
    demo = PineconeLangChainDemo()
    demo.add_documents()
    demo.run_queries()

if __name__ == "__main__":
    main()
```

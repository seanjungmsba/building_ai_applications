"""
faiss_langchain_example.py

Demonstrates how to use FAISS with LangChain and OpenAI embeddings for semantic vector search.

Dependencies:
    - langchain-openai
    - langchain-community
    - python-dotenv
"""

import os
from uuid import uuid4
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load API key
load_dotenv()

# Create embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Setup FAISS persistence path
SAVE_DIR = "./faiss_langchain"
os.makedirs(SAVE_DIR, exist_ok=True)
INDEX_PATH = os.path.join(SAVE_DIR, "openai-vector-index")

# Create or load FAISS store
if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
    vector_store = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    print(f"✅ Loaded FAISS index from {INDEX_PATH}")
else:
    vector_store = FAISS.from_documents(documents=[], embedding=embeddings)
    print(f"🆕 Created new FAISS index")

# Add documents
documents = [
    Document("I had chocolate chip pancakes and scrambled eggs.", metadata={"source": "tweet"}),
    Document("Tomorrow's weather is cloudy and 62 degrees.", metadata={"source": "news"}),
    Document("LangChain is awesome for building LLM apps.", metadata={"source": "tweet"}),
    Document("Bank robbers stole $1 million in a heist.", metadata={"source": "news"}),
    Document("That movie was amazing. A must-watch!", metadata={"source": "tweet"}),
]

vector_store.add_documents(documents)
vector_store.save_local(INDEX_PATH)

# Run similarity search
print("\n🔍 Top Matches:")
results = vector_store.similarity_search(
    "LangChain provides abstractions for LLMs", k=2, filter={"source": "tweet"}
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

# Search with scores
print("\n🧠 Scored Results:")
results = vector_store.similarity_search_with_score("Will it be hot tomorrow?", k=1)
for doc, score in results:
    print(f"* [SIM={score:.3f}] {doc.page_content} [{doc.metadata}]")

# Use retriever
print("\n🔁 Retrieved:")
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"k": 1, "score_threshold": 0.5}
)
res = retriever.invoke("Breaking into a bank is a federal crime", filter={"source": "news"})
print(f"* {res[0].page_content} [{res[0].metadata}]")

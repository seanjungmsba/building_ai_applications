
# 🧠 ChromaDB + LangChain Vector Search Demo

This project demonstrates how to use **ChromaDB** with **LangChain** and **OpenAI embeddings** to enable vector-based search, similarity scoring, and retrieval.

---

## 🐳 Setting Up ChromaDB with Docker

Create a file called `docker-compose.yml`:

```yaml
version: '3.9'
services:
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma/chroma
    environment:
      - ALLOW_RESET=true
      - ANONYMIZED_TELEMETRY=false
```

Then start Chroma (make sure Docker Desktop is running):

```bash
docker-compose up -d
```

---

## 🔐 .env File

```env
OPENAI_API_KEY=your-openai-api-key
```

---

## 🧪 Installation

```bash
pip install -r requirements.txt
```

---

## 📁 File Overview

```
.
├── chroma_index.py               # Initializes Chroma collection
├── chroma_langchain_example.py   # Vector search & retrieval with LangChain
├── .env                          # OpenAI API key
├── docker-compose.yml            # Local ChromaDB container
├── requirements.txt              # Dependencies
```

---

## 🚀 Running the Demo

```bash
python chroma_langchain_example.py
```

You’ll see:
- Similarity search results
- Search with similarity scores
- Retrieved documents using score thresholding

---

## 🧾 Key Features

- Uses OpenAI’s `text-embedding-3-large` for semantic similarity
- Stores and indexes documents with metadata
- Fast, lightweight, and ideal for local development

---

## ✅ When to Use ChromaDB

- Lightweight applications
- Academic projects and prototypes
- Local, embedded vector databases without external dependencies

---

## 🧠 Compared to Others

| Feature           | ChromaDB       | QDrant        | Pinecone      |
|------------------|----------------|---------------|---------------|
| Open Source       | ✅              | ✅             | ❌             |
| Self-hostable     | ✅              | ✅             | ❌ (cloud)     |
| Filtering         | ⚠️ Limited     | ✅ Advanced    | ⚠️ Basic       |
| Scalability       | ⚠️ Medium      | ✅ High        | ✅ High        |
| Best for          | Dev / local    | Prod / RAG    | Managed scale |

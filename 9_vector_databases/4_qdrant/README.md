
# 🚀 QDrant + LangChain Vector Search Demo

This project demonstrates how to set up and interact with a **QDrant** vector database using the **LangChain** framework and **OpenAI embeddings**. It shows how to index documents, perform similarity search, and implement a retriever using QDrant locally.

---

## 📦 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

Ensure you also have Docker and Docker Compose installed to run QDrant locally.

---

## 🐳 Running QDrant Locally

Create a `docker-compose.yml` file:

```yaml
version: '3.7'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC API
    volumes:
      - ./qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__TELEMETRY_DISABLED=true
```

Start the container (Make sure Docker Desktop is running):

```bash
docker-compose up -
```

Once completed, stop the container via Docker Desktop

---

## 🔐 Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key
```

---

## 📁 File Structure

```
.
├── qdrant_index.py               # Initializes the QDrant collection
├── qdrant_langchain_example.py   # LangChain integration and demo
├── .env                          # Your OpenAI API key
├── docker-compose.yml            # QDrant local setup
├── requirements.txt              # Dependencies
```

---

## 📜 What Each File Does

### `qdrant_index.py`

- Connects to a local QDrant instance
- Checks for an existing collection
- Creates a new collection if one does not exist

### `qdrant_langchain_example.py`

- Uses OpenAI's `text-embedding-3-large` to embed documents
- Adds documents to the QDrant collection
- Performs similarity search and retrieval using LangChain's `QdrantVectorStore`

---

## 🧪 Running the Demo

```bash
python qdrant_langchain_example.py
```

Output will show:
- Top-2 similarity search results
- Similarity scores
- Retriever-based document matching

---

## 📊 Why QDrant?

- ✅ Open Source & Self-hostable
- 🔍 High-performance filtering and vector search
- 🧩 Perfect for local RAG applications

---

## 🔄 QDrant vs Pinecone

| Feature                | QDrant          | Pinecone         |
|------------------------|------------------|------------------|
| Open Source            | ✅ Yes           | ❌ No             |
| Local Deployment       | ✅ Yes           | ❌ Cloud-only     |
| Filtering Capabilities | ✅ Advanced      | ⚠️ Basic          |
| Customization          | ✅ Full Control  | ❌ Limited        |
| Setup Complexity       | ⚠️ Needs Docker  | ✅ Managed        |
| Enterprise Reliability | ⚠️ DIY           | ✅ SLA-backed     |

---

## 🙋‍♂️ When to Choose QDrant?

- You want **control** over your data and infrastructure.
- You prefer **open-source** or need local development/testing.
- You require **complex filtering logic** alongside similarity search.

---

## ✅ Summary

This project helps you:
- Run QDrant locally via Docker
- Store and retrieve documents using OpenAI + LangChain
- Explore Retrieval-Augmented Generation (RAG) workflows using open-source tools

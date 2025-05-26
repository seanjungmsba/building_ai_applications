
# ⚡ LanceDB + LangChain Demo

This project shows how to use **LanceDB**, an embedded vector database, with **LangChain** and **OpenAI embeddings**.

---

## 🚀 Why LanceDB?

- **Embedded**: No separate server needed
- **Fast Cold Starts**: Great for serverless and desktop apps
- **Lightweight & Efficient**: Built on Apache Arrow
- **Open Source**: Flexible and free to use

---

## 🔧 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your `.env`

```env
OPENAI_API_KEY=your-openai-api-key
```

---

## 📂 File Overview

```
.
├── lancedb_index.py              # Initializes embedded LanceDB vector table
├── lancedb_langchain_example.py  # Vector search demo using LangChain + OpenAI
├── .env                          # Your API key for OpenAI
├── requirements.txt              # Dependencies
```

---

## 🧪 Run the Demo

```bash
python lancedb_langchain_example.py
```

You’ll see:
- Top-2 similarity search results
- Scored semantic similarity
- Document retrieval using threshold

---

## 🧠 When to Use LanceDB

| Use Case                        | LanceDB  |
|---------------------------------|----------|
| Lightweight local vector search | ✅        |
| Embedded ML apps (edge/serverless) | ✅    |
| Avoiding external DBs           | ✅        |
| Production RAG at scale         | ❌        |
| Advanced filtering              | ⚠️ Basic |

---

## 🔄 Compared to Others

| Feature           | LanceDB     | ChromaDB     | QDrant       | Pinecone     |
|------------------|-------------|--------------|--------------|--------------|
| Server Required   | ❌ Embedded | ✅ Optional   | ✅ Yes        | ✅ Cloud-only |
| Setup Simplicity  | ✅           | ✅            | ⚠️ Moderate   | ✅ Easy       |
| Scale             | ⚠️ Local     | ⚠️ Mid-size   | ✅ Large      | ✅ Huge       |
| Filters & Search  | ⚠️ Basic     | ✅ Good       | ✅ Rich       | ⚠️ Basic      |
| Cost              | ✅ Free     | ✅ Free       | ✅ Free       | ❌ Usage Fees |

---

## 🙌 Ideal For

- Local RAG demos
- Edge AI applications
- Developer prototyping
- Serverless functions

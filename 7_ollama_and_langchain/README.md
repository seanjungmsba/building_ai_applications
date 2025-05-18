# 🧪 Project 4 – Synthetic Data ETL Pipeline with LLMs

## 📌 Overview

This project demonstrates how to use **Large Language Models (LLMs)** to automate the generation, transformation, and profiling of **synthetic datasets** for ETL (Extract, Transform, Load) pipelines.

With tools like **Langchain**, **Ollama**, and **LangServe**, you’ll programmatically prompt LLMs to generate realistic data structures (e.g., tabular data), normalize them, and run advanced **data profiling** using **Polars**, **Pandas**, and **YData Profiling**.

> ✅ Ideal for: AI-driven data engineering, MLOps, or augmenting analytics pipelines with synthetic data.

---

## 🎯 Project Goals

* Generate synthetic datasets using a local or hosted LLM (via **Ollama**).
* Build a **LangChain**-powered ETL chain that:

  * Sends prompts to the LLM.
  * Parses and structures LLM outputs (e.g., JSON, CSV, dict).
  * Transforms the data using **Polars**/**Pandas**.
  * Profiles the synthetic data using **YData Profiling**.
* Serve the pipeline as an API using **LangServe** (LangChain’s FastAPI wrapper).

---

## 🧠 Why Use LLMs for Synthetic ETL?

| Traditional ETL                             | LLM-Based ETL                                                 |
| ------------------------------------------- | ------------------------------------------------------------- |
| Requires hard-coded data generation rules   | Dynamically generates schema-aware examples                   |
| Limited flexibility in language-based tasks | Can generate documentation, column descriptions, labels, etc. |
| Not human-readable                          | Can explain and profile data in plain English                 |

---

## 🛠️ Technologies Used

| Tool / Library         | Role                                                   |
| ---------------------- | ------------------------------------------------------ |
| 🧠 **LangChain**       | LLM orchestration and chaining                         |
| 💻 **Ollama**          | Lightweight local LLM runner (e.g., LLaMA2, Mistral)   |
| ⚙️ **LangServe**       | Serve chains via FastAPI                               |
| 📊 **Polars**          | High-performance DataFrame engine (for transformation) |
| 🐼 **Pandas**          | General-purpose data manipulation                      |
| 🧬 **YData Profiling** | Automated data profiling and EDA reports               |

---

## 🧬 Example Use Case

1. **Prompt**: *"Generate a table with 100 fake customer support tickets, including timestamps, ticket types, sentiment scores, and priority labels."*
2. **LLM Response**: Unstructured or semi-structured JSON/text response.
3. **Normalization**: Parse and structure with `langchain.output_parsers`, convert to `Polars`/`Pandas`.
4. **Profiling**: Apply `ydata-profiling` to evaluate distributions, nulls, correlations, etc.
5. **API Serving**: LangServe wraps the entire flow as an API for integration in external tools or dashboards.

---

## 🔁 High-Level Pipeline Diagram

![Project Diagram](https://data-engineer-academy.s3.us-east-1.amazonaws.com/ai-course/assets/section-two/diagrams/langchain_ollama_project_diagram.png)

---

## 📦 Future Extensions

* Export cleaned + profiled datasets to Snowflake, PostgreSQL, or S3.
* Use `LangChain agents` to dynamically generate SQL or validation logic.
* Add synthetic NLP/text datasets for sentiment, translation, etc.
* Integrate with `Great Expectations` or `dbt` for test coverage.

---

## 🚀 How to Run

1. Make sure you have `ollama` running locally:

   ```bash
   ollama run mistral
   ```

2. Install dependencies:

   ```bash
   pip install langchain langserve pandas polars ydata-profiling
   ```

3. Run your main LangServe app (typically `main.py` or `app.py`):

   ```bash
   uvicorn app:app --reload
   ```

4. Use the `/generate` endpoint to get synthetic data via LLMs.

---

## 🧠 Prerequisites

* Familiarity with Python, HTTP APIs, and LLM prompt engineering.
* A local or cloud Ollama environment.
* LangChain and FastAPI knowledge is helpful but not required.


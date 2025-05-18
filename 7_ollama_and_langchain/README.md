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

---

## Ollama

### Overview

**Ollama** is an open-source platform that enables you to run large language models (LLMs) directly on your local machine. It’s built with efficiency and usability in mind, making it a compelling alternative to cloud-based model hosting. With Ollama, you can harness the power of models like Meta's Llama 3 and others without relying on an internet connection once the model is downloaded.

Whether you're experimenting with generative AI, building prototypes, or running inference in private environments, Ollama offers a lightweight and streamlined solution.

---

### ✨ Key Features

* **Local-first**: Run powerful LLMs locally without sending data to external servers.
* **Cross-platform**: Supports macOS, Windows, and Linux environments.
* **Model flexibility**: Easily pull and switch between various models from the Ollama model library.
* **CLI-powered**: Simple and scriptable terminal-based usage for developers.

---

### 🔧 Installation & Setup

To get started with Ollama:

1. Visit the official [Ollama website](https://ollama.com/).
2. Download and install the Ollama Desktop application compatible with your operating system.
3. Once installed, open your **Terminal** (or Command Prompt on Windows) and run:

   ```bash
   ollama help
   ```

   This will verify that the CLI is installed and show you the available commands.

---

### 🧠 Running a Specific Model (Example: LLaMA 3.1 8B)

Ollama supports many popular models, including variants of LLaMA, Mistral, and more. Here’s how to run Meta’s LLaMA 3.1 (8B) model locally:

1. Go to the [LLaMA 3.1 8B model page](https://ollama.com/library/llama3.1).

2. Copy the command shown on the right-hand side of the page, or simply run:

   ```bash
   ollama run llama3.1
   ```

3. The CLI will automatically:

   * Download the model weights and store them locally.
   * Initialize the model runtime.
   * Provide you with a terminal-based interface to interact with the model.

This command essentially turns your local machine into an LLM inference server, capable of answering prompts and performing text generation in real-time — all without needing cloud APIs.

---

### 📚 Resources & Documentation

* [Ollama Documentation](https://ollama.com/docs)
* [Model Library](https://ollama.com/library)
* [GitHub Repository](https://github.com/ollama/ollama) (if applicable)

---

### 🧪 Example Use Cases

* Chatbot development with full privacy
* LLM-based data analysis pipelines
* Prompt engineering experimentation
* Local AI agents for automation or research

---

### 💡 Tip

For programmatic interaction, you can pair Ollama with frameworks like **LangChain** or create custom interfaces using the **Ollama API**, which exposes endpoints for developers to integrate LLMs into applications.

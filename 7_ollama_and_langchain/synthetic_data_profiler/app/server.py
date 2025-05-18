"""
Module: api_server

This module launches a FastAPI web server for interacting with a local LLM via LangChain and Ollama.
It exposes endpoints to serve LLM chains and generate synthetic JSON data based on industry-specific prompts.

Endpoints:
- `/` : Redirects to the Swagger UI documentation.
- `/ollama_llama3` : Serves a LangChain-compatible LLaMA 3.1 model endpoint via LangServe.
- `/get_synthetic_data` : Accepts input parameters to generate structured synthetic data using an LLM chain.
"""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_ollama import ChatOllama

# Custom workflow logic (prompt chain + data extraction)
from langchain_prompt_workflow import chain, get_json_data

# Create the FastAPI app instance
app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    """
    Redirects the root path ("/") to the interactive Swagger UI documentation ("/docs").

    Returns
    -------
    RedirectResponse
        A response object that redirects users to the OpenAPI UI for easier exploration.
    """
    return RedirectResponse("/docs")


# ---------------------------------------------
# LLM ROUTE (served using LangServe)
# ---------------------------------------------

# Adds a runnable route at /ollama_llama3 that serves the LLaMA 3.1 model
# This allows you to POST directly to this endpoint to get responses from the model
add_routes(
    app,
    runnable=ChatOllama(model='llama3.1').bind(),  # Instantiate and bind the model
    path='/ollama_llama3'  # Exposed path for this LLM endpoint
)


# ---------------------------------------------
# SYNTHETIC DATA GENERATION ENDPOINT
# ---------------------------------------------

@app.post("/get_synthetic_data")
def get_synthetic_data(industry: str, num_rows: int):
    """
    Generates synthetic dataset using a LangChain pipeline connected to a local Ollama-hosted LLM.

    Parameters
    ----------
    industry : str
        The industry context for which to generate synthetic data (e.g., "Finance", "Healthcare").
    
    num_rows : int
        The desired number of rows in the output dataset.

    Returns
    -------
    dict
        A JSON-formatted synthetic dataset with the requested number of records tailored to the given industry.

    Example Request (POST)
    ----------------------
    curl -X POST "http://localhost:8080/get_synthetic_data?industry=Retail&num_rows=5"
    """
    return get_json_data(
        chain=chain,
        industry=industry,
        num_rows=num_rows
    )


# ---------------------------------------------
# ENTRY POINT
# ---------------------------------------------

if __name__ == "__main__":
    # Launch the FastAPI server on host 0.0.0.0 (accessible externally) at port 8080
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

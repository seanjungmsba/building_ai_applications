"""
Purpose: This script defines a pipeline for generating synthetic datasets using a locally hosted LLM via Ollama.
Pipeline: PromptTemplate → LLM (ChatOllama) → JSON parsing.
Key Feature: It ensures the LLM’s raw response is parsed cleanly even when wrapped in markdown formatting like ```json ... ```
"""

# Import the `loads` function to convert JSON strings into Python dictionaries/lists
from json import loads

# Import LangChain's prompt templating class
from langchain.prompts import PromptTemplate

# Import message class used to handle responses from the language model
from langchain_core.messages import AIMessage

# Import base class for runnables that support serialization
from langchain_core.runnables.base import RunnableSerializable

# Import LangChain's wrapper for running LLMs via Ollama locally
from langchain_ollama import ChatOllama


# ------------------------------
# PROMPT TEMPLATE DEFINITION
# ------------------------------

# Define a simple string template for generating synthetic JSON datasets
example_prompt_template = """
Generate a dataset for the industry: {industry} with exactly {num_rows} rows in JSON format.
"""

# ------------------------------
# LLM (Ollama) SETUP
# ------------------------------

# Initialize the LLM (LLaMA 3.1 model via Ollama) with a creativity temperature of 0.7
llm = ChatOllama(model='llama3.1:8b', temperature=0.7)

# ------------------------------
# PROMPT TEMPLATE INITIALIZATION
# ------------------------------

# Create a LangChain prompt using the template defined above
# It expects two input variables: 'industry' and 'num_rows'
prompt = PromptTemplate(
    template=example_prompt_template,
    input_variables=[
        'industry',
        'num_rows'
    ]
)

# ------------------------------
# CHAIN DEFINITION
# ------------------------------

# This creates a LangChain Expression Language (LCEL) chain
# The output of the prompt is piped into the LLM
chain = prompt | llm


# ------------------------------
# FUNCTION TO GENERATE SYNTHETIC JSON DATA
# ------------------------------

def get_json_data(chain: RunnableSerializable, industry: str, num_rows: int) -> dict:
    """
    Uses the LLM chain to generate a JSON dataset for a given industry with a specified number of rows.
    """

    # Invoke the chain with the input variables (industry and number of rows)
    # This returns an AIMessage object which contains the raw response text
    response: AIMessage = chain.invoke(
        input={
            'industry': industry,
            'num_rows': num_rows
        }
    )

    # Extract the actual response text content from the AIMessage object
    json_response: str = response.content

    # Check if the response is wrapped in markdown-style code blocks (```json ... ```)
    if "```" in json_response:
        # Find the first '[' which should mark the beginning of the JSON array
        start = json_response.find("[")
        # Find the closing ']' to mark the end of the JSON array
        end = json_response.find("]", start)
        # Extract only the JSON array from the response
        actual_json = json_response[start:end+1].strip()
    else:
        # If not wrapped, just use the response as-is
        actual_json = json_response.strip()

    # Convert the JSON string into a Python dictionary/list object
    json_obj = loads(actual_json)

    # Print the result for debugging or visibility
    print(json_obj)

    # Return the parsed JSON object
    return json_obj
 

if __name__ == "__main__":
    # Example industry and number of rows to generate
    industry = "Healthcare"
    num_rows = 5

    # Call the function to get synthetic data
    synthetic_dataset = get_json_data(chain, industry, num_rows)

    # Print the generated dataset nicely
    print("\nGenerated Synthetic Dataset:")
    for idx, row in enumerate(synthetic_dataset, start=1):
        print(f"{idx}: {row}")

    """
    Generated Synthetic Dataset:
        1: {'Name': "St. Michael's Hospital", 'City': 'Toronto', 'State': 'Ontario', 'Employees': 1000, 'Revenue': 50000000}
        2: {'Name': 'Cedars-Sinai Medical Center', 'City': 'Los Angeles', 'State': 'California', 'Employees': 1200, 'Revenue': 60000000}
        3: {'Name': 'Massachusetts General Hospital', 'City': 'Boston', 'State': 'Massachusetts', 'Employees': 1500, 'Revenue': 70000000}
        4: {'Name': 'University of Pennsylvania Health System', 'City': 'Philadelphia', 'State': 'Pennsylvania', 'Employees': 1800, 'Revenue': 80000000}
        5: {'Name': 'NewYork-Presbyterian Hospital', 'City': 'New York City', 'State': 'New York', 'Employees': 2000, 'Revenue': 90000000}
    """
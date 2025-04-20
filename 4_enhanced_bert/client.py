import streamlit as st
from requests import post

'''
📄 Streamlit Frontend Client UI for Text Summarization App

This is a lightweight frontend interface built with Streamlit that sends user input 
to a locally running Flask server (`model_api.py`) and displays the generated summary.

🔧 Usage:
Run the following command to start:
    streamlit run client.py
'''

# Set the page title
st.title("RoBERTa Text Summarizer")

# Input widget for user to type or paste text to summarize
text = st.text_area("Enter text to summarize:")

# Local API port used by Flask backend
API_PORT = 8080

# Button to trigger summarization
if st.button(label='Summarize'):
    # Send the user-entered text to the Flask API via POST request
    response = post(f'http://127.0.0.1:{API_PORT}/summary', json={'text': text})

    # Convert server response into Python dictionary
    response = response.json()

    # Extract the generated summary from the response
    summary = response['summary']

    # Display the summary below the text area
    st.markdown('### Article Summary')
    st.write(summary)

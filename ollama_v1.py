import streamlit as st
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

st.title("Ollama chatbot with Streamlit")

input_text = st.text_input("Hey, whats your query?")
chatbot_llm = OllamaLLM(model="llama3.2:3b")

if input_text:
    response=chatbot_llm.invoke(input_text)
    if st.button("Send"):
        st.write("Response:", response)
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

prompt=ChatPromptTemplate([
    ('system','You are an Helpful AI Asssitant'),
    ('human','Question:{Question}')
])

def generate_resposne(query,llm,temperature):
    model=ChatOllama(model=llm,temperature=temperature)
    parser=StrOutputParser()
    chain=prompt |model|parser

    result=chain.invoke({"Question":query})

    return result

st.title("Chatbot powered bY Ollama")
query=st.text_input('What is your Query?')

st.sidebar.title('settings')

llm=st.sidebar.selectbox("Select the LLM",['llama3.2:3b','tinyllama:latest','qwen3:4b'])

temperature=st.sidebar.slider("Select the Temperature",min_value=0.0,max_value=2.0)

if st.button('Answer'):
    if query:
        response=generate_resposne(query,llm,temperature)
        st.write(response)
    else:
        st.warning('Please Enter Query')

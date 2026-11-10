import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt=ChatPromptTemplate([
    ('system','You are an Helpful AI Asssitant'),
    ('human','Question:{Question}')
])

def generate_resposne(query,llm,temperature,max_tokens,api_key):
    model=ChatOpenAI(model_name=llm,temperature=temperature,max_tokens=max_tokens,openai_api_key=api_key)
    parser=StrOutputParser()
    chain=prompt |model|parser

    result=chain.invoke({"Question":query})

    return result

st.title("Chatbot powered bY OpenAI")
query=st.text_input('What is your Query?')

st.sidebar.title('settings')

api_key=st.sidebar.text_input('Provide the API key here',type='password')

llm=st.sidebar.selectbox("Select the LLM",['gpt-4o-mini','gpt-4.1-mini','gpt-4.1'])

temperature=st.sidebar.slider("Select the Temperature",min_value=0.0,max_value=2.0)

max_tokens=st.sidebar.slider("Select the max tokens",min_value=50,max_value=200)

if st.button('Answer'):
    if query and api_key:
        response=generate_resposne(query,llm,temperature,max_tokens,api_key)
        st.write(response)
    else:
        st.warning('Please Enter Query or API_KEY')

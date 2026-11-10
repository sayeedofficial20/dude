import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(model='qwen/qwen3.8-27b')

prompt=ChatPromptTemplate.from_template('''Answer from the following context only.
Provide the most accurate results based on the context only
{context}

Question: {input}''')

def generate_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader('data')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=69)
        st.session_state.chunks=st.session_state.splitter.split_documents(st.session_state.docs)

        st.session_state.vector_store=FAISS.from_documents(st.session_state.chunks,embedding= st.session_state.embeddings)
        
        st.write('Vector Database is readY!')

st.title('GROQ RAG CHATBOT')
st.write('Please click the button below to ADD embedding of your personal Data')

if st.button('Generate Embeddings'):
    generate_embedding()

user_prompt=st.text_input("Enter the Query to search in Uploaded documnt")

if st.button('Answer'):
    if user_prompt:
        document_context=create_stuff_documents_chain(model,prompt)
        retriever=st.session_state.vector_store.as_retriever()
        retriever_data=create_retrieval_chain(retriever,document_context)
        response=retriever_data.invoke({'input':user_prompt})

        st.write(response['answer'])

        with st.expander('Context from the Document:'):
            if "context" in response:
                for i,doc  in enumerate(response['context']):
                    st.write(doc.page_content)




    # model=ChatGroq(model=llm,temperature=temperature,max_tokens=max_tokens,openai_api_key=api_key)
    # parser=StrOutputParser()
    # chain=prompt |model|parser

    # result=chain.invoke({"Question":query})

#     return result

# st.title("Chatbot powered bY Groq")
# query=st.text_input('What is your Query?')

# st.sidebar.title('settings')

# api_key=st.sidebar.text_input('Provide the API key here',type='password')

# llm=st.sidebar.selectbox("Select the LLM",['gpt-4o-mini','gpt-4.1-mini','gpt-4.1'])

# temperature=st.sidebar.slider("Select the Temperature",min_value=0.0,max_value=2.0)

# max_tokens=st.sidebar.slider("Select the max tokens",min_value=50,max_value=200)

# if st.button('Answer'):
#     if query and api_key:
#         response=generate_resposne(query,llm,temperature,max_tokens,api_key)
#         st.write(response)
#     else:
#         st.warning('Please Enter Query or API_KEY')

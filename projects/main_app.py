import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

st.title('')

api_key=st.text_input("provide with the GROQ API Key",type='password')

if api_key:
    model=ChatGroq(model='qwen/qwen3.8-27b',groq_api_key=api_key)

    session_id=st.text_input("please provide Yout Session ID",value='default-session')

    if 'store' not in st.session_state:
        st.session_state.store={}

    upload_file=st.file_uploader("Uplaod the File",type='pdf')

    if upload_file:
        temp_pdf="./temporary.pdf"
        with open(temp_pdf,'wb') as f:
            f.write(upload_file.getvalue())

        documents=PyPDFLoader(temp_pdf).load()
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
        chunks=splitter.split_documents(documents)
        vector_store=Chroma.from_documents(chunks,embedding=OpenAIEmbeddings(),persist_directory='./chromadb')
        retriever=vector_store.as_retriever()

        contxt_sys_prompt='''Given a chat history and lastest user question,
        rephrase the question to make is standalone,Do not answer,only rephrase.'''

        contxt_prompt=ChatPromptTemplate([
            ('system',contxt_sys_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ])

        history_aware_retriever=create_history_aware_retriever(model,retriever,contxt_prompt)

        sys_prompt='''You are Helpful AI Assistant.For Answering to the Question,Use provided context only.
        If Not Sure,Say I don't know
        context
        {context}
        '''
        qa_prompt=ChatPromptTemplate([
                ('system',sys_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ])
        document_chain=create_stuff_documents_chain(model,qa_prompt)

        retrieve_chain=create_retrieval_chain(history_aware_retriever,document_chain)

        def get_session_history(session_id):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conv_rag_chain=RunnableWithMessageHistory(
            retrieve_chain,
            get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )

        user_query=st.text_input("Ask question about your PDF")

        if user_query:
            session_history=get_session_history(session_id)
            response=conv_rag_chain.invoke({'input':user_query},config={'configurable':{"session_id":session_id}})

            st.subheader('Assistant Answer')
            st.write(response['answer'])

            with st.expander("chat_history"):
                st.write(session_history.messages)

else:
    st.warning('Please Enter Your API to comtinue.....')

import streamlit as st
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
embed_model = 'text-embedding-3-small'
st.title('ProductLens AI')

api_key_groq = st.text_input(
    "Enter GROQ API Key",
    type="password",
    key="groq_api_key"
)

# api_key_gemini = st.text_input(
#     "Enter Gemini API Key",
#     type="password",
#     key="gemini_api_key"
# )

api_key_openai = st.text_input(
    "Enter OpenAI API Key",
    type="password",
    key="openai_api_key"
)


def generate_collection_name(url, model_name):
    unique_value = f"{url.strip()}_{model_name}"

    url_hash = hashlib.sha256(
        unique_value.encode("utf-8")
    ).hexdigest()[:16]

    return f"product_{url_hash}"
if api_key_groq and api_key_openai:
    model=ChatGroq(model='qwen/qwen3.8-27b',groq_api_key=api_key_groq)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=api_key_gemini)
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key_openai)

    session_id=st.text_input("please provide Yout Session ID",value='default-session')

    if 'store' not in st.session_state:
        st.session_state.store={}

    product_link = st.text_input("Enter Product Link",
    placeholder="Paste a link from Amazon, Flipkart, Meesho, or Myntra..."
    )

    if product_link:
        # Generate collection name based on product URL
        collection_name = generate_collection_name(product_link,embed_model)


        # Create or load the collection
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory='./chromadb_productlens'
        )
        if vector_store._collection.count() == 0:

            documents=WebBaseLoader(product_link).load()
            splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
            chunks=splitter.split_documents(documents)
            vector_store.add_documents(chunks)
        retriever=vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
                    "k": 4,
                    "fetch_k": 20,
                    "lambda_mult": 0.5
                    }
        )
        contxt_sys_prompt = """
You are a query rewriting assistant for a RAG retrieval system.

Your ONLY task is to rewrite the latest user question into a standalone search query using chat history when necessary.

STRICT RULES:
1. NEVER answer the user's question.
2. NEVER provide explanations, lists, or product details as an answer.
3. Use chat history ONLY to resolve references such as "it", "they", or "those products".
4. Preserve the original intent of the question.
5. If the question is already standalone, return it unchanged.
6. Output ONLY the rewritten question as plain text.
"""
        contxt_prompt=ChatPromptTemplate([
            ('system',contxt_sys_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ])

        from langchain_core.output_parsers import StrOutputParser

        question_rewriter = contxt_prompt | model | StrOutputParser()

        history_aware_retriever=create_history_aware_retriever(model,retriever,contxt_prompt)

        sys_prompt='''You are a helpful AI assistant specializing in context-based question answering.
        Answer the user's question using ONLY the information explicitly provided in the retrieved context.
        If Not Sure,Say I don't know
        context
        {context}
        '''
        qa_prompt=ChatPromptTemplate([
                ('system',sys_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ])
        document_chain=create_stuff_documents_chain(model,qa_prompt,output_parser=StrOutputParser())

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
        with st.form("question_form"):
            user_query = st.text_input("Ask question")
            submitted = st.form_submit_button("Ask")

        if submitted and user_query:
            session_history=get_session_history(session_id)
            try:
                response=conv_rag_chain.invoke({'input':user_query},config={'configurable':{"session_id":session_id}})

                st.subheader('Assistant Answer')
                st.write(response['answer'])

                with st.expander("chat_history"):
                    st.write(session_history.messages)

            except Exception as e:
                st.error("""⚠️ Currently facing the Issue {e}.Please try again later.""")

else:
    st.warning('Please Enter Your API to comtinue.....')

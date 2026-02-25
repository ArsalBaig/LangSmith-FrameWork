# Importing Lib's

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()
@traceable(name= 'load_pdf_and_vectorstore')
@st.cache_resource
def load_vectorstore(pdf_path):
    if pdf_path.endswith('.pdf'):
        loader = PyPDFLoader(pdf_path)
    elif pdf_path.endswith('.txt'):
        loader = TextLoader(pdf_path)
    else:
        st.error("Only PDF and TXT files supported")
        return None
    
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)
    
    embeddings = load_embeddings()
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=f".chroma_{os.path.basename(pdf_path)}"
    )
    return vectorstore

#  Applying Embedding & Vectorstor.
@traceable(name= 'load_embeddings')
@st.cache_resource
def load_embeddings():
    """Load embeddings once"""
    return HuggingFaceEmbeddings(
        model_name='all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'batch_size': 128,
            'normalize_embeddings': True
        }
    )
    
# Initilizing LLM.

@traceable(name= 'load_llm')
@st.cache_resource
def load_llm():
    """Cache LLM"""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

@traceable(name= 'splitting_docs')
def format_docs(split_docs):
    return '\n\n'.join(doc.page_content for doc in split_docs)

# Streamlit App UI.
st.set_page_config(page_title="RAG App", page_icon="📚", layout="wide")
st.title("📚 RAG Q&A")

with st.sidebar:
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader('Choose a file', type=['pdf', 'txt'])

if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.pdf_path = temp_path
    st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")

if st.session_state.pdf_path:
    try:
        with st.spinner("Loading and indexing document..."):
            vectorstore = load_vectorstore(st.session_state.pdf_path)
        
            llm = load_llm()
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'Answer the question only when you know the answer. Otherwise say "I have insufficient knowledge about this."'),
            ('human', 'Question: {question}\n\nContext: {context}')
        ])
        
        # Parallel Execution Pipeline.        
        parallel = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough() # RunnablePassthrough() takes whatever comes-in and do nothing with it remain unchange.
        })
        
        chain = parallel | prompt | llm | StrOutputParser()
        
        question = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Searching..."):
                    response = chain.invoke(question.strip())
                st.success(response)
            else:
                st.warning("Please enter a question")
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("👆 Upload a PDF or TXT file to get started")
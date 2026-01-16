import os
import tempfile
import streamlit as st
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.document_loaders import TextLoader, PyPDFLoader

st.set_page_config(
    page_title="RAG Knowledge Agent",
    page_icon="📚🧠",
    layout="centered"
)

st.title("📚🧠 RAG Knowledge Agent")
st.caption("Ask questions and get answers grounded in your own documents.")

openai_api_key = st.text_input(
    "OpenAI API Key",
    type="password",
    help="Your key is used only during this session."
)

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

uploaded_files = st.file_uploader(
    "Upload documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

def load_documents(files) -> List[str]:
    documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            file_path = tmp.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        else:
            loader = TextLoader(file_path)
            docs = loader.load()

        documents.extend(docs)

    return documents

def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

def answer_question(query, vector_store):
    retriever_docs = vector_store.similarity_search(query, k=4)

    context = "\n\n".join([doc.page_content for doc in retriever_docs])

    prompt = f"""
    Answer the question using ONLY the context below.
    If the answer is not present, say you don't know.

    Context:
    {context}

    Question:
    {query}
    """

    model = ChatOpenAI(temperature=0)
    response = model([HumanMessage(content=prompt)])

    return response.content

if uploaded_files and openai_api_key:
    with st.spinner("Processing documents..."):
        docs = load_documents(uploaded_files)
        vector_store = build_vector_store(docs)

    st.success("Documents processed successfully!")

    question = st.text_input("Ask a question based on your documents")

    if question:
        with st.spinner("Searching and generating answer..."):
            answer = answer_question(question, vector_store)

        st.subheader("📌 Answer")
        st.write(answer)

st.caption(
    "This is a demo RAG system. Answers are strictly limited to uploaded documents."
)


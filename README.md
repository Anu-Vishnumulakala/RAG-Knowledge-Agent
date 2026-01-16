# RAG-Knowledge-Agent
A Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions with answers strictly grounded in the provided content.
## Project Overview
Large Language Models are powerful but often hallucinate when answering questions without reliable context. This project solves that problem by implementing a **Retrieval-Augmented Generation (RAG)** pipeline that ensures answers are generated **only from user-uploaded documents**. The system combines document retrieval with language generation to deliver accurate, explainable, and trustworthy responses.
## Key Features
- **Document Upload**  
  Supports PDF and TXT files.
- **Smart Text Chunking**  
  Splits documents into overlapping chunks for better semantic retrieval.
- **Vector Embeddings**  
  Converts document chunks into embeddings using OpenAI models.
- **Vector Database (FAISS)**  
  Stores and retrieves the most relevant document chunks efficiently.
- **Context-Grounded Question Answering**  
  Answers questions strictly based on retrieved document content.
- **Interactive Streamlit Interface**  
  Simple and intuitive UI for uploading files and asking questions.



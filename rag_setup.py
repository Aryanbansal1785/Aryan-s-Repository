import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_pdf_to_vectorstore(filepath: str):
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    #Embed and store in ChromaDB
    os.makedirs("vectorstore", exist_ok=True)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="vectorstore"
    )
    
    print(f"Indexed{len(chunks)} chunks from PDF")
    return vectorstore
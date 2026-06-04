import os
from db_setup import load_file_to_db
from rag_setup import load_pdf_to_vectorstore

def handle_file(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in [".csv", ".xlsx", ".xls"]:
        print("detected tabular file ,loafing into SQLite..")
        cols, rows= load_file_to_db(filepath)
        return "sql", cols, rows
    
    elif ext == ".pdf":
        print("Detected PDF, loading into ChromaDB..")
        vectorstore = load_pdf_to_vectorstore(filepath)
        return "rag", vectorstore, None
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")

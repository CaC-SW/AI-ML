import os
import pandas as pd
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

PERSIST_DIR = "./chroma_db"
client = chromadb.Client(Settings(persist_directory=PERSIST_DIR, anonymized_telemetry=False))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def safe_read_csv(file_path):
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc, errors='replace') as f:
                return pd.read_csv(f)
        except Exception as e:
            continue
    raise ValueError(f"Could not read {file_path} with tried encodings.")

def load_all_csvs(base_dir="Dataset"):
    text_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = safe_read_csv(file_path)
                combined_text = "\n".join(df.astype(str).fillna("").agg(" ".join, axis=1))
                chunks = splitter.split_text(combined_text)
                text_chunks.extend(chunks)
    return text_chunks

def vectorstore(chunks):
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        collection = client.get_or_create_collection(name="rag_docs", embedding_function=embedding_fn)
    else:
        collection = client.create_collection(name="rag_docs", embedding_function=embedding_fn)
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"doc_{i}"])
    return collection

def retrieve_top_k(query, collection, k=3):
    result = collection.query(query_texts=[query], n_results=k)
    return result['documents'][0]



def query(prompt, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]


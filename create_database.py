from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
import requests

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

BASE_URL = "https://docs.aws.amazon.com/pdfs/bedrock/latest/userguide/bedrock-ug.pdf"
DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma")
PDF_FILE = "bedrock_user_guide.pdf"
BATCH_SIZE = 1000

def load_or_download_pdf():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = DATA_DIR / PDF_FILE

    if not pdf_path.exists():
        print(f"PDF not found locally. Downloading from source...")
        response = requests.get(BASE_URL)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        print(f"PDF downloaded and saved to {pdf_path}.")
    else:
        print(f"PDF found locally at {pdf_path}.")
    
    loader = PyPDFLoader(DATA_DIR / PDF_FILE)
    documents = loader.load()
    return documents


def split_documents_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100, length_function=len)
    chunks = splitter.split_documents(documents)
    return chunks


def embed_chunks_to_vectorstore(chunks):
    """
    Embeds document chunks into a persistent Chroma vector store.
    Batching is used to stay within model input size limits.
    """
    vector_store = Chroma(embedding_function = EMBEDDINGS, persist_directory = CHROMA_DIR)
    
    print(f"Embedding {len(chunks)} chunks (batch size: {BATCH_SIZE})...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        vector_store.add_documents(batch)
    
    print(f"Embedding complete. Vector store saved to {CHROMA_DIR}.")
    return vector_store

def generate_data_store():
    if not CHROMA_DIR.exists():
        documents = load_or_download_pdf()
        chunks = split_documents_into_chunks(documents)
        vector_store = embed_chunks_to_vectorstore(chunks)
    else:
        vector_store = Chroma(embedding_function=EMBEDDINGS, persist_directory=CHROMA_DIR,)

    return vector_store
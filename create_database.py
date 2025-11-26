from dotenv import load_dotenv
from langchain_chroma import Chroma
from openai import OpenAI
from pathlib import Path
import json
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma")

client = OpenAI(api_key=api_key)

def json_to_chunks(json_path):
    data = json.load(open(json_path))
    chunks = [] # each QA under for Topic will be one chunk
    
    for outer_idx, (topic, qa_list) in enumerate(data.items()):
        for inner_idx, qa in enumerate(qa_list):
            text = (
                f"[Topic]: {topic}\n"
                f"[Question]: {qa["question"]}\n"
                f"[Answer]: {qa["answer"]}"
            )

            chunks.append({
                "text": text,
                "metadata": {
                    "idx": f"{outer_idx}:{inner_idx}",
                    "topic": topic,
                }
            })
    
    return chunks


def embed_json_chunks_to_vectorstore(chunks):
    print(f"Loaded {len(chunks)} chunks from JSON.")

    vector_store = Chroma(embedding_function=None, persist_directory=CHROMA_DIR)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["metadata"]["idx"] for c in chunks]

    embeddings = []
    for text in texts:
        response = client.embeddings.create(model="text-embedding-3-small", input=text)
        embeddings.append(response.data[0].embedding)
    
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
    # vector_store.persist()
    return vector_store


def generate_json_datastore(json_file):
    """Pipeline to generate datastore from JSON"""
    if CHROMA_DIR.exists():
        return Chroma(embedding_function=None, persist_directory=CHROMA_DIR)
    
    chunks = json_to_chunks(json_path = DATA_DIR / json_file)
    vector_store = embed_json_chunks_to_vectorstore(chunks=chunks)
    return vector_store

import os
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pathlib import Path

from prompt import PROMPT_TEMPLATE

BASE_URL = "https://docs.aws.amazon.com/pdfs/bedrock/latest/userguide/bedrock-ug.pdf"
DATA_DIR = Path("/Users/sayalimoghe/Documents/Learn/GitHub/rag-assistant/data")
CHROMA_DIR = Path("/Users/sayalimoghe/Documents/Learn/GitHub/rag-assistant/chroma")
PDF_FILE = "bedrock_user_guide.pdf"
BATCH_SIZE = 1000

# Load environment variables
load_dotenv()

# Check the API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# Initialize embeddings
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

# Initialize client
client = OpenAI(api_key = api_key)

# -------------------------------------------------------------------------------------------------
# Step 1: Download PDF if not already present
# -------------------------------------------------------------------------------------------------
def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = DATA_DIR / PDF_FILE

    if not pdf_path.exists():
        print(f"PDF not found locally at: {pdf_path}")
        print(f"Downloading from {BASE_URL} ...")
        response = requests.get(BASE_URL)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded and saved to {pdf_path}")
    else:
        print(f"PDF already exists at {pdf_path}")
    
    # PyPDFLoader uses the pypdf library under the hood.
    # For other PDF loading options, check out alternatives like UnstructuredLoader, PyMuPDF, or PDFPlumber depending on your needs.
    loader = PyPDFLoader(DATA_DIR / PDF_FILE)
    documents = loader.load()
    return documents

# -------------------------------------------------------------------------------------------------
# Step 2: Split the data into chunks
# -------------------------------------------------------------------------------------------------
def split_data(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100, length_function=len)
    chunks = splitter.split_documents(documents)
    return chunks

# -------------------------------------------------------------------------------------------------
# Step 3: Embed the chunks
# -------------------------------------------------------------------------------------------------
def chunk_embedding(chunks):
    vector_store = Chroma(
        embedding_function = EMBEDDINGS,
        persist_directory = CHROMA_DIR,  # Save data locally
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        vector_store.add_documents(batch)
    
    print(f"Embedded and saved {len(chunks)} chunks to {CHROMA_DIR}.")
    return vector_store
    
    # embedded_chunks = []
    # for chunk in chunks:
    #     response = client.embeddings.create(input=chunk["text"], model=CONFIG["EMBED_MODEL"])
    #     vector = response.data[0].embedding
    #     embedded_chunks.append({
    #         "text": chunk["text"],
    #         "source": chunk["source"],
    #         "embedding": vector
    #     })
    # return embedded_chunks

# -------------------------------------------------------------------------------------------------
# Step 4: Query for Relevant Data
# -------------------------------------------------------------------------------------------------
def find_relevant_docs(vector_store, query, k=5):
    # query_vector = EMBEDDINGS.embed_query(query)
    relevant_docs = vector_store.similarity_search_with_score(query, k)
    return relevant_docs

# -------------------------------------------------------------------------------------------------
# Step 5: Generate a response
# -------------------------------------------------------------------------------------------------
def build_context_from_docs(docs: list, max_chars: int = 3000) -> str:
    parts = []
    total = 0

    for i, item in enumerate(docs, start=1):
        # Unpack (Document, score)
        if isinstance(item, tuple):
            doc, score = item
        else:
            doc, score = item, None

        header = f"[DOC {i} | source: {doc.metadata.get('source','unknown')} | id: {doc.id if doc.id else '-'}]"
        snippet = doc.page_content.strip()

        block = header + "\n" + snippet + "\n\n"

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts)

def generate_response(query, context):
    input = PROMPT_TEMPLATE.format(context=context, question=query)
    response = client.responses.create(
        model = "gpt-4o-mini",
        input = input
    )

    return response

def main():
    if not CHROMA_DIR.exists():
        print(f"Database not found. Creating database from {BASE_URL}.")
        documents = load_data()
        print(f"{PDF_FILE} file contains {len(documents)} pages.")
        # documents[0] = (page_content, metadata)
        
        chunks = split_data(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        # TODO: Understand the data

        vector_store = chunk_embedding(chunks=chunks)
        # The "batch size" error occurs when the total size of all chunks sent at once to the embedding API exceeds the model's maximum input size.
        # Therefore, we must both:
        #   1. Keep chunks reasonably small (e.g., 500-1000 characters)
        #   2. Embed documents in **smaller batches** (e.g., 500 chunks at a time)
        # This ensures that each API call stays within limits and avoids batch size errors.
        print(f"Downloaded and saved to {CHROMA_DIR}")
    else:
        vector_store = Chroma(
            embedding_function = EMBEDDINGS,
            persist_directory = CHROMA_DIR,
        )
        print(f"Embeddings already exist at {CHROMA_DIR}")
    
    # The score means the distance. The first one has the min value.
    query = "What is AWS Bedrock?"
    top_docs = find_relevant_docs(vector_store, query, k=5) 
    if len(top_docs) == 0:
        print(f"Unable to find matching results.")

    # print(top_docs[0][0].metadata.keys())

    parts = build_context_from_docs(top_docs)
    print("Context building is completed.")
    
    response = generate_response(query, parts)
    print(f"Here is a response for your query:")
    # print(type(response))
    print(response.output[0].content[0].text)
    
if __name__ == "__main__":
    main()


from dotenv import load_dotenv
from openai import OpenAI
from prompt_template import PROMPT_TEMPLATE
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(api_key = api_key)

def retrieve_relevant_documents(vector_store, query, k=5):
    return vector_store.similarity_search_with_score(query, k)

def build_context_from_docs(docs):
    parts = "\n\n---\n\n".join([doc.page_content.strip() for doc, _score in docs])
    sources = list(set([f"{doc.metadata.get('idx','unknown')}:{doc.metadata.get('page','-')}" for doc, _score in docs])) 
    return parts, sources

def generate_llm_response(query, context):
    input = PROMPT_TEMPLATE.format(context=context, question=query)
    response = client.responses.create(model="gpt-4o", input=input)
    return response
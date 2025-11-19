from dotenv import load_dotenv
from openai import OpenAI
from prompt import PROMPT_TEMPLATE
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

client = OpenAI(api_key = api_key)

def retrieve_relevant_documents(vector_store, query, k=5):
    return vector_store.similarity_search_with_score(query, k)

def build_context_from_docs(docs, max_chars):
    parts = []
    total = 0

    for i, item in enumerate(docs, start=1):
        # Unpack (Document, score)
        if isinstance(item, tuple):
            doc, score = item
        else:
            doc, score = item, None

        header = (
            f"[DOC {i} | source: {doc.metadata.get('source','unknown')} | "
            f"page: {doc.metadata.get('page','-')} | id: {doc.id or '-'}]"
        )        
        snippet = doc.page_content.strip()

        block = header + "\n" + snippet + "\n\n"

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts)

def generate_llm_response(query, context):
    input = PROMPT_TEMPLATE.format(context=context, question=query)
    response = client.responses.create(model="gpt-4o-mini", input=input )
    return response
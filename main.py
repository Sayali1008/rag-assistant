from create_database import generate_data_store
from query_data import retrieve_relevant_documents, build_context_from_docs, generate_llm_response
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Your search query")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    vector_store = generate_data_store()

    top_docs = retrieve_relevant_documents(vector_store, args.query, k=5)
    if not top_docs:
        print("No relevant content found for the query.")
        return
    
    # The score means the distance. The first one has the min value.
    parts, sources = build_context_from_docs(top_docs)
    response = generate_llm_response(args.query, parts)

    print(f"Context:\n{parts}\n")
    print(response.output[0].content[0].text)
    print(f"\nSources:, {sources}\n")
    
if __name__ == "__main__":
    main()


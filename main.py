from create_database import generate_data_store
from query_data import retrieve_relevant_documents, build_context_from_docs, generate_llm_response

def main():
    print("\nInitializing vector store...")
    vector_store = generate_data_store()
    print("Vector store ready.\n")

    query = "How do you import a pre-trained model into bedrock"
    print(f"Running query: '{query}'")

    top_docs = retrieve_relevant_documents(vector_store, query, k=5)
    if not top_docs:
        print("No relevant content found for the query.")
        return
    
    # The score means the distance. The first one has the min value.
    print(f"Retrieved {len(top_docs)} relevant documents.")

    parts = build_context_from_docs(top_docs)
    print("Context successfully assembled from retrieved documents.\n")

    response = generate_llm_response(query, parts)
    print("LLM response generated:\n")
    print(response.output[0].content[0].text)
    
if __name__ == "__main__":
    main()


from create_database import generate_json_datastore
from query_data import retrieve_relevant_documents, build_context_from_docs, generate_llm_response
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Your search query")
    args = parser.parse_args()

    json_file = "ml_algorithms.json"
    vector_store = generate_json_datastore(json_file=json_file)

    top_docs = retrieve_relevant_documents(vector_store, args.query, k=3)
    if not top_docs:
        print("No relevant content found for the query.")
        return
    
    # The score means the distance. The first one has the minmum value indictating closest distance.
    parts, sources = build_context_from_docs(top_docs)
    response = generate_llm_response(args.query, parts)

    print(response.output[0].content[0].text)

    print(f"\nSources:")
    for (doc, score) in zip(top_docs, sources):
        print(score, doc[1])
    
if __name__ == "__main__":
    main()

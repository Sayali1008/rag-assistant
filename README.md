# RAG System Prototype

This repository contains a Retrieval-Augmented Generation (RAG) prototype built using LangChain, ChromaDB, and OpenAI's embedding and LLM APIs. It downloads the AWS Bedrock User Guide, embeds it into a vector store, and answers user queries using relevant document context.

---

## RAG Architecture Diagram
```
            ┌─────────────────────────┐
            │        User Query       │
            └────────────┬────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │    Vector Store    │
              │    (ChromaDB)      │
              └─────────┬──────────┘
                        │ Similarity Search
                        ▼
            ┌───────────────────────────┐
            │ Retrieved Relevant Chunks │
            └────────────┬──────────────┘
                         │ Context Assembly
                         ▼
            ┌───────────────────────────┐
            │     Prompt Template       │
            │ (Context + User Question) │
            └────────────┬──────────────┘
                         │
                         ▼
            ┌───────────────────────────┐
            │  OpenAI LLM (gpt-4o-mini) │
            └────────────┬──────────────┘
                         │ Generated Answer
                         ▼
            ┌──────────────────────────┐
            │       Final Output       │
            └──────────────────────────┘
```

## Project Structure
```
.
├── main.py
├── create_database.py
├── query_data.py
├── prompt.py
├── data/              # Downloaded PDFs
├── chroma/            # Persisted Chroma vector store
└── requirements.txt
```

---

## Working

This repository implements a traditional Retrieve→Augment→Generate pipeline.

### **1. Document Loading**

* The AWS Bedrock User Guide PDF is downloaded if not already present.
* The PDF is parsed page-by-page.

### **2. Text Chunking**

The PDF is split into overlapping text chunks:

* `chunk_size = 700`
* `chunk_overlap = 100`

Chunking allows the embedding model to process text while maintaining context.

### **3. Embedding and Vector Store Creation**

* Embeddings are generated with **OpenAI text-embedding-3-small**.
* ChromaDB is used as the persistent vector store.
* Chunks are embedded **in batches** to avoid API input-size errors.

📌 **Batch Size Notes:**

* The OpenAI embedding API limits input size per request.
* Sending too many chunks at once causes:
  `ValueError: Batch size ... exceeds max batch size ...`
* Therefore, chunk text must be small *and* embedding requests must be batched.

### **4. Retrieval**

For a given query:

* The vector store performs **semantic similarity search**.
* Returned items include `(Document, score)` pairs.

📌 **Score Notes:**

* The score is a *distance* metric.
* A **lower score** indicates a **closer match** to the query.

### **5. Context Packaging**

* Retrieved document chunks are formatted with metadata.
* Context is truncated if it exceeds `max_chars = 3000` to keep prompts manageable.

### **6. LLM Response Generation**

* A prompt template inserts the context and the user query.
* OpenAI `gpt-4o-mini` is used to generate a response.


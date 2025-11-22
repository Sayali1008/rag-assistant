## 1. Project Notes

### 1.1 Embedding Strategy

#### Batch Size Guidelines
- Batch size errors occur when the **total size** of chunks sent in a single request exceeds the model’s maximum input limit.
- To avoid this:
  - Keep individual chunks small (≈ **500–1000 characters**).
  - Process chunks in **smaller batches** (e.g., **≤ 500 chunks per API call**).

#### Alternate Embedding Code (Per-Chunk Embedding)
```python
embedded_chunks = []
for chunk in chunks:
    response = client.embeddings.create(input=chunk["text"], model=CONFIG["EMBED_MODEL"])
    vector = response.data[0].embedding
    embedded_chunks.append({
        "text": chunk["text"],
        "source": chunk["source"],
        "embedding": vector
    })
return embedded_chunks
```

---

## 2. Quality of Answers (RAG / LLM Performance Factors)

1. **Source Materials**
   - Accuracy, completeness, and relevance of underlying documents.

2. **Text Splitting Strategy**
   - Chunk size, semantic coherence, overlap logic.

3. **LLM Model & Prompting**
   - Model selection, prompt design, system instructions.

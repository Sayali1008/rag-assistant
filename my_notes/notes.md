## Factors Affecting the System

**Batch Size**
- Batch size errors occur when the **total size** of chunks sent in a single request exceeds the model’s maximum input limit.
- To avoid this:
  - Keep individual chunks small (≈ **500–1000 characters**).
  - Process chunks in **smaller batches** (e.g., **≤ 500 chunks per API call**).

## Strategies to Improve Quality of Responses (LLM Performance Factors)**

**1. Improve Document Retrieval**
- Increase the number of retrieved documents (k).
- Discard documents below a similarity score to avoid noise.  
- Rerank retrieved documents - using a smaller LLM or embedding-based similarity to rank by relevance after retrieval.
- Hybrid search - Combine embeddings with keyword search or BM25 to improve recall for specific terms.  

**2. Model Selection & Parameters**
- Using larger models for complex queries; smaller models for simple or high-throughput queries.  
- _Temperature and top-p tuning_:  
  - Lower temperature (0–0.3) for factual, precise answers.  
  - Higher temperature (0.7–1.0) for creative or exploratory answers.  

**3. Multi-turn / Conversational Enhancement**
- Allow follow-up queries to leverage prior context. Example: “In the last answer you mentioned X. Can you expand on Y?”  
- Generate initial answer → check against documents → refine.

**4. Advanced Enhancements**
- Collect user feedback to improve context selection or prompts over time.  
- Include headings, section type, or table captions for better retrieval.  
- Encourage step-by-step reasoning using retrieved context.  
- Use retrieval to query external APIs or structured knowledge for dynamic answers.


## Chunk Embedding for QA Dataset

**Chunk size = ONE QA pair per chunk**  
- Standard method for OpenAI, Google, Anthropic, and most production-grade RAG systems

**Why Per-Q/A Chunk Is Superior**
1. _High retrieval accuracy_: Exact QA pair retrieval  
2. _Semantic coherence_: One idea, one answer, one conceptual scope  
3. _Prevents hallucinations_: Reduces unrelated context inclusion  
4. _Better metadata control_: Track topic, subtopic, difficulty, QA index, source

**Why You Should not Chunk by Topic**
- Overlong chunks (500–10,000 tokens)  
- Mixed semantic signals  
- Poor retriever precision  
- Reduced RAG answer quality  
- High hallucination rates  
- Only acceptable if each topic is extremely short or for topic classification


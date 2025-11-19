PROMPT_TEMPLATE = """
You are an expert assistant. Use ONLY the context provided to answer the question.
If the answer is not in the context, say "I don't know" rather than hallucinating.
When you use content from a context item, cite its source in square brackets, e.g. [source: doc1.pdf page 4].

Context:
{context}

Question:
{question}

Answer:
"""
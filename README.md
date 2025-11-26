# Retrieval-Augmented Generation on Machine Learning QA Dataset

This repository contains a Retrieval-Augmented Generation (RAG) system built using Chroma from LangChain, for bulding the vector database, and OpenAI's APIs. It constructs a synthetic dataset using several topics from the machine learning paradigm and stores this data as a question-answering JSON in `ml_algorithms.json`. Then, we can query on the data.

---

## Project Structure
```
.
├── data/
├── chroma/
├── generate_data.py
├── create_database.py
├── query_data.py
├── prompt_template.py
└── main.py
```

---


## Testing the System

This section covers new test questions, what this RAG system should handle, and its limitations based on the `ml_algorithms.json` dataset.

---

### New Questions to Test the RAG System

These questions are **not in the dataset** but are answerable by combining content from it.

#### Supervised Learning
- How is supervised learning different from unsupervised learning?  
- Why is cross-validation important in supervised learning?  
- What problems typically require regression vs classification?

#### Linear Regression
- How do outliers affect a linear regression model?  
- When should you prefer multiple linear regression over simple linear regression?

#### Logistic Regression
- Why do logistic regression models use the log-odds instead of predicting values directly?  
- How would you interpret an odds ratio greater than 1?

#### Decision Trees / Random Forest / Gradient Boosting
- Why are decision trees prone to overfitting?  
- How do Random Forests reduce variance in predictions?  
- Why do gradient boosting models require careful tuning?  
- What does “tree depth” influence in boosting models?

#### SVM
- How does the margin of an SVM affect its generalization?  
- When would you choose a linear kernel over an RBF kernel?

#### K-NN
- Why does KNN perform poorly on high-dimensional data?  
- What happens if you choose k=1 in KNN?

#### Naive Bayes
- Why does Naive Bayes work well for text classification despite its independence assumption?  
- How does Laplace smoothing change probability estimates?

#### Neural Networks
- What problem do activation functions solve?  
- Why do deep networks suffer from vanishing gradients?

#### Unsupervised Learning / Clustering / Dimensionality Reduction
- When would you use PCA instead of t-SNE?  
- Why is determining the number of clusters difficult in clustering?  
- What is the purpose of a silhouette score?

#### Reinforcement Learning
- How is reward different from a label?  
- What is the role of a policy in reinforcement learning?

#### RAG
- Why does RAG often produce more factual answers than a plain LLM?  
- What role does retrieval play in reducing hallucinations?

#### Zero-Shot Learning
- How do semantic attributes help a model classify unseen categories?  
- Why is ZSL useful for tasks with constantly changing labels?

---

### What This RAG System *Should* Be Able to Answer

The system should answer questions that are:
- Directly stated in the dataset  
- Paraphrased from dataset content  
- Combinations of 1–3 related facts

**Examples:**
- Definitions (`What is XGBoost?`, `What is dimensionality reduction?`)  
- Comparisons implicitly covered (`How is Gradient Boosting different from Random Forest?`)  
- Advantages / disadvantages of algorithms  
- Assumptions (linear regression, logistic regression)  
- Hyperparameters and roles (Random Forest, LightGBM, XGBoost)  
- High-level RL concepts (exploration/exploitation, model-free vs model-based)  
- RAG architecture questions  
- Zero-shot learning concepts

---

### What the RAG System *Cannot* Answer

It cannot handle:
- Numerical calculations or derivations  
- Code examples or step-by-step implementations  
- Best-algorithm recommendations  
- Real-world domain-specific advice  
- Tasks not explicitly covered in the dataset  

**Concrete examples:**
- “Explain LSTMs vs GRUs.”  
- “How do transformers work?”  
- “Derive the logistic regression cost function.”  
- “Write PyTorch code for a CNN.”  
- “Which algorithm is best for fraud detection?”  
- “How do you evaluate a retrieval quality score?”  

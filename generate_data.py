from dotenv import load_dotenv
from openai import OpenAI
import json
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")
client = OpenAI(api_key = api_key)

PROMPT = """
You are an AI assistant tasked with generating a structured JSON dataset of high-quality question-answer pairs. 

Instructions:

1. Generate exactly 5 question-answer pairs for each of the topics listed below.
2. The output must be a single JSON object, where each key is the topic name and its value is a list of 5 objects. Each object should have two keys: "question" and "answer".
3. Ensure all questions are unique within and across topics.
4. The answers must be detailed, ranging from 5 to 5000 characters.
5. Maintain consistency and coherence: the questions and answers should relate to each other, such that a question asked in one part of the JSON could reference or connect logically to information from other QAs.
6. The JSON must be valid and parseable. Do not include any extra text outside the JSON.
7. Use clear, factual, and concise language. Avoid speculative answers.

Here is an example format for the JSON output:
{{
  "Topic1": [
    {{"question": "...", "answer": "..."}},
    ...
  ],
  "Topic2": [
    ...
  ]
}}

Topics:
{topics}
"""


def append_to_json_list(filepath, new_data):
    """
    Appends new_data (a dictionary) to a JSON file that contains a list of dictionaries.
    If the file doesn't exist or is empty, it creates a new file with a list containing new_data.
    """
    data = []
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            if not isinstance(data, dict):
                data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is empty, we start with an empty list
        data = {}

    data.update(new_data)

    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    filename = "./data/ml_algorithms.json"

    topics = [
        "Supervised Learning", "Linear Regression", "Logistic Regression",
        "Decision Trees", "SVM", "K Nearest Neighbor", "Naive Bayes",
        "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM",
        "CatBoost", "Neural Networks", "Unsupervised Learning", "Clustering",
        "Dimensionality Reduction", "Association Rule", "Reinforcement Learning",
        "Model-Based Methods", "Model-Free Methods",
        "RAG", "MCP", "Zero Shot Learning"
    ]

    print(f"Topics being processed: {topics}")

    prompt = PROMPT.format(topics = topics)

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{ "role": "user", "content": prompt }],
        temperature=0.7
    )

    json_text = response.choices[0].message.content.strip()

    # Parse JSON
    try:
        qa_json = json.loads(json_text)
        append_to_json_list(filepath=filename, new_data=qa_json)
        print(f"Data successfully saved to {filename}")
    except json.JSONDecodeError:
        print("Error decoding JSON. Raw output:")
        print(json_text)
        qa_json = {}

if __name__ == "__main__":
    main()

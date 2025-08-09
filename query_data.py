import argparse
import json
# from dataclasses import dataclass
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from langchain_huggingface import HuggingFaceEmbeddings

model_options=[
    "sentence-transformers/all-MiniLM-L6-v2", 
    "sentence-transformers/all-mpnet-base-v2", 
    "BAAI/bge-large-en", 
    "intfloat/e5-base-v2", 
    "SproutsAI/embedding-model",
    "sentence-transformers/static-retrieval-mrl-en-v1",
    "sentence-transformers/all-MiniLM-L12-v2"
    ]

model_option_index=0
HuggingFaceWords2Vector = HuggingFaceEmbeddings(model_name=model_options[model_option_index])

CHROMA_PATH = f"chroma_{model_options[model_option_index].replace('/', '_').replace('-', '_')}"

# Load quiz data from JSON file
def load_quiz_data():
    try:
        with open('test_questions.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: test_questions.json file not found!")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
here are the options:
{options}

Respond only the Letter of the correct options like A, B, C and D
"""

def query_data():
    # Load quiz data from JSON
    quiz_data = load_quiz_data()
    
    if not quiz_data:
        print("No quiz data loaded. Exiting.")
        return
    
    # Prepare the DB.
    embedding_function = HuggingFaceWords2Vector
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Create HuggingFace pipeline for text generation
    print("Creating HuggingFace pipeline...")
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
    )
    
    # Wrap the pipeline in LangChain
    model = HuggingFacePipeline(pipeline=hf_pipeline)
    print("HuggingFace pipeline created.")

    for x, questions in enumerate(quiz_data, 1):
        print(f"Question {x} of {len(quiz_data)}")
        question = questions["question"]
        options = questions["options"]
        answer = questions["answer"]

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(question, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt = PROMPT_TEMPLATE.format(context=context_text, question=question, options=options)

        # Use the HuggingFace model instead of OpenAI
        response_text = model.predict(prompt)

        print("=================================\n")
        print(f"question: {question}\n")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        print(f"response_text: {response_text}\n")
        print(f"correct answer: {answer}\n")

# if __name__ == "__main__":
#     main()
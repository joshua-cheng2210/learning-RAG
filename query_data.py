import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from langchain_huggingface import HuggingFaceEmbeddings
HuggingFaceWords2Vector = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


CHROMA_PATH = "chroma"

quiz_data = [
    {
        "question": "Who does Alice follow down the rabbit hole at the beginning of the story?",
        "options": [
            "The White Rabbit",
            "The Cheshire Cat",
            "The Mad Hatter",
            "The Queen of Hearts"
        ],
        "answer": "The White Rabbit"
    },
    {
        "question": "What item does Alice find that makes her shrink?",
        "options": [
            "A small cake with 'EAT ME' written on it",
            "A bottle labeled 'DRINK ME'",
            "A mushroom",
            "A fan"
        ],
        "answer": "A small cake with 'EAT ME' written on it"
    },
    {
        "question": "Who continually shouts 'Off with their heads!'?",
        "options": [
            "The Queen of Hearts",
            "The King of Hearts",
            "The Duchess",
            "The Gryphon"
        ],
        "answer": "The Queen of Hearts"
    },
    {
        "question": "What animal is constantly smiling?",
        "options": [
            "The Cheshire Cat",
            "The March Hare",
            "The Dormouse",
            "The Caterpillar"
        ],
        "answer": "The Cheshire Cat"
    },
    {
        "question": "What game does the Queen of Hearts play using flamingos and hedgehogs?",
        "options": [
            "Croquet",
            "Chess",
            "Tennis",
            "Bowling"
        ],
        "answer": "Croquet"
    },
    {
        "question": "What advice does the Caterpillar give Alice regarding her size?",
        "options": [
            "That one side of the mushroom will make her grow taller and the other side will make her grow shorter.",
            "To eat more cake.",
            "To drink from the bottle.",
            "To fan herself rapidly."
        ],
        "answer": "That one side of the mushroom will make her grow taller and the other side will make her grow shorter."
    },
    {
        "question": "Who attends the Mad Tea-Party with the Mad Hatter?",
        "options": [
            "The March Hare and the Dormouse",
            "The White Rabbit and the Dodo",
            "The Queen of Hearts and the King of Hearts",
            "The Gryphon and the Mock Turtle"
        ],
        "answer": "The March Hare and the Dormouse"
    },
    {
        "question": "Why is it always tea-time at the Mad Tea-Party?",
        "options": [
            "Because Time has stopped for the Mad Hatter.",
            "They enjoy tea more than any other meal.",
            "It's a riddle they are trying to solve.",
            "They are waiting for someone to join them."
        ],
        "answer": "Because Time has stopped for the Mad Hatter."
    },
    {
        "question": "What does Alice use as a mallet in the Queen's croquet game?",
        "options": [
            "A flamingo",
            "A hedgehog",
            "A wooden stick",
            "Her arm"
        ],
        "answer": "A flamingo"
    },
    {
        "question": "Who recites 'Tis the Voice of the Lobster' for Alice?",
        "options": [
            "The Mock Turtle",
            "The Gryphon",
            "The White Rabbit",
            "The Mad Hatter"
        ],
        "answer": "The Mock Turtle"
    }
]


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
here are the options:
{options}
"""


def main():
    # Prepare the DB.
    embedding_function = HuggingFaceWords2Vector
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    for questions in quiz_data:
        question = questions["question"]
        options = questions["options"]
        answer = questions["answer"]

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(question, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        print(f"Context: {context_text}")
        
        # Create HuggingFace pipeline for text generation
        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
        )
        
        # Wrap the pipeline in LangChain
        model = HuggingFacePipeline(pipeline=hf_pipeline)

        prompt = PROMPT_TEMPLATE.format(context=context_text, question=question, options=options)

        # Use the HuggingFace model instead of OpenAI
        response_text = model.predict(prompt)

        # print(f"prompt: {prompt}\n")
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        # formatted_response = f"Response: {response_text}\nSources: {sources}"
        # print(formatted_response)
        print("=================================\n")
        print(f"question: {question}\n")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        print(f"response_text: {response_text}\n")
        print(f"correct answer: {answer}\n")


if __name__ == "__main__":
    main()
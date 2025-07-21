from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
# openai.api_key = os.environ['OPENAI_API_KEY']

HuggingFaceWords2Vector = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def main():
    # Get embedding for a word.
    embedding_function = HuggingFaceWords2Vector
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance", embeddings=HuggingFaceWords2Vector)
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")

def get_embedding(word: str, debug=0):
    embedding_function = HuggingFaceWords2Vector
    vector = embedding_function.embed_query(word)
    if debug:
        print(f"Vector for '{word}': {vector}")
        print(f"Vector length: {len(vector)}")
        print(f"Min value: {min(vector):.4f}")
        print(f"Max value: {max(vector):.4f}")
    return vector

def compare_embeddings(word1: str, word2: str):
    # embedding_function = HuggingFaceWords2Vector
    
    evaluator = load_evaluator("pairwise_embedding_distance", embeddings=HuggingFaceWords2Vector)

    # vector1 = embedding_function.embed_query(word1)
    # vector2 = embedding_function.embed_query(word2)
    # word  = word1
    # vector = vector1
    # print(f"Word: '{word}'")
    # print(f"Vector length: {len(vector)}")
    # print(f"Min value: {min(vector):.4f}")
    # print(f"Max value: {max(vector):.4f}")

    diff = evaluator.evaluate_string_pairs(prediction=word1, prediction_b=word2)

    print(f"Vector for '{word1}' and vector for '{word2}' has the difference of {diff}")


# if __name__ == "__main__":
#     main()

# compare_embeddings("girl loves boy", "girl loves boy")
# compare_embeddings("boy", "girl")
# compare_embeddings("boy loves girl", "girl loves boy")
# compare_embeddings("king", "queen")
# compare_embeddings("aaa", "bbb")
# compare_embeddings("aaa", "zzz")
# compare_embeddings("Malaysia", "Kuala Lumpur")
# compare_embeddings("Japan", "Tokyo")

def read_md_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# Usage
content = read_md_file("books/alice_in_wonderland.md")
content_embeddings = get_embedding(content, debug=0)
print(f"embeddings: {content_embeddings}\n Total characters: {len(content)}\n len of embeddings: {len(content_embeddings)}\n")
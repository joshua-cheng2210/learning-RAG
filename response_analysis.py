import os
import json

model_response_directory = "quiz_results"

for model_response_fp in os.listdir(model_response_directory):
    avg_relevance_sources = []
    count = 0
    num_questions = 0
    with open(os.path.join(model_response_directory, model_response_fp), "r") as f:
        model_responses = json.load(f)
        for response in model_responses:
            if response["is_correct"] == True:
                count += 1
            num_questions += 1
            avg_relevance_sources.append(response["avg relevance sources"])
    print(f"Model: {model_response_fp}, Correct: {count}/{num_questions}, Avg Relevance: {sum(avg_relevance_sources) / len(avg_relevance_sources) if avg_relevance_sources else 0}")
                
import os
import json

model_response_directory = "test_quiz_results"
incorrect_answers = []

for model_response_fp in os.listdir(model_response_directory):
    with open(os.path.join(model_response_directory, model_response_fp), "r") as f:
        model_response = json.load(f)
        for question in model_response:
            if question["is_correct"] == False and len(question["response"]) != 1:
                save = {
                    "model" : model_response_fp.replace(".json", ""),
                    "response" : question["response"],
                    "options" : question["options"],
                }
                incorrect_answers.append(save)

print(incorrect_answers)
json.dump(incorrect_answers, open("incorrect_answers.json", "w"), indent=4)
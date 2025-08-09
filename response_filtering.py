import os
import json

model_response_directory = "test_quiz_results"
incorrect_answers = []

for model_response_fp in os.listdir(model_response_directory):
    edit_model_response = []
    with open(os.path.join(model_response_directory, model_response_fp), "r") as f:
        model_response = json.load(f)
        for question in model_response:
            if question["is_correct"] == False and len(question["response"]) != 1:
                response = question["response"]
                edit_response = response.replace('-', '').strip()

                save = {
                    "model" : model_response_fp.replace(".json", ""),
                    "response" : edit_response,
                    "options" : question["options"],
                    "correct_answer" : question["correct_answer"]
                }
                incorrect_answers.append(save)

                question["response"] = edit_response
                question["is_correct"] = True if edit_response.upper() == question["correct_answer"].upper() else False
            edit_model_response.append(question)

    with open(f"{model_response_directory}/{model_response_fp}", "w") as f:
        json.dump(edit_model_response, f, indent=4)

# print(incorrect_answers)
# json.dump(incorrect_answers, open("incorrect_answers.json", "w"), indent=4)
with open("incorrect_answers.json", "w") as f:
    json.dump(incorrect_answers, f, indent=4)
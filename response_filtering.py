import os
import json

model_response_directory = "quiz_results"
incorrect_answers = []

for model_response_fp in os.listdir(model_response_directory):
    edit_model_response = []
    with open(os.path.join(model_response_directory, model_response_fp), "r") as f:
        model_response = json.load(f)
        for question in model_response:
            if question["is_correct"] == False and len(question["response"]) != 1:
                response = question["response"]
                original_response = str(question["response"])
                edit_response = response.replace('-', '').strip()
                alternate_correct_answer = ""

                

                question["response"] = edit_response
                # question["is_correct"] = True if edit_response.upper() == question["correct_answer"].upper() else False
                if edit_response.upper() == question["correct_answer"].upper():
                    question["is_correct"] = True
                else:
                    if question["correct_answer"].upper().strip() == "A":
                        alternate_correct_answer = question["options"][0][4:].replace('-', '').strip()
                    elif question["correct_answer"].upper().strip() == "B":
                        alternate_correct_answer = question["options"][1][4:].replace('-', '').strip()
                    elif question["correct_answer"].upper().strip() == "C":
                        alternate_correct_answer = question["options"][2][4:].replace('-', '').strip()
                    elif question["correct_answer"].upper().strip() == "D":
                        alternate_correct_answer = question["options"][3][4:].replace('-', '').strip()

                    if alternate_correct_answer.upper() == edit_response.upper():
                        question["is_correct"] = True
                    else:
                        if edit_response.upper().startswith(alternate_correct_answer.upper()):
                            question["is_correct"] = True
                        else:
                            question["is_correct"] = False
                
                save = {
                    "model" : model_response_fp.replace(".json", ""),
                    "original_response" : original_response,
                    "edit_response" : edit_response,
                    "options" : question["options"],
                    "correct_answer" : question["correct_answer"],
                    "alternate_correct_answer" : alternate_correct_answer
                }
                incorrect_answers.append(save)
                        
            edit_model_response.append(question)

    with open(f"{model_response_directory}/{model_response_fp}", "w") as f:
        json.dump(edit_model_response, f, indent=4)

# print(incorrect_answers)
# json.dump(incorrect_answers, open("incorrect_answers.json", "w"), indent=4)
with open("incorrect_answers.json", "w") as f:
    json.dump(incorrect_answers, f, indent=4)
import json

def extract_question_answer(dataset):
    extracted_data = []

    for entry in dataset["questions"]:
        question = entry["body"]

        if "ideal_answer" in entry:
            answer = entry["ideal_answer"][0]
            extracted_data.append({"question": question, "answer": answer})
        elif "exact_answer" in entry:
            answer = ", ".join([" ".join(words) for words in entry["exact_answer"]])
            extracted_data.append({"question": question, "answer": answer})

    return extracted_data

def main():
    with open("../data/BioASQ/BioASQ-training12b/training12b_new.json", 'r', encoding='utf-8') as json_file:
        dataset = json.load(json_file)

    extracted_data = extract_question_answer(dataset)

    # Save the extracted data to a new JSON file
    with open("../data/processed/bioASQ.json", 'w', encoding='utf-8') as json_file:
        json.dump(extracted_data, json_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
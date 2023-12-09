import json

def extract_question_answer(dataset):
    extracted_data = []

    for entry_id, entry_data in dataset.items():
        question = entry_data["QUESTION"]
        long_answer = entry_data["LONG_ANSWER"]

        extracted_data.append({"question": question, "answer": long_answer})

    return extracted_data

def main():
    with open("../data/pubmed_qa/ori_pqaa.json", 'r', encoding='utf-8') as json_file:
        dataset = json.load(json_file)

    extracted_data = extract_question_answer(dataset)

    # Save the extracted data to a new JSON file
    with open("../data/processed/pubmed_qaa.json", 'w', encoding='utf-8') as json_file:
        json.dump(extracted_data, json_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
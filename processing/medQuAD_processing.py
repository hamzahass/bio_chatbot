import os
import xml.etree.ElementTree as ET
import json

def process_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    qa_pairs = []

    for qa_pair in root.findall('.//QAPair'):
        question_element = qa_pair.find('./Question')
        answer_element = qa_pair.find('./Answer')

        if question_element is not None and answer_element is not None:
            question = question_element.text
            answer = answer_element.text

            if question is not None and answer is not None:
                qa_pairs.append({"question": question.strip(), "answer": answer.strip()})

    return qa_pairs

def process_directories(main_directory):
    all_qa_pairs = []
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                qa_pairs = process_xml(file_path)
                all_qa_pairs.extend(qa_pairs)

    return all_qa_pairs

def main():
    main_directory = "../data/MedQuAD"
    output_file = "../data/processed/medQuAD.json"

    all_qa_pairs = process_directories(main_directory)

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(all_qa_pairs, json_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
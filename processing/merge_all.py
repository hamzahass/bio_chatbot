import json
import os

def merge_datasets(folder_path, output_filename):
    merged_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                merged_data.extend(data)

    # Save the merged data to a new JSON file
    output_path = os.path.join(folder_path, output_filename)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(merged_data, json_file, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     folder_path = "../data/processed"
#     output_filename = "final_qa_dataset.json"
#     merge_datasets(folder_path, output_filename)

def count_entries(output_file):
    with open(output_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        return len(data)

if __name__ == "__main__":
    output_file = "../data/processed/final_qa_dataset.json"

    num_entries = count_entries(output_file)

    print(f"Number of entries in {output_file}: {num_entries}")
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def word_count(text):
    return len(text.split(' '))

def tokenize_dataset(dataset, tokenizer):
    tokenized_texts = []
    for text in tqdm(dataset['combined'].tolist(), desc="Tokenizing"):
        text = text.lower()
        tokenized_texts.append(tokenizer(text, truncation=True, padding='max_length', max_length=1024, return_tensors='pt'))
    return tokenized_texts

class BioGPTDataset(Dataset):
    def __init__(self, encodings, tokenizer):
        self.encodings = encodings
        self.pad_token_id = tokenizer.pad_token_id

    def __getitem__(self, idx):
        item = {key: torch.squeeze(val) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.cat([item['input_ids'][1:], torch.tensor([self.pad_token_id])])
        return item

    def __len__(self):
        return len(self.encodings)

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    tqdm.pandas()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("########################### PREPROCESSING ########################")
    # Data Preprocessing
    medical_data = pd.read_json("datasets/final.json")
    medical_data['combined'] = "Question : " + medical_data['question'] + " Answer : " + medical_data['answer']
    medical_data['word_count'] = medical_data['combined'].progress_apply(word_count)
    medical_data = medical_data[medical_data['word_count'] < 1020]
    print("########################### TOKENIZATION ########################")
    # Splitting the dataframe
    train_med, test_med = train_test_split(medical_data, test_size=0.2)
    val_med, test_med = train_test_split(test_med, test_size=0.5)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    tokenized_train = tokenize_dataset(train_med, tokenizer)
    tokenized_val = tokenize_dataset(val_med, tokenizer)
    tokenized_test = tokenize_dataset(test_med, tokenizer)

    # Dataset and DataLoader
    train_dataset = BioGPTDataset(tokenized_train, tokenizer)
    val_dataset = BioGPTDataset(tokenized_val, tokenizer)
    test_dataset = BioGPTDataset(tokenized_test, tokenizer)


    # Initialize the model
    model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt").to(device)

    # Training Arguments and Trainer Initialization
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=6000,
        weight_decay=0.01,
        learning_rate= 1e-4,
        logging_dir='./logs',
        logging_steps=1000,
        evaluation_strategy="steps",
        save_steps=8000,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()

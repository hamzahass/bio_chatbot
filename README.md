# Advancing Medical Diagnostics: A Dual Approach with Fine-Tuning and Retrieval Augmented Generation in Large Language Models

This project investigates the innovative application of Large Language Models (LLMs) in the realm of medical knowledge extraction and diagnosis. 

Our objective is two-fold: first, to develop a sophisticated medical assistant tool designed to aid doctors in diagnosing patients more efficiently and accurately; and second, to compare the effectiveness of fine-tuning LLMs against a more cost-effective approach using a vector database. 

We have leveraged the fine-tuning of advanced models, specifically BioGPT, alongside a meticulously curated dataset from BioASQ, MedQuAD, and PubMed, structured in a question-answer format with contextual information. This dataset provides a rich source of medical knowledge, deeply rooted in real-world contexts. 

To optimize the training process, we employed Low-Rank Adaptation (LoRA), a method that significantly accelerates fine-tuning of large models while reducing computational memory requirements and overall training time. 

However, due to the immense size of models like Meditron 7B and our limited hardware resources, we opted for a vector database approach. This database stores the embeddings of contexts and can retrieve the most relevant one for given queries, and then inputting this information into the LLM. This method utilizes the LLM's ability to understand and generate language based on the provided context, offering a practical way to customize responses or analyses without extensive retraining. Our project aims to set a new standard in the application of LLMs for medical purposes, enhancing the diagnostic capabilities of healthcare professionals.


### Data

The data can be accesses through the following [link](https://drive.google.com/drive/folders/1C9xZqJB2pIShfn62ArqRMf_A0i0zqZZB?usp=sharing) (google drive). The downloaded data folder can replace the one in the repo.

import os
import re
import torch
import emoji
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import logging as log
log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s : %(message)s")

class YouTubeCommentGenerator:
    def __init__(self, model_name="gpt2", dataset=None, model_path="./fine_tuned_youtube_model"):
        """
        Initializes the model and tokenizer.
        Loads the model from the given path if it exists, otherwise loads the base GPT2 model.
        """
        self.dataset = None
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure padding token is set

        # Initially load the base GPT2 model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        log.info(f"Loaded base GPT-2 model from {model_name}")

    def clean_text(self, text):
        """
        Cleans input text by removing extra spaces, special characters, and fixing common issues.
        Keep emoji and URL since those attributes are common in spam
        removes only unnecessary special characters while keeping punctuation, numbers, and URLs."""
        text = emoji.replace_emoji(text)
        text = text.lower()
        # Step 1: Remove special characters but keep letters, numbers, spaces, and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,=!?\'":;/\-]', '', text)

        # Step 2: Remove excessive spaces
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'br /', '', text)
        return text

    def load_dataset(self, dataset):
        """
        Loads and processes the dataset from a list of strings or CSV file.
        """
        if isinstance(dataset, list):  # If dataset is a list of comments
            df = pd.DataFrame(dataset, columns=["CONTENT"])
        elif isinstance(dataset, str):  # If dataset is a CSV file path
            df = pd.read_csv(dataset)
        else:
            raise ValueError("Dataset must be either a list of comments or a CSV file path.")

        dataset = Dataset.from_pandas(df)

        # Tokenize the dataset
        def tokenize_function(examples):
            """Cleans and tokenizes text data."""
            cleaned_texts = [self.clean_text(text) for text in examples["CONTENT"]]
            return self.tokenizer(cleaned_texts, truncation=True, padding="max_length", max_length=128)

        dataset = dataset.map(tokenize_function, batched=True)
        return dataset

    def fine_tune(self, epochs=3, batch_size=2, output_dir="./fine_tuned_youtube_model"):
        """
        Fine-tunes the GPT-2 model on the dataset.
        """
        if not hasattr(self, 'dataset') or not self.dataset:
            raise ValueError("Dataset not loaded. Provide a dataset before fine-tuning.")

        # Split the dataset into train and eval using train_test_split() of Hugging Face Datasets
        dataset_split = self.dataset.train_test_split(test_size=0.1)  # 90% for training, 10% for validation
        train_dataset = dataset_split['train']
        val_dataset = dataset_split['test']

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",  # Evaluate at the end of every epoch
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # Provide the validation dataset here
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        trainer.train()
        log.info("Fine-tuning completed.")
        self.save_model(output_dir)

    def generate_comment(self, prompt="This video is"):
        """
        Generates a YouTube-style comment based on the given prompt.
        """
        prompt = self.clean_text(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50, num_return_sequences=1,
                                      no_repeat_ngram_size=2, temperature=0.8, top_k=50, top_p=0.95, do_sample=True,
                                      repetition_penalty=1.2, pad_token_id=self.tokenizer.eos_token_id)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_comment = self.clean_generated_text(generated_text[len(prompt):].strip())
        return generated_comment

    def save_model(self, path="fine_tuned_youtube_model"):
        """
        Saves the fine-tuned model and tokenizer.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        log.info(f"Model saved to {path}")

    def load_fine_tuned_model(self, path="fine_tuned_youtube_model"):
        """
        Loads a previously fine-tuned model.
        """
        if os.path.exists(path):
            self.model = GPT2LMHeadModel.from_pretrained(path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(path)
            log.info(f"Fine-tuned model loaded from {path}")
        else:
            log.info(f"Fine-tuned model not found at {path}. Proceeding with base model.")

    def clean_generated_text(self, text):
        """Removes unwanted trailing special characters from generated output."""
        text = text.strip()
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]+$', '', text)  # Remove trailing special characters
        return text

if __name__ == "__main__":
    # Example dataset (can be replaced with a CSV file or list of comments)
    sample_comments = "Youtube-Spam-Dataset.csv"

    # Initialize model with sample dataset
    generator = YouTubeCommentGenerator(dataset=sample_comments)

    # Check if fine-tuned model exists, and load it if present
    generator.load_fine_tuned_model()

    # If model is not fine-tuned yet, fine-tune it
    if not os.path.exists(generator.model_path):
        log.info("Fine-tuning the model...")
        generator.dataset = generator.load_dataset(sample_comments)  # Load dataset for fine-tuning
        # for i in range(100):
        #     print(f"{i+1}. {generator.dataset[i]['CONTENT']}")
        generator.fine_tune(epochs=3, batch_size=2)

    # Generate sample comments
    log.info("Generating YouTube Comments:")
    results = []
    clean_text_fn = generator.clean_text
    df = pd.read_csv("youtube_spam_detection/Youtube-Spam-Dataset.csv")
    for i in range(len(df)):
        prompt = clean_text_fn(df['CONTENT'].loc[i])
        for _ in range(5):
            generated_comment = generator.generate_comment(prompt)
            results.append(generated_comment)
    df_out = pd.DataFrame(results, columns=["CONTENT"])
    df_out.to_csv("generated_comments.csv")

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
    def __init__(self, model_name="gpt2", dataset=None, model_path=None, comment_type="spam"):
        """
        Initializes the model and tokenizer.
        Loads the model from the given path if it exists, otherwise loads the base GPT-2 model.
        """
        self.dataset = None
        self.comment_type = comment_type.lower()
        self.model_name = model_name
        # Set default model paths if not provided
        if model_path is None:
            if self.comment_type == "spam":
                self.model_path = "./fine_tuned_youtube_model_spam"
            else:
                self.model_path = "./fine_tuned_youtube_model_nonspam"
        else:
            self.model_path = model_path

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure padding token is set
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        log.info(f"Loaded base GPT-2 model from {model_name} for {self.comment_type} comments")

    def clean_text(self, text):
        """
        Enhanced cleaning for comments:
        - Converts emojis to colon-delimited text using demojize.
        - Converts text to lowercase.
        - Removes unwanted characters while preserving letters, numbers, spaces, punctuation,
          and URL/mention symbols.
        - Collapses multiple spaces and repeated punctuation.
        - Removes specific unwanted patterns like variations of "br /".
        """
        text = emoji.demojize(text)
        text = text.lower()
        allowed_chars = r"a-zA-Z0-9\s.,=!?\'\":;/\-\@\#\%\&"
        text = re.sub(f"[^{allowed_chars}]", '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s*br\s*/\s*', '', text)
        text = re.sub(r'([!?.])\1+', r'\1', text)
        return text

    def load_dataset(self, dataset):
        """
        Loads and processes the dataset from a list of strings or CSV file.
        If the CSV file contains a "label" column, it filters the data based on the desired comment type.
        For spam fine-tuning, only rows with label 1 are kept; for non-spam, only rows with label 0.
        After filtering, the "label" column is dropped so that it is not passed to the model.
        """
        if isinstance(dataset, list):
            df = pd.DataFrame(dataset, columns=["CONTENT"])
        elif isinstance(dataset, str):
            df = pd.read_csv(dataset)
        else:
            raise ValueError("Dataset must be either a list of comments or a CSV file path.")

        # If the dataset has labels, filter based on the desired comment type.
        if "label" in df.columns:
            if self.comment_type == "spam":
                df = df[df["label"] == 1]
            elif self.comment_type == "nonspam":
                df = df[df["label"] == 0]
            # Drop the label column for unsupervised LM fine-tuning.
            df = df.drop(columns=["label"])

        # Rename the comment column to "CONTENT" if needed.
        if "comment" in df.columns and "CONTENT" not in df.columns:
            df = df.rename(columns={"comment": "CONTENT"})

        dataset = Dataset.from_pandas(df)

        def tokenize_function(examples):
            """Cleans and tokenizes text data."""
            cleaned_texts = [self.clean_text(text) for text in examples["CONTENT"]]
            return self.tokenizer(cleaned_texts, truncation=True, padding="max_length", max_length=128)

        dataset = dataset.map(tokenize_function, batched=True)
        return dataset

    def fine_tune(self, epochs=3, batch_size=2, output_dir=None):
        """
        Fine-tunes the GPT-2 model on the filtered dataset.
        """
        if not hasattr(self, 'dataset') or not self.dataset:
            raise ValueError("Dataset not loaded. Provide a dataset before fine-tuning.")

        if output_dir is None:
            output_dir = self.model_path

        dataset_split = self.dataset.train_test_split(test_size=0.1)  # 90% train, 10% validation
        train_dataset = dataset_split['train']
        val_dataset = dataset_split['test']

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
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
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        trainer.train()
        log.info("Fine-tuning completed.")
        self.save_model(output_dir)

    def generate_comment(self, prompt=None):
        """
        Generates a YouTube-style comment based on the given prompt.
        Uses different default prompts based on the comment type.
        """
        if prompt is None:
            if self.comment_type == "nonspam":
                prompt = "i really enjoyed this video because"
            else:
                prompt = "click this link now to win a free"
        else:
            prompt = self.clean_text(prompt)

        if self.comment_type == "nonspam":
            prompt = prompt + " i think"
        else:
            prompt = prompt + " limited offer"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_comment = self.clean_generated_text(generated_text[len(prompt):].strip())
        return generated_comment

    def save_model(self, path=None):
        """
        Saves the fine-tuned model and tokenizer.
        """
        if path is None:
            path = self.model_path
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        log.info(f"Model saved to {path}")

    def load_fine_tuned_model(self, path=None):
        """
        Loads a previously fine-tuned model.
        """
        if path is None:
            path = self.model_path
        if os.path.exists(path):
            self.model = GPT2LMHeadModel.from_pretrained(path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(path)
            log.info(f"Fine-tuned model loaded from {path}")
        else:
            log.info(f"Fine-tuned model not found at {path}. Proceeding with base model.")

    def clean_generated_text(self, text):
        """Removes unwanted trailing special characters from generated output."""
        text = text.strip()
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]+$', '', text)
        return text


if __name__ == "__main__":
    # Example: Fine-tune two separate models—one for spam and one for nonspam.
    sample_comments = "Youtube-Spam-Dataset.csv"  # CSV file with columns "comment" and "label"

    # Fine-tune Spam Model
    spam_generator = YouTubeCommentGenerator(comment_type="spam", dataset=sample_comments)
    spam_generator.load_fine_tuned_model()  # Try loading if exists
    if not os.path.exists(spam_generator.model_path):
        log.info("Fine-tuning the spam model...")
        spam_generator.dataset = spam_generator.load_dataset(sample_comments)
        spam_generator.fine_tune(epochs=3, batch_size=2)

    log.info("Generating sample spam comments:")
    for i in range(5):
        log.info(spam_generator.generate_comment())

    # Fine-tune Non-Spam Model
    nonspam_generator = YouTubeCommentGenerator(comment_type="nonspam", model_path="./fine_tuned_youtube_model_nonspam",
                                                dataset=sample_comments)
    nonspam_generator.load_fine_tuned_model()  # Try loading if exists
    if not os.path.exists(nonspam_generator.model_path):
        log.info("Fine-tuning the non-spam model...")
        nonspam_generator.dataset = nonspam_generator.load_dataset(sample_comments)
        nonspam_generator.fine_tune(epochs=3, batch_size=2)

    log.info("Generating sample non-spam comments:")
    for i in range(5):
        log.info(nonspam_generator.generate_comment())

    # ---- Generate Synthetic Comments for All Original Comments ----
    original_df = pd.read_csv(sample_comments)
    synthetic_comments = []
    for idx, row in original_df.iterrows():
        original_text = row.get("comment", row.get("CONTENT", ""))
        label = row.get("label", None)
        if not original_text or label is None:
            continue
        if label == 1:
            for _ in range(5):
                synthetic = spam_generator.generate_comment(prompt=original_text)
                synthetic_comments.append({
                    "original_comment": original_text,
                    "synthetic_comment": synthetic,
                    "label": 1
                })
        elif label == 0:
            for _ in range(5):
                synthetic = nonspam_generator.generate_comment(prompt=original_text)
                synthetic_comments.append({
                    "original_comment": original_text,
                    "synthetic_comment": synthetic,
                    "label": 0
                })
    synthetic_df = pd.DataFrame(synthetic_comments)
    synthetic_df.to_csv("all_generated_comments.csv", index=False)
    log.info("All generated synthetic comments saved to all_generated_comments.csv")

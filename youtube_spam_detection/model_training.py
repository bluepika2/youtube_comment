import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def tokenize_function(examples, tokenizer):
    """Tokenizes dataset for Transformer models"""
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def prepare_datasets(train_dataset, test_dataset, model_name):
    """Tokenizes datasets"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Map the tokenize_function over the datasets
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]  # This should remove the 'text' column.
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # Set the format of the datasets so that only the necessary columns are passed to the model
    # We assume that "label" is already present.
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_dataset, test_dataset

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F-1 score"""
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def train_model(model_name, train_dataset, test_dataset):
    """Train a Transformer model for spam detection"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, test_dataset = prepare_datasets(train_dataset, test_dataset, model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"./{model_name}-spam-model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(f"./{model_name}-spam-model")
    tokenizer.save_pretrained(f"./{model_name}-spam-model")

    return model, tokenizer

def load_or_train_model(model_name, train_dataset, test_dataset):
    """
    Checks if a model is already trained. If yes, loads the model; otherwise, trains it.
    """
    model_dir = f"./{model_name}-spam-model"
    if os.path.exists(model_dir):
        print(f"Found {model_dir} directory. Load the model without training")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return model, tokenizer
    else:
        print(f"Model directory {model_dir} not found. Training model {model_name}...")
        return train_model(model_name, train_dataset, test_dataset)
import re
import emoji
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


def clean_text(text):
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

def load_dataset(filepath="Youtube-Spam-Dataset.csv"):
    """Loads a dataset and preprocesses text."""
    df = pd.read_csv(filepath)  # Must contain 'comment' and 'label' columns
    df['label'] = df['label'].astype(int)
    df['comment'] = df['comment'].apply(clean_text)
    return df

def split_dataset(df, test_size=0.2):
    """Splits the dataset into training and test sets."""
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['comment'], df['label'], test_size=test_size, random_state=42
    )
    train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
    test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})
    return train_dataset, test_dataset
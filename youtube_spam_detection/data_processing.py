import re
import emoji
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Clean input text by:
      - Converting emojis to their textual representation (e.g., 😀 -> :grinning_face:)
      - Lowercasing the text
      - Removing unnecessary special characters while preserving letters, numbers, spaces,
        punctuation, emojis (in textual form), and common URL symbols.
      - Removing excessive whitespace and specific unwanted patterns.
    """
    # Convert emojis to text representation to preserve their meaning.
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text)
    text = text.lower()
    # Allowed characters: letters, numbers, whitespace, common punctuation, and URL symbols (@, #, %)
    allowed_chars = r"a-zA-Z0-9\s\.\,\=\!\?\'\":;/\-\@\#\%\&"
    text = re.sub(f"[^{allowed_chars}]", '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove specific unwanted pattern "br /" with optional surrounding spaces.
    text = re.sub(r'\s*br\s*/\s*', '', text)
    return text

def load_dataset(filepath: str = "Youtube-Spam-Dataset.csv") -> pd.DataFrame:
    """
    Loads a dataset from a CSV file and preprocesses the 'comment' column.
    The CSV file must contain 'comment' and 'label' columns.
    """
    df = pd.read_csv(filepath)
    if 'comment' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must contain 'comment' and 'label' columns.")
    df['label'] = df['label'].astype(int)
    df['comment'] = df['comment'].apply(clean_text)
    return df

def split_dataset(df, test_size=0.2):
    """Splits the dataset into training and test sets."""
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['comment'], df['label'], test_size=test_size, random_state=42, stratify=df['label']
    )
    train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
    test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})
    return train_dataset, test_dataset
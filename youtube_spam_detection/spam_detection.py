from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_name, model_path=None):
    """
    Loads a fine-tuned spam detection model.
    Args:
        model_name (str): The base model name (e.g., "distilbert-base-uncased").
        model_path (str, optional): The directory where the model is saved.
            If not provided, defaults to "./{model_name}-spam-model".

    Returns:
        model, tokenizer: The loaded model and tokenizer.
    """
    if model_path is None:
        model_path = f"./{model_name}-spam-model"
    else:
        model_path = model_path + f"/{model_name}-spam-model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def classify_comment(comment, model, tokenizer):
    """
        Classifies a YouTube comment as spam (1) or not spam (0).

        Args:
            comment (str): The input comment text.
            model: The fine-tuned classification model.
            tokenizer: The corresponding tokenizer.

        Returns:
            str: "Spam" if classified as spam, else "Not Spam".
    """
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
    # Ensure input_ids (and attention_mask if available) are passed to the model.
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")

    if input_ids is None:
        raise ValueError("Tokenizer did not return 'input_ids'.")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    prediction = torch.argmax(outputs.logits, axis=1).item()
    return "Spam" if prediction == 1 else "Not Spam"

def predict_comments(comments, model_name="distilbert-base-uncased"):
    """Predicts spam for a list of comments."""
    model, tokenizer = load_model(model_name)
    for comment in comments:
        result = classify_comment(comment, model, tokenizer)
        print(f"Comment: {comment}\nPrediction: {result}\n")
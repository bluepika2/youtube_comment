from transformers import Trainer, AutoTokenizer
from model_training import tokenize_function, load_or_train_model, compute_metrics

def prepare_dataset_for_evaluation(dataset, model_name):
    """Tokenizes and formats the dataset for evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset

def evaluate_model(model, test_dataset, model_name):
    """Evaluates a trained model on the test dataset."""
    prepared_test_dataset = prepare_dataset_for_evaluation(test_dataset, model_name)
    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    eval_results = trainer.evaluate(prepared_test_dataset)
    return eval_results

def evaluate_models(models, train_dataset, test_dataset):
    """Trains and evaluates multiple Transformer models."""
    results = {}

    for model_name in models:
        print(f"\nTraining & Evaluating {model_name}...")

        # Train Model
        model, _ = load_or_train_model(model_name, train_dataset, test_dataset)
        # Evaluate Model
        metrics = evaluate_model(model, test_dataset, model_name)
        results[model_name] = metrics

        # Print Results
        print(f"\nResults for {model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value}")

    return results
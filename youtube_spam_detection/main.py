import pandas as pd
from data_processing import load_dataset, split_dataset
from model_training import train_model
from model_evaluation import evaluate_models
from spam_detection import predict_comments
import logging
log = logging.getLogger(__name__)

# Define dataset paths
ORIGINAL_DATASET_PATH = "Youtube-Spam-Dataset_org.csv"
GENERATED_DATASET_PATH = "all_generated_comments.csv"

# Load original dataset
log.info("Loading original dataset...")
df_original = pd.read_csv(ORIGINAL_DATASET_PATH)
# Ensure the original dataset has the required columns "comment" and "label"
if "comment" not in df_original.columns:
    df_original = df_original.rename(columns={"CONTENT": "comment"})

# Load generated synthetic comments dataset
log.info("Loading synthetic (generated) dataset...")
df_generated = pd.read_csv(GENERATED_DATASET_PATH)
# Our generated dataset has columns: "original_comment", "synthetic_comment", and "label"
# Rename "synthetic_comment" to "comment" to match the original dataset
df_generated = df_generated.rename(columns={"synthetic_comment": "comment"})

# Merge the original and synthetic datasets
log.info("Merging original and generated datasets...")
df_merged = pd.concat([df_original, df_generated[["comment", "label"]]], ignore_index=True)

# Optional: Check class distribution
spam_count = len(df_merged[df_merged["label"] == 1])
nonspam_count = len(df_merged[df_merged["label"] == 0])
log.info(f"Class distribution before balancing: Spam = {spam_count}, Non-Spam = {nonspam_count}")

# If the synthetic generation has dramatically increased one class, you might downsample the majority.
# For example, if spam is overrepresented:
if spam_count > nonspam_count:
    log.info("Downsampling spam class to balance the dataset...")
    df_spam = df_merged[df_merged["label"] == 1].sample(n=nonspam_count, random_state=42)
    df_nonspam = df_merged[df_merged["label"] == 0]
    df_merged = pd.concat([df_spam, df_nonspam], ignore_index=True)
elif nonspam_count > spam_count:
    log.info("Downsampling non-spam class to balance the dataset...")
    df_nonspam = df_merged[df_merged["label"] == 0].sample(n=spam_count, random_state=42)
    df_spam = df_merged[df_merged["label"] == 1]
    df_merged = pd.concat([df_spam, df_nonspam], ignore_index=True)

# Shuffle the merged dataset
df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
log.info(f"Merged dataset shape (balanced): {df_merged.shape}")

# Save merged dataset temporarily so that load_dataset (which expects a CSV path) can use it.
MERGED_DATASET_PATH = "merged_dataset.csv"
df_merged.to_csv(MERGED_DATASET_PATH, index=False)
log.info(f"Merged dataset saved to {MERGED_DATASET_PATH}")

# Load dataset using your data_processing module (which cleans and preprocesses the text)
log.info("Loading and preprocessing merged dataset...")
merged_df = load_dataset(MERGED_DATASET_PATH)

# Split into training & test sets
log.info("Splitting merged dataset into training & test sets...")
train_dataset, test_dataset = split_dataset(merged_df)

# Define models to evaluate
models_to_test = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]

# Train & Evaluate Models
log.info("Training & Evaluating models on merged dataset...")
evaluation_results = evaluate_models(models_to_test, train_dataset, test_dataset)

# Select the best model based on F1 score
best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['f1'])
best_metrics = evaluation_results[best_model_name]
log.info(f"The best model is {best_model_name} with metrics: {best_metrics}")

# Predict Spam on New Comments
new_comments = [
    "Click this link now to win a free iPhone! 🚀🔥 http://freemoney.xyz",
    "Great content, I really enjoyed your video!",
    "Earn $500 daily with this trick 👉 http://spamlink.com"
]

log.info("Predicting spam comments...")
predict_comments(new_comments)

log.info("Pipeline execution completed!")

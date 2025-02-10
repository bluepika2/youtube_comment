from data_processing import load_dataset, split_dataset
from model_training import train_model
from model_evaluation import evaluate_models
from spam_detection import predict_comments
import logging as log
log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s : %(message)s")

# Define dataset path
DATASET_PATH = "Youtube-Spam-Dataset_org.csv"

# Load Dataset
log.info("Loading dataset...")
df = load_dataset(DATASET_PATH)

# Split into Training & Test sets
log.info("Splitting dataset into training & test sets...")
train_dataset, test_dataset = split_dataset(df)

# Define Models to Evaluate
models_to_test = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]
# models_to_test = ["distilbert-base-uncased"]
# Train & Evaluate Models
log.info("Training & Evaluating models...")
evaluation_results = evaluate_models(models_to_test, train_dataset, test_dataset)

# Predict Spam on New Comments
new_comments = [
    "Click this link now to win a free iPhone! 🚀🔥 http://freemoney.xyz",
    "Great content, I really enjoyed your video!",
    "Earn $500 daily with this trick 👉 http://spamlink.com"
]

log.info("Predicting spam comments...")
predict_comments(new_comments)

log.info("Pipeline execution completed!")
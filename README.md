# YouTube Spam Detector

A full-stack application that analyzes YouTube video comments to detect spam, evaluate sentiment, and identify adult content. In addition, the project provides tools to train and evaluate spam detection models using Transformer-based architectures, as well as a component to generate synthetic YouTube comments.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [Running the Web Application](#running-the-web-application)
  - [Training & Evaluating Models](#training--evaluating-models)
  - [Generating Synthetic Comments](#generating-synthetic-comments)
- [Pushing the Model to Hugging Face](#pushing-the-model-to-hugging-face)
- [License](#license)

## Overview

This repository implements a YouTube comment analysis tool that leverages a Flask-based dashboard for interactive analysis of video statistics and comment data. The project provides a pipeline for data processing, training Transformer-based models for spam detection, and evaluating model performance using metrics such as accuracy, precision, recall, and F1 score. Additionally, it includes a module to generate synthetic YouTube comments using a fine-tuned GPT-2 model.

## Features

- **Comment Analysis:** Fetches YouTube video details and comments, then analyzes sentiment (using VADER), spam status, and potential adult content.
- **Interactive Dashboard:** A modern web interface with Bootstrap and Chart.js that displays video KPIs, charts for sentiment/spam/adult content analysis, recent video history, and more.
- **Spam Detection Models:** Implements training and evaluation of multiple Transformer models (e.g., DistilBERT, BERT, RoBERTa) for spam detection.
- **Synthetic Comment Generation:** Uses GPT-2 to generate synthetic YouTube comments, either as spam or non-spam.
- **Model Push:** Provides a script to push the trained model and tokenizer to the Hugging Face Hub.

## Project Structure

- **Application Files:**
  - `__init__.py` – Creates and configures the Flask application.
  - `routes.py` – Defines the Flask routes for processing YouTube URLs, fetching video details and comments, performing sentiment analysis, spam detection, and rendering results.
  - `result.html` & `index.html` – HTML templates for the dashboard and main page.
  - `run.py` – Entry point to run the Flask web server.

- **Model & Data Pipeline Files:**
  - `spam_detection.py` – Functions for loading Transformer models from Hugging Face and classifying comments as spam or not.
  - `final_model_save_push.py` – Script to prepare the final model and push it to the Hugging Face Hub.
  - `main.py` – Main pipeline that loads, preprocesses, and merges datasets, trains models, evaluates performance, and runs spam predictions.
  - `model_evaluation.py` – Contains functions to evaluate different Transformer models on the test dataset.
  - `model_training.py` – Implements the training pipeline for Transformer-based spam detection models.
  - `data_processing.py` – Provides functions for cleaning text data and preparing datasets for model training.
  - `youtube_comment_generator.py` – A module to fine-tune a GPT-2 model for generating synthetic YouTube comments.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/youtube-spam-detector.git
   cd youtube-spam-detector
   ```

2. **Set up a virtual environment(recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. **Install the required dependencies:**

    ```bash
   pip install -r requirements.txt
   ```
## Environment Variables

- **YOUTUBE_API_KEY:** Your YouTube Data API key.
- **UNSPLASH_ACCESS_KEY:** Your Unsplash API access key for fetching images.
- **INFERENCE_SERVICE_URL:** URL endpoint for remote spam classification (if using a remote inference service).
- **HUGGING_API_KEY:** Your Hugging Face API key for pushing models.

## Usage

### Running the Web Application

The web dashboard allows you to input a YouTube video URL, analyze comments, and view interactive charts. 

To run the Flask app:
```
python run.py
```
### Training & Evaluating Models

The pipeline in `main.py` handles dataset merging, training, and evaluation of Transformer-based models. 

To run the pipeline, simply execute:
```
python main.py
```
This will:

- Load the original and synthetic datasets.
- Merge and balance the data.
- Tokenize and split the dataset.
- Train and evaluate models such as DistilBERT, BERT, and RoBERTa.
- Print evaluation metrics for each model.
- Run sample spam predictions.

### Generating Synthetic Comments
The module `youtube_comment_generator.py` enables fine-tuning a GPT-2 model to generate synthetic YouTube comments (for both spam and non-spam categories).

To fine-tune and generate comments, run the module directly:
```
python youtube_comment_generator.py
```
The script will:

- Fine-tune models based on provided datasets.
- Generate sample synthetic comments.
- Save the generated comments to a CSV file (all_generated_comments.csv).

## Pushing the Model to Hugging Face
The script `final_model_save_push.py` is used to copy the final trained model files and push them to a repository on the Hugging Face Hub.
Before running, ensure you have set your HUGGING_API_KEY. Then run:
```
python final_model_save_push.py
```
This will log into Hugging Face, prepare the final model folder, and push both the model and tokenizer to your specified repo.

## License
This project is licensed under the MIT License.
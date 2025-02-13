from flask import Blueprint, render_template, request
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
import sys
import emoji
import requests
from youtube_spam_detection.spam_detection import load_model_from_hub, classify_comment  # Import spam detection functions

main = Blueprint('main', __name__)

# YouTube API Key
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
# Load the spam detection model (this assumes the model is already trained)
hf_token = os.getenv("HF_TOKEN")  # Ensure HF_TOKEN is set in Heroku config
repo_name = "bluepika2/youtube-spam-detection"  # Replace with your HF repo name
# Load the model only once
try:
    model, tokenizer = load_model_from_hub(repo_name, use_auth_token=hf_token)
except Exception as e:
    MODEL, TOKENIZER = None, None
    print(f"Error loading model: {e}")

# Add the 'youtube_spam_detection' directory to sys.path so we can import spam_detection.py
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'youtube_spam_detection')
sys.path.append(models_dir)

def get_video_id(url):
    """
    Extract video ID from YouTube URL
    :param url:
    :return:
    """
    pattern = (
        r"(?:https?:\/\/)?(?:www\.)?"
        r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)"
        r"([^\"&?\/\s]{11})"
    )
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_comments(video_id, max_comments=100):
    """
    Fetch comments from a YouTube video
    :param video_id:
    :return:
    """
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    video_response = youtube.videos().list(part='snippet', id=video_id).execute()
    video_snippet = video_response['items'][0]['snippet']
    uploader_channel_id = video_snippet['channelId']

    comments = []
    nextPageToken = None
    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=nextPageToken
            ).execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                if comment['authorChannelId']['value'] != uploader_channel_id:
                    comments.append(comment['textDisplay'])
            nextPageToken = response.get('nextPageToken')

            if not nextPageToken:
                break

        except Exception as e:
            return f"Error fetching comments: {e}"
    return comments

def sentiment_scores(comment, analyzer):
    sentiment_dict = analyzer.polarity_scores(comment)
    return sentiment_dict['compound']

def analyze_sentiment(comments):
    """
    Analyze sentiment of comments
    :param comments:
    :return:
    """
    hyperlink_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    threshold_ratio = 0.1
    relevant_comments = []
    # cleaned comments by filtering URL and too many emojis with threshold
    for comment_text in comments:
        comment_text = comment_text.lower().strip()
        emojis = emoji.emoji_count(comment_text)

        text_characters = len(re.sub(r'\s', '', comment_text))

        if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
            if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
                relevant_comments.append(comment_text)


    analyzer = SentimentIntensityAnalyzer()
    analyzed_comments = []

    for comment in relevant_comments:
        score = sentiment_scores(comment, analyzer)
        if score > 0.05:
            sentiment = "Positive"
        elif score < -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        analyzed_comments.append({"text": comment, "sentiment": sentiment})
    return analyzed_comments

def is_adult_content(comment):
    """
    Checks whether a comment might be pushing adult content.
    This example uses a simple keyword-based method.
    """
    adult_keywords = [
        "xxx", "porn", "adult video", "nude", "explicit", "sex", "hot", "adult"
    ]
    pattern = re.compile("|".join(adult_keywords), re.IGNORECASE)
    return bool(pattern.search(comment))

# Use the Unsplash API to fetch images.
def fetch_unsplash_images(query, per_page=1):
    unsplash_access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not unsplash_access_key:
        print("UNSPLASH_ACCESS_KEY not set.")
        return []

    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "per_page": per_page
    }
    headers = {
        "Authorization": f"Client-ID {unsplash_access_key}"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print("Error fetching images from Unsplash:", response.status_code, response.text)
        return []

    data = response.json()
    image_urls = []
    for photo in data.get("results", []):
        image_urls.append(photo["urls"]["regular"])
    return image_urls

@main.route("/", methods=["GET", "POST"])
def index():
    # Fetch hero and carousel images from Unsplash.
    hero_images = fetch_unsplash_images("youtube", per_page=1)
    carousel_images = fetch_unsplash_images("technology", per_page=3)
    if request.method == "POST":
        url = request.form['youtube_url']
        video_id = get_video_id(url)

        if not video_id:
            return render_template("index.html", error="Invalid YouTube URL",
                                   hero_images=hero_images, carousel_images=carousel_images)

        comments = fetch_comments(video_id, max_comments=100)

        if isinstance(comments, str): # Error case
            return render_template("index.html", error=comments,
                                   hero_images=hero_images, carousel_images=carousel_images)

        analyzed_comments = analyze_sentiment(comments)

        # Initialize aggregate counters.
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        spam_counts = {"Spam": 0, "Not Spam": 0}
        adult_counts = {"Adult Content": 0, "Non Adult": 0}
        repeated_counts = {}

        # Process each comment: classify spam and update aggregate counts.
        for item in analyzed_comments:
            # Classify the comment and add the spam key.
            spam_result = classify_comment(item["text"], model, tokenizer)
            item["spam"] = spam_result
            # Detect adult content.
            if is_adult_content(item["text"]):
                item["adult"] = "Adult Content"
            else:
                item["adult"] = "Non Adult"

            # Update aggregate counts.
            sentiment_counts[item["sentiment"]] += 1
            spam_counts[spam_result] += 1
            adult_counts[item["adult"]] += 1

            # Count repeated comments (using the original text).
            text = item["text"]
            repeated_counts[text] = repeated_counts.get(text, 0) + 1

            # Filter repeated comments: only keep those that appear more than once.
        repeated_comments = {k: v for k, v in repeated_counts.items() if v > 1}

        return render_template("result.html",
                               sentiment_counts=sentiment_counts,
                               spam_counts=spam_counts,
                               adult_counts=adult_counts,
                               repeated_comments=repeated_comments,
                               hero_images=hero_images,
                               carousel_images=carousel_images)
    return render_template("index.html", hero_images=hero_images, carousel_images=carousel_images)


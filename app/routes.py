from flask import Blueprint, render_template, request
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
import sys
import emoji
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
    threshold_ratio = 0.65
    relevant_comments = []
    # cleaned comments by filtering URL and too many emojis with threshold
    for comment_text in comments:
        comment_text = comment_text.lower().strip()
        emojis = emoji.emoji_count(comment_text)

        text_characters = len(re.sub(r'\s', '', comment_text))

        if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
            if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
                relevant_comments.append(comment_text)

    # polarity = []
    # positive_comments = []
    # negative_comments = []
    # neutral_comments = []
    # sentiments = {"positive": 0, "neutral": 0, "negative": 0}
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

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form['youtube_url']
        video_id = get_video_id(url)

        if not video_id:
            return render_template("index.html", error="Invalid YouTube URL")

        comments = fetch_comments(video_id, max_comments=100)

        if isinstance(comments, str): # Error case
            return render_template("index.html", error=comments)

        analyzed_comments = analyze_sentiment(comments)


            # For each comment, classify as Spam or Not Spam
        for item in analyzed_comments:
            spam_result = classify_comment(item["text"], model, tokenizer)
            item["spam"] = spam_result

            # Aggregate results for the dashboard.
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        spam_counts = {"Spam": 0, "Not Spam": 0}

        for item in analyzed_comments:
            sentiment_counts[item["sentiment"]] += 1
            spam_counts[item["spam"]] += 1

        # Pass the aggregated counts to the dashboard template.
        return render_template("result.html",
                               sentiment_counts=sentiment_counts,
                               spam_counts=spam_counts)
    return render_template("index.html")

# def sentiment_scores(comment, polarity):
#     sentiment_object = SentimentIntensityAnalyzer()
#     sentiment_dict = sentiment_object.polarity_scores(comment)
#     polarity.append(sentiment_dict['compound'])
#     return polarity
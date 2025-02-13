from flask import Blueprint, render_template, request
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
import sys
import emoji
import requests
from youtube_spam_detection.spam_detection import load_model_from_hub, classify_comment  # Import spam detection functions

# Add the 'models' directory (which contains spam_detection.py) to sys.path.
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'models')
sys.path.append(models_dir)

main = Blueprint('main', __name__)

# YouTube API Key from environment variables
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

def get_video_id(url):
    pattern = (
        r"(?:https?:\/\/)?(?:www\.)?"
        r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)"
        r"([^\"&?\/\s]{11})"
    )
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_video_details(video_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    response = youtube.videos().list(part='snippet,statistics', id=video_id).execute()
    if response['items']:
        video = response['items'][0]
        snippet = video['snippet']
        stats = video.get('statistics', {})
        return {
            "title": snippet.get("title", "N/A"),
            "channelTitle": snippet.get("channelTitle", "N/A"),
            "description": snippet.get("description", ""),
            "publishedAt": snippet.get("publishedAt", "N/A"),
            "viewCount": stats.get("viewCount", "0"),
            "likeCount": stats.get("likeCount", "0"),
            "commentCount": stats.get("commentCount", "0")
        }
    return {}


def fetch_comments(video_id, max_comments=100):
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
                    comments.append({
                        "text": comment['textDisplay'],
                        "author": comment.get('authorDisplayName', 'Unknown'),
                        "author_channel_id": comment['authorChannelId']['value']
                    })
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
    analyzer = SentimentIntensityAnalyzer()
    # Process each comment dict; add a "sentiment" key.
    for comment in comments:
        cleaned = comment["text"].lower().strip()
        score = sentiment_scores(cleaned, analyzer)
        if score > 0.05:
            comment["sentiment"] = "Positive"
        elif score < -0.05:
            comment["sentiment"] = "Negative"
        else:
            comment["sentiment"] = "Neutral"
    return comments


def is_adult_content(comment):
    adult_keywords = ["xxx", "porn", "adult video", "nude", "explicit", "sex", "hot", "adult"]
    pattern = re.compile("|".join(adult_keywords), re.IGNORECASE)
    return bool(pattern.search(comment))


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
    image_urls = [photo["urls"]["regular"] for photo in data.get("results", [])]
    return image_urls


@main.route("/", methods=["GET", "POST"])
def index():
    # Fetch images for hero and carousel.
    hero_images = fetch_unsplash_images("youtube", per_page=1)
    carousel_images = fetch_unsplash_images("technology", per_page=3)

    if request.method == "POST":
        url = request.form['youtube_url']
        max_comments_str = request.form.get('max_comments', '100')
        try:
            max_comments = max(1, int(max_comments_str))
        except ValueError:
            max_comments = 100
        video_id = get_video_id(url)
        if not video_id:
            return render_template("index.html", error="Invalid YouTube URL",
                                   hero_images=hero_images, carousel_images=carousel_images)

        # Get video details.
        video_details = get_video_details(video_id)
        # Fetch comments (with author info).
        comments = fetch_comments(video_id, max_comments=max_comments)
        if isinstance(comments, str):
            return render_template("index.html", error=comments,
                                   hero_images=hero_images, carousel_images=carousel_images)

        analyzed_comments = analyze_sentiment(comments)

        # Load spam detection model from Hugging Face Hub.
        hf_token = os.getenv("HF_TOKEN")
        repo_name = "bluepika2/youtube-spam-detection"  # Replace with your HF repository name
        try:
            model, tokenizer = load_model_from_hub(repo_name, use_auth_token=hf_token)
        except Exception as e:
            return render_template("index.html", error=f"Error loading spam detection model: {e}",
                                   hero_images=hero_images, carousel_images=carousel_images)

        # Initialize counters.
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        spam_counts = {"Spam": 0, "Not Spam": 0}
        adult_counts = {"Adult Content": 0, "Non Adult": 0}
        repeated_counts = {}
        author_counts = {}

        for item in analyzed_comments:
            spam_result = classify_comment(item["text"], model, tokenizer)
            item["spam"] = spam_result
            sentiment_counts[item["sentiment"]] += 1
            spam_counts[spam_result] += 1
            if is_adult_content(item["text"]):
                item["adult"] = "Adult Content"
            else:
                item["adult"] = "Non Adult"
            adult_counts[item["adult"]] += 1
            # Count repeated comments.
            text = item["text"]
            repeated_counts[text] = repeated_counts.get(text, 0) + 1
            # Count authors.
            author = item.get("author", "Unknown")
            author_counts[author] = author_counts.get(author, 0) + 1

        repeated_comments = {k: v for k, v in repeated_counts.items() if v > 1}

        return render_template("result.html",
                               video_details=video_details,
                               sentiment_counts=sentiment_counts,
                               spam_counts=spam_counts,
                               adult_counts=adult_counts,
                               repeated_comments=repeated_comments,
                               author_counts=author_counts,
                               hero_images=hero_images,
                               carousel_images=carousel_images)
    return render_template("index.html", hero_images=hero_images, carousel_images=carousel_images)

from flask import Blueprint, render_template, request, session
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
import sys
import html
import emoji
import requests
from youtube_spam_detection.spam_detection import load_model_from_hub, classify_comment  # Import spam detection functions
from youtube_spam_detection.data_processing import clean_text

# Add the 'models' directory (which contains spam_detection.py) to sys.path.
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'models')
sys.path.append(models_dir)

main = Blueprint('main', __name__)

# YouTube API Key from environment variables
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

def prepare_display_text(text: str) -> str:
    """
    Prepares a version of the text for display by unescaping HTML entities
    and removing HTML tags like <br>.
    """
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities (e.g., &#39; -> ')
    text = html.unescape(text)
    # Remove <br> tags (and similar HTML tags)
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def get_video_id(url):
    pattern = (
        r"(?:https?:\/\/)?(?:www\.)?"
        r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)"
        r"([^\"&?\/\s]{11})"
    )
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_video_details(video_id):
    try:
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
    except Exception as e:
        print(f"Error fetching video details: {e}")
    return {}

def fetch_comments(video_id, max_comments=100):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    comments = []
    nextPageToken = None

    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=nextPageToken
            ).execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    "text": comment['textDisplay'],
                    "author": comment.get('authorDisplayName', 'Unknown'),
                })

            nextPageToken = response.get('nextPageToken')
            if not nextPageToken:
                break
        except Exception as e:
            print(f"Error fetching comments: {e}")
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

def is_spam(comment):
    spam_keywords = ["subscribe", "giveaway", "click here", "earn money", "visit my channel"]
    return "Spam" if any(keyword in comment.lower() for keyword in spam_keywords) else "Not Spam"

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

    # Ensure session is initialized
    session.setdefault('recent_videos', [])

    if request.method == "POST":
        url = request.form['youtube_url']
        max_comments_str = request.form.get('max_comments', '100')
        keyword_filter = request.form.get('keyword_filter', '').strip().lower()

        try:
            max_comments = max(1, int(max_comments_str))
        except ValueError:
            max_comments = 100

        video_id = get_video_id(url)
        if not video_id:
            return render_template("index.html", error="Invalid YouTube URL",
                                   hero_images=hero_images, carousel_images=carousel_images)

        # Save video to recent history
        if video_id not in session['recent_videos']:
            session['recent_videos'].insert(0, video_id)
            session['recent_videos'] = session['recent_videos'][:5]  # Keep only last 5

        # Get video details.
        video_details = get_video_details(video_id)

        # Fetch comments (with author info).
        comments = fetch_comments(video_id, max_comments=max_comments)
        if isinstance(comments, str):
            return render_template("index.html", error=comments,
                                   hero_images=hero_images, carousel_images=carousel_images)

        # Filter comments by keyword if specified
        if keyword_filter:
            comments = [c for c in comments if keyword_filter in c['text'].lower()]

        # After fetching comments
        for i, comment in enumerate(comments):
            # Store the original text
            comments[i]["original_text"] = comment["text"]
            # Clean for model input (already using your clean_text function)
            comments[i]["text"] = clean_text(comment["text"])
            # Prepare a display version that unescapes HTML entities and removes HTML tags
            comments[i]["display_text"] = prepare_display_text(comment["original_text"])

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
        repeated_comments = {}  # Dictionary for repeated comments with count and authors.
        author_counts = {}

        for item in analyzed_comments:
            spam_result = classify_comment(item["text"], model, tokenizer)
            rule_spam_result = is_spam(item["text"])
            item["spam"] = "Spam" if spam_result == "Spam" or rule_spam_result == "Spam" else "Not Spam"
            sentiment_counts[item["sentiment"]] += 1
            spam_counts[item["spam"]] += 1

            item["adult"] = "Adult Content" if is_adult_content(item["text"]) else "Non Adult"
            adult_counts[item["adult"]] += 1

            comment_text = item["display_text"]
            if comment_text not in repeated_comments:
                repeated_comments[comment_text] = {"count": 0, "authors": []}
            repeated_comments[comment_text]["count"] += 1
            author = item.get("author", "Unknown")
            if author not in repeated_comments[comment_text]["authors"]:
                repeated_comments[comment_text]["authors"].append(author)

            author_counts[author] = author_counts.get(author, 0) + 1

        # Keep only repeated comments (appearing more than once)
        repeated_comments = {k: v for k, v in repeated_comments.items() if v["count"] > 1}

        # Sort authors by number of comments and get top 5.
        top_commenters = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        session['recent_videos'] = [video_id] + session.get('recent_videos', [])[:4]

        return render_template("result.html",
                               video_details=video_details,
                               sentiment_counts=sentiment_counts,
                               spam_counts=spam_counts,
                               adult_counts=adult_counts,
                               repeated_comments=repeated_comments,
                               top_commenters=top_commenters,
                               recent_videos=session['recent_videos'],
                               keyword_filter=keyword_filter,
                               hero_images=hero_images,
                               carousel_images=carousel_images,
                               analyzed_comments=analyzed_comments)
    return render_template("index.html", hero_images=hero_images, carousel_images=carousel_images)

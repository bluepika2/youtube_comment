from flask import Flask, render_template, request
from pytube import YouTube

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    video_info = None
    if request.method == 'POST':
        url = request.form['youtube_url']
        try:
            yt = YouTube(url)
            video_info = {
                'title': yt.title,
                'thumbnail': yt.thumbnail_url,
                'length': yt.length,
                'author': yt.author,
                'views': yt.views
            }
        except Exception as e:
            video_info = {'error': str(e)}

    return render_template('index.html', video_info=video_info)

if __name__ == '__main__':
    app.run(debug=True)
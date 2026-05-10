# youtube_api.py
import re
from googleapiclient.discovery import build

class YouTubeDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        # Ініціалізація клієнта YouTube API
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)

    def extract_video_id(self, url):
        """Парсинг video_id з різних форматів URL YouTube."""
        pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None

    def get_video_data(self, url):
        """Отримання статистики та коментарів відео."""
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError("Невірний формат URL YouTube.")

        # 1. Отримання базової статистики
        video_request = self.youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        video_response = video_request.execute()
        
        if not video_response['items']:
            raise ValueError("Відео не знайдено.")

        stats = video_response['items'][0]['statistics']
        snippet = video_response['items'][0]['snippet']

        video_info = {
            'title': snippet['title'],
            'views': int(stats.get('viewCount', 0)),
            'likes': int(stats.get('likeCount', 0)),
            'comments_count': int(stats.get('commentCount', 0))
        }

        # 2. Отримання коментарів (до 100 штук для аналізу)
        comments = []
        try:
            comment_request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText"
            )
            comment_response = comment_request.execute()

            for item in comment_response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
        except Exception as e:
            print(f"Коментарі вимкнені або недоступні: {e}")

        return video_info, comments
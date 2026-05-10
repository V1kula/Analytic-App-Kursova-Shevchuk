# nlp_processing.py
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# завантаження необхідних словників NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text, language='english'):
    """Очищення, токенізація та видалення стоп-слів з тексту."""
    # Зведення до нижнього регістру
    text = str(text).lower()
    
    # видалення URL-посилань, спеціальних символів та цифр
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-ЯіІїЇєЄ\s]', '', text)
    
    # токенізація
    tokens = word_tokenize(text)
    
    # видалення стоп-слів
    stop_words = set(stopwords.words(language))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

def process_comments_list(comments):
    """Обробка списку коментарів та повернення DataFrame."""
    df = pd.DataFrame(comments, columns=['raw_text'])
    df['clean_text'] = df['raw_text'].apply(preprocess_text)
    return df
# sentiment_analysis.py
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import nltk

nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    translator = GoogleTranslator(source='auto', target='en')

    def get_sentiment(text):
        if not text or len(text.strip()) < 3:
            return 'Нейтральний'
        
        try:
            # переклад на англійську для точного аналізу
            translated = translator.translate(text)
            score = analyzer.polarity_scores(translated)['compound']
            
            if score >= 0.05:
                return 'Позитивний'
            elif score <= -0.05:
                return 'Негативний'
            else:
                return 'Нейтральний'
        except:
            # якщо переклад не вдався, то повертаємо нейтральний
            return 'Нейтральний'

    # Застосовуємо до колонки з текстом
    df['sentiment'] = df['raw_text'].apply(get_sentiment)
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    
    return df, sentiment_counts
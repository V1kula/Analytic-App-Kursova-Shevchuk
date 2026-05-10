# ml_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class PopularityPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def train_dummy_model(self):
        """Створення синтетичного датасету для навчання моделі."""
        np.random.seed(42)
        # генерація 500 записів: перегляди, лайки, коментарі, % позитиву
        views = np.random.randint(1000, 5000000, 500)
        likes = views * np.random.uniform(0.01, 0.1, 500)
        comments = likes * np.random.uniform(0.05, 0.2, 500)
        pos_percent = np.random.uniform(20, 90, 500)
        
        # логіка визначення популярності: високий engagement або багато переглядів з позитивом
        engagement_rate = likes / views
        target = (
            ((views > 1000000) & (engagement_rate > 0.01)) |  
            ((views > 100000) & (engagement_rate > 0.03)) |   
            ((engagement_rate > 0.07) & (pos_percent > 60))   
            ).astype(int)

        X = pd.DataFrame({
            'views': views, 'likes': likes, 'comments_count': comments, 'pos_percent': pos_percent
        })
        y = target

        # тренування
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict_popularity(self, views, likes, comments, pos_percent):
        """Прогноз популярності для конкретного відео."""
        if not self.is_trained:
            self.train_dummy_model()
            
        features = pd.DataFrame({
            'views': [views], 'likes': [likes], 'comments_count': [comments], 'pos_percent': [pos_percent]
        })
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]
        
        return "Популярне 🔥" if prediction == 1 else "Не популярне 📉", probability
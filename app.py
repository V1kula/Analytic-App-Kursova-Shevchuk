# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import time 
from youtube_api import YouTubeDataFetcher
from nlp_processing import process_comments_list
from sentiment_analysis import analyze_sentiment
from ml_model import PopularityPredictor

# аналітичний модуль
def analyze_strengths_weaknesses(stats, sentiment_counts):
    """Евристичний аналіз сильних та слабких сторін контенту."""
    strengths, weaknesses = [], []
    engagement_rate = (stats['likes'] / stats['views']) * 100 if stats['views'] > 0 else 0
    pos_percent = sentiment_counts.get('Позитивний', 0)
    neg_percent = sentiment_counts.get('Негативний', 0)

    # аналіз сильних сторін
    if engagement_rate > 3: 
        strengths.append(f"Високий рівень залучення: {engagement_rate:.1f}% (норма 1-2%).")
    if pos_percent > 60:
        strengths.append(f"Висока лояльність аудиторії: {pos_percent:.1f}% позитивних відгуків.")
    if stats['comments_count'] > (stats['views'] * 0.005):
        strengths.append("Сильна дискусійна активність у коментарях.")

    # Аналіз слабких сторін
    if engagement_rate < 0.8: 
        weaknesses.append(f"Дуже низька конверсія у лайки: {engagement_rate:.1f}%.")
    if neg_percent > 20:
        weaknesses.append(f"Високий рівень токсичності/негативу: {neg_percent:.1f}% коментарів.")
    if stats['views'] > 100000 and stats['comments_count'] < 100:
        weaknesses.append("Відео переглядають, але воно не стимулює до обговорення.")

    return strengths, weaknesses

# інтерфейс streamlit
st.set_page_config(page_title="Video Analytics API", layout="wide")
st.title("📊 Аналіз та прогноз популярності YouTube відео")

API_KEY = "AIzaSyCe4C8lhPmPMelbeTLcmUC7Jqqe_51gkbk"
fetcher = YouTubeDataFetcher(API_KEY)
predictor = PopularityPredictor()

url_input = st.text_input("Введіть URL відео YouTube:")

if st.button("Аналізувати"):
    # відлік часу
    start_time = time.time() 

    with st.spinner("Збір та обробка даних..."):
        try:
            # збір даних
            stats, raw_comments = fetcher.get_video_data(url_input)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Перегляди", f"{stats['views']:,}")
            col2.metric("Лайки", f"{stats['likes']:,}")
            col3.metric("Коментарі", f"{stats['comments_count']:,}")

            if raw_comments:
                # обробка та sentiment analysis
                df_clean = process_comments_list(raw_comments)
                df_sentiment, sentiment_dist = analyze_sentiment(df_clean)
                pos_percent = sentiment_dist.get('Позитивний', 0)

                # прогнозування
                prediction, prob = predictor.predict_popularity(
                    stats['views'], stats['likes'], stats['comments_count'], pos_percent
                )
                
                st.subheader(f"🤖 Прогноз моделі: {prediction} (Ймовірність: {prob:.1%})")

                # візуалізація
                st.markdown("### Візуалізація результатів")
                c1, c2 = st.columns(2)
                
                with c1:
                    fig_pie = px.pie(
                        values=sentiment_dist.values, 
                        names=sentiment_dist.index, 
                        title="Розподіл тональності коментарів",
                        color=sentiment_dist.index,
                        color_discrete_map={'Позитивний':'#2ecc71', 'Нейтральний':'#95a5a6', 'Негативний':'#e74c3c'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    metrics_df = pd.DataFrame({
                        'Метрика': ['Перегляди', 'Лайки', 'Коментарі'],
                        'Кількість': [stats['views'], stats['likes'], stats['comments_count']]
                    })
                    # логарифмічна шкала через велику різницю між переглядами та коментарями
                    fig_bar = px.bar(metrics_df, x='Метрика', y='Кількість', log_y=True, title="Метрики (Логарифмічна шкала)")
                    st.plotly_chart(fig_bar, use_container_width=True)

                # аналіз сильних та слабких сторін
                st.markdown("### 🔍 Аналітичний звіт")
                strengths, weaknesses = analyze_strengths_weaknesses(stats, sentiment_dist)
                
                st.success("**✅ Сильні сторони:**\n" + ("\n".join([f"- {s}" for s in strengths]) if strengths else "- Яскраво виражених сильних сторін не знайдено."))
                st.error("**⚠️ Слабкі сторони (зони росту):**\n" + ("\n".join([f"- {w}" for w in weaknesses]) if weaknesses else "- Критичних слабких сторін не знайдено."))

                # таблиця даних
                with st.expander("Показати оброблені коментарі (Таблиця)"):
                    st.dataframe(df_sentiment[['raw_text', 'sentiment']].head(10))

                # кінець часу та вивід результату
                execution_time = round(time.time() - start_time, 2)
                st.info(f"⏱️ **Час виконання аналізу:** {execution_time} секунд")
                print(f"--- Аналіз завершено за {execution_time} секунд ---") # Дублюємо в консоль

            else:
                st.warning("Коментарі під цим відео відключені або їх немає.")

        except Exception as e:
            st.error(f"Помилка: {e}")
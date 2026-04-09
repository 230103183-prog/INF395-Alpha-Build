import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
import pymorphy3
from scipy.sparse import hstack
import os

# --- 1. КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Anti-Fraud Audio AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 2. КЭШИРОВАНИЕ И ИНИЦИАЛИЗАЦИЯ (Senior Practice) ---
# Загружаем тяжелые ресурсы один раз при старте сервера
@st.cache_resource(show_spinner="Инициализация NLP ядра...")
def init_nlp():
    nltk.download('stopwords', quiet=True)
    morph = pymorphy3.MorphAnalyzer()
    ru_stopwords = stopwords.words('russian')
    # Тот же кастомный список, что и при обучении
    custom_stopwords = ru_stopwords + ['это', 'понимать', 'твой', 'алло', 'блядь', 'ваш', 'мочь', 'знать', 'наш',
                                       'нужно', 'ну', 'да', 'вот', 'так']
    return morph, custom_stopwords


@st.cache_resource(show_spinner="Загрузка весов модели...")
def load_models():
    try:
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        return tfidf, rf_model
    except FileNotFoundError:
        st.error(
            "❌ Ошибка: Не найдены файлы моделей. Убедитесь, что папка 'models' с файлами .pkl лежит рядом с app.py.")
        st.stop()


morph, custom_stopwords = init_nlp()
tfidf, rf_model = load_models()


# --- 3. ФУНКЦИИ ПРЕДОБРАБОТКИ ---
def clean_and_lemmatize(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zа-яё\s]', '', text)
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized_words)


def highlight_fraud_words(text):
    """Подсвечивает опасные слова для наглядности (Explainable AI)"""
    fraud_triggers = ['счёт', 'система', 'номер', 'код', 'деньга', 'тысяча', 'продиктовать', 'написать']
    highlighted = text
    for word in fraud_triggers:
        # Простая подсветка через HTML
        highlighted = re.sub(rf'(?i)({word}[а-я]*)',
                             r'<span style="color: red; font-weight: bold; background-color: #ffe6e6; padding: 2px; border-radius: 4px;">\1</span>',
                             highlighted)
    return highlighted


# --- 4. ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС (UI) ---
st.title("🛡️ Multimodal Fraud Detection System")
st.markdown("Система анализа телефонных звонков на основе транскрипции (NLP) и акустических метаданных (ASR).")

# Боковая панель для настроек
with st.sidebar:
    st.header("⚙️ Настройки модели")
    st.info("В Альфа-версии используется агрегированный порог предсказания Random Forest.")
    # Идеальный порог, который нашел твой ROC-AUC скрипт
    threshold = st.slider("Порог чувствительности (Threshold)", min_value=0.0, max_value=1.0, value=0.83, step=0.01)

    st.markdown("---")
    st.markdown("**Важные признаки для модели:**")
    st.caption(
        "1. ASR Confidence (Уверенность распознавания)\n2. Speech Rate (Слов в секунду)\n3. Наличие паттернов мошенников в тексте")

# Главная рабочая область
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Ввод данных звонка")
    st.info("Имитация пайплайна распознавания речи (ASR Mockup). Введите данные из логов звонка.")

    audio_text = st.text_area(
        "Транскрипция звонка (Text):",
        height=150,
        placeholder="Например: Здравствуйте, это служба безопасности банка. На вашем счету зафиксирована подозрительная активность. Продиктуйте код из смс..."
    )

    duration = st.number_input("Длительность аудио (секунды):", min_value=1.0, value=30.0, step=1.0)
    asr_conf = st.slider("Уверенность ASR (asr_conf_mean):", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

    analyze_btn = st.button("🔍 Проанализировать звонок", use_container_width=True, type="primary")

with col2:
    st.subheader("📊 Результаты анализа (Инференс)")

    if analyze_btn:
        if not audio_text.strip():
            st.warning("⚠️ Пожалуйста, введите транскрипцию звонка.")
        else:
            with st.spinner("Анализ мультимодальных признаков..."):
                # 1. Считаем производные фичи
                word_count = len(audio_text.split())
                speech_rate = word_count / (duration + 0.001)  # Защита от деления на 0

                # 2. NLP Пайплайн
                clean_text = clean_and_lemmatize(audio_text)
                X_text = tfidf.transform([clean_text])

                # 3. Подготовка числовых фичей (ВАЖНО: Порядок как при обучении!)
                # 'speech_rate', 'duration', 'asr_conf_mean'
                X_numeric = np.array([[speech_rate, duration, asr_conf]])

                # 4. Склейка
                X_combined = hstack([X_text, X_numeric]).tocsr()

                # 5. Предикт
                prob = rf_model.predict_proba(X_combined)[0][1]  # Берем вероятность класса 1 (Фрод)
                is_fraud = prob >= threshold

                # --- ВЫВОД РЕЗУЛЬТАТОВ ---
                if is_fraud:
                    st.error(f"🚨 **ВНИМАНИЕ: ОБНАРУЖЕН ФРОД!** (Вероятность: {prob:.1%})")
                else:
                    st.success(f"✅ **Звонок безопасен.** (Вероятность фрода: {prob:.1%})")

                st.progress(prob)

                # Метрики
                m1, m2, m3 = st.columns(3)
                m1.metric("Скорость речи", f"{speech_rate:.2f} слов/сек")
                m2.metric("Слов распознано", int(word_count))
                m3.metric("Уверенность ИИ", f"{asr_conf:.1%}")

                # Explainability (Интерпретируемость)
                st.markdown("### 🔍 Триггеры в тексте:")
                highlighted_text = highlight_fraud_words(audio_text)
                st.markdown(f"> {highlighted_text}", unsafe_allow_html=True)

    else:
        st.caption("Введите данные и нажмите кнопку анализа, чтобы увидеть результаты.")
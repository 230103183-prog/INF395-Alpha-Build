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
import whisper
import tempfile

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ---
@st.cache_resource(show_spinner="Initializing AI Models (NLP & ASR)... This may take a minute.")
def load_all_models():
    nltk.download('stopwords', quiet=True)
    morph = pymorphy3.MorphAnalyzer()
    stop_words = stopwords.words('russian')
    stop_words.extend(
        ['это', 'понимать', 'твой', 'алло', 'ваш', 'мочь', 'знать', 'наш', 'нужно', 'ну', 'да', 'вот', 'так'])

    # Загружаем наши веса Random Forest и TF-IDF
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    rf_model = joblib.load('models/rf_model.pkl')

    # Загружаем легкую модель Whisper для распознавания речи
    asr_model = whisper.load_model("tiny")

    return morph, stop_words, tfidf, rf_model, asr_model


morph, stop_words, tfidf, rf_model, asr_model = load_all_models()


# --- ФУНКЦИИ ---
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zа-яё\s]', '', text)
    words = text.split()
    lemmatized = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized)


def get_fraud_highlights(text):
    triggers = ['счёт', 'система', 'номер', 'код', 'деньга', 'тысяча', 'продиктовать', 'написать']
    highlighted = text
    for word in triggers:
        highlighted = re.sub(rf'(?i)({word}[а-я]*)',
                             r'<span style="color: #ff4b4b; font-weight: bold; background-color: rgba(255,75,75,0.1); padding: 2px; border-radius: 4px;">\1</span>',
                             highlighted)
    return highlighted


# --- ИНТЕРФЕЙС ---
st.title("🛡️ FraudGuard: Multimodal Audio Analysis")
st.markdown("AI-driven system for detecting telephone fraud using automated ASR transcription and NLP metadata.")

with st.sidebar:
    st.header("Model Configuration")
    threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.83, 0.01)
    st.divider()
    st.markdown("**Key Features:**")
    st.caption("1. Automated Audio Transcription\n2. Speech Rate (Words/Sec)\n3. Keyword Analysis")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Audio Input")

    audio_file = st.file_uploader("Upload Call Audio", type=['wav', 'mp3', 'm4a'])

    # Длительность пока вводим с плеера, чтобы не утяжелять проект библиотекой librosa
    duration = st.number_input("Audio Duration (seconds):", min_value=1.0, value=30.0,
                               help="Check the player below and enter the total seconds.")

    if audio_file:
        st.audio(audio_file)
        st.info("Audio loaded. Click the button to automatically transcribe and analyze.")

    asr_confidence = st.slider("ASR Confidence Score (Mockup):", 0.0, 1.0, 0.85)

    run_analysis = st.button("Transcribe & Analyze Call", use_container_width=True, type="primary")

with col2:
    st.subheader("2. AI Analysis Results")

    if run_analysis:
        if not audio_file:
            st.warning("⚠️ Please upload an audio file first.")
        else:
            transcription = ""
            # --- БЛОК АВТОМАТИЧЕСКОГО РАСПОЗНАВАНИЯ (WHISPER) ---
            with st.spinner("🎙️ Transcribing audio with Whisper AI... Please wait."):
                # Whisper нужен физический файл, поэтому создаем временный
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                    tmp.write(audio_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # Транскрибируем (language="ru" для ускорения и точности)
                    result = asr_model.transcribe(tmp_path, language="ru")
                    transcription = result["text"]
                finally:
                    os.unlink(tmp_path)  # Очищаем временный файл

            # Показываем результат распознавания
            st.text_area("Recognized Text (Auto-generated):", transcription, height=120)

            # --- БЛОК АНАЛИЗА НА ФРОД ---
            with st.spinner("🧠 Analyzing fraud risk..."):
                if not transcription.strip():
                    st.error("Model couldn't recognize any speech. Try another file.")
                else:
                    words = transcription.split()
                    word_count = len(words)
                    speech_rate = word_count / (duration + 0.001)

                    cleaned = preprocess_text(transcription)
                    x_text = tfidf.transform([cleaned])
                    x_num = np.array([[speech_rate, duration, asr_confidence]])
                    x_combined = hstack([x_text, x_num]).tocsr()

                    prob = rf_model.predict_proba(x_combined)[0][1]
                    is_fraud = prob >= threshold

                    if is_fraud:
                        st.error(f"🚨 FRAUD DETECTED (Probability: {prob:.1%})")
                    else:
                        st.success(f"✅ CALL SECURE (Fraud Probability: {prob:.1%})")

                    st.progress(prob)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Speech Rate", f"{speech_rate:.2f} w/s")
                    m2.metric("Word Count", word_count)
                    m3.metric("AI Confidence", f"{asr_confidence:.1%}")

                    st.markdown("### Risk Analysis")
                    highlighted = get_fraud_highlights(transcription)
                    st.markdown(f"**Trigger Words Found:**\n> {highlighted}", unsafe_allow_html=True)
    else:
        st.info("Awaiting input data for processing.")
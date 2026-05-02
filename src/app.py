import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import pymorphy3
import whisper
import os
import librosa
import warnings
import time
from scipy.sparse import hstack
import os

# ─────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #090d14;
    --surface:   #0f1520;
    --surface2:  #141d2e;
    --border:    #1e2d45;
    --accent:    #00d4ff;
    --accent2:   #0088cc;
    --danger:    #ff4757;
    --warn:      #ffa502;
    --safe:      #2ed573;
    --text:      #e8f0fe;
    --muted:     #6b7fa8;
    --font:      'Space Grotesk', sans-serif;
    --mono:      'JetBrains Mono', monospace;
}

/* ── Reset ── */
html, body, [class*="css"] {
    font-family: var(--font) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

/* ── Main container ── */
.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Custom header ── */
.fraud-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 2rem;
    background: linear-gradient(135deg, var(--surface) 0%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.fraud-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.fraud-header .logo-badge {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
}
.fraud-header h1 {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin: 0 !important;
    color: var(--text) !important;
}
.fraud-header .subtitle {
    font-size: 0.8rem;
    color: var(--muted);
    font-family: var(--mono);
    margin-top: 2px;
    letter-spacing: 0.05em;
}
.status-pill {
    margin-left: auto;
    background: rgba(46,213,115,0.1);
    border: 1px solid var(--safe);
    color: var(--safe);
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: var(--mono);
    letter-spacing: 0.08em;
    display: flex; align-items: center; gap: 6px;
}
.status-pill::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--safe);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 160px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 0 0 10px 10px;
}
.metric-card.accent::after  { background: var(--accent); }
.metric-card.danger::after  { background: var(--danger); }
.metric-card.warn::after    { background: var(--warn); }
.metric-card.safe::after    { background: var(--safe); }
.metric-label {
    font-size: 0.68rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    font-family: var(--mono);
}
.metric-value {
    font-size: 1.9rem; font-weight: 700;
    line-height: 1.1; margin: 0.2rem 0;
}
.metric-value.accent { color: var(--accent); }
.metric-value.danger { color: var(--danger); }
.metric-value.warn   { color: var(--warn); }
.metric-value.safe   { color: var(--safe); }
.metric-sub { font-size: 0.73rem; color: var(--muted); }

/* ── Verdict banner ── */
.verdict-fraud {
    background: linear-gradient(135deg, rgba(255,71,87,0.15), rgba(255,71,87,0.05));
    border: 1px solid var(--danger);
    border-left: 4px solid var(--danger);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.verdict-safe {
    background: linear-gradient(135deg, rgba(46,213,115,0.12), rgba(46,213,115,0.04));
    border: 1px solid var(--safe);
    border-left: 4px solid var(--safe);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.verdict-title {
    font-size: 1.2rem; font-weight: 700;
    letter-spacing: -0.01em;
}
.verdict-sub { font-size: 0.82rem; color: var(--muted); margin-top: 0.3rem; }

/* ── Risk timeline ── */
.phrase-row {
    display: flex; align-items: flex-start;
    gap: 1rem; padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    border-radius: 0;
    transition: background 0.15s;
}
.phrase-row:hover { background: rgba(255,255,255,0.02); }
.phrase-row:last-child { border-bottom: none; }
.phrase-time {
    font-family: var(--mono); font-size: 0.72rem;
    color: var(--muted); min-width: 100px;
    padding-top: 2px; white-space: nowrap;
}
.phrase-text { flex: 1; font-size: 0.88rem; line-height: 1.5; }
.risk-bar-wrap { width: 120px; padding-top: 4px; }
.risk-bar-bg {
    height: 6px; background: var(--border);
    border-radius: 3px; overflow: hidden;
}
.risk-bar-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.5s ease;
}
.risk-pct {
    font-family: var(--mono); font-size: 0.7rem;
    text-align: right; margin-top: 3px;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.7rem; font-family: var(--mono);
    color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.12em; margin: 1.5rem 0 0.8rem;
    display: flex; align-items: center; gap: 8px;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: var(--border);
}

/* ── Cards & panels ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}
.panel-header {
    padding: 0.85rem 1.2rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.78rem; font-family: var(--mono);
    color: var(--accent); text-transform: uppercase;
    letter-spacing: 0.08em; display: flex;
    align-items: center; gap: 6px;
}

/* ── Model selector radio ── */
.stRadio > label { font-size: 0.82rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.82rem !important; }

/* ── Sliders ── */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #000 !important;
    border: none !important;
    font-weight: 600 !important;
    font-family: var(--font) !important;
    letter-spacing: 0.02em;
    padding: 0.6rem 2rem !important;
    border-radius: 8px !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    font-family: var(--font) !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background-color: var(--accent) !important; }
.stProgress { background: var(--surface2) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 8px 8px 0 0;
    gap: 0;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.7rem 1.2rem !important;
    color: var(--muted) !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px;
    padding: 1rem !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}

/* ── Line chart override ── */
[data-testid="stVegaLiteChart"] canvas { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="fraud-header">
    <div class="logo-badge">🛡️</div>
    <div>
        <h1>FraudShield AI</h1>
        <div class="subtitle">TELEPHONE FRAUD ANALYZER · v2.0</div>
    </div>
    <div class="status-pill">SYSTEM ACTIVE</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_ml_components():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    data_grouping = joblib.load(os.path.join(MODELS_DIR, 'fraud_model_rf_nlp.pkl'))
    data_segments = joblib.load(os.path.join(MODELS_DIR, 'fraud_model_segments.pkl'))
    data_audio    = joblib.load(os.path.join(MODELS_DIR, 'fraud_model_audio.pkl'))
    data_hybrid   = joblib.load(os.path.join(MODELS_DIR, 'fraud_model_hybrid.pkl'))
    
    stt_model     = whisper.load_model("base")
    morph         = pymorphy3.MorphAnalyzer()

    return data_grouping, data_segments, data_audio, data_hybrid, stt_model, morph

try:
    data_grouping, data_segments, data_audio, data_hybrid, whisper_model, morph = load_ml_components()
    models_loaded = True
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    models_loaded = False
    st.stop()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def clean_and_lemmatize(text):
    text = re.sub(r'[^a-zа-яё\s]', '', text.lower())
    words = text.split()
    return ' '.join([morph.parse(w)[0].normal_form for w in words])

@st.cache_data(show_spinner=False)
def transcribe_audio(audio_bytes):
    temp_path = "temp_cache_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    result = whisper_model.transcribe(temp_path, language="ru") # Keep RU as per model training
    return result["segments"], temp_path

@st.cache_data(show_spinner=False)
def extract_audio_features(audio_path, start_time, end_time):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time - start_time)
            if len(y) == 0:
                return np.zeros(15)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=300)
            pitch_mean = np.nanmean(f0) if np.any(voiced_flag) else 0
            rms = librosa.feature.rms(y=y)
            rms_mean = np.mean(rms)
            return np.hstack([mfccs_mean, pitch_mean, rms_mean])
    except Exception:
        return np.zeros(15)

def risk_color(val):
    if val >= 0.75:   return "#ff4757"
    elif val >= 0.50: return "#ffa502"
    elif val >= 0.30: return "#ffdd59"
    else:             return "#2ed573"

def risk_label(val):
    if val >= 0.75:   return "HIGH"
    elif val >= 0.50: return "MEDIUM"
    elif val >= 0.30: return "LOW"
    else:             return "NORMAL"

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)

    MODEL_OPTIONS = {
        "NLP — Entire Call":       ("1", data_grouping, 0.83),
        "NLP — Phrasal Analysis": ("2", data_segments, 0.70),
        "Audio — Voice Only":      ("3", data_audio,    0.65),
        "Hybrid — NLP + Audio":    ("4", data_hybrid,   0.79),
    }

    model_name = st.radio("Analysis Algorithm:", list(MODEL_OPTIONS.keys()))
    model_id, active_data, default_thresh = MODEL_OPTIONS[model_name]

    acc = active_data.get('accuracy', 0.0)
    st.markdown(f"""
    <div class="metric-card accent" style="margin:0.8rem 0;">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value accent">{acc:.1%}</div>
        <div class="metric-sub">{model_name}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Detection Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("", 0.0, 1.0, default_thresh, 0.01,
                          help="Fraud probability at which a call is marked as dangerous")

    st.markdown(f"""
    <div style="background:var(--surface2); border:1px solid var(--border); border-radius:8px;
                padding:0.7rem 1rem; font-family:var(--mono); font-size:0.75rem; color:var(--muted);">
        Threshold: <span style="color:var(--warn);">{threshold:.0%}</span><br>
        Below → <span style="color:var(--safe);">NORMAL</span><br>
        Above → <span style="color:var(--danger);">FRAUD</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">System Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:var(--muted); line-height:1.7;">
    🎙️ Speech-to-Text: <b style="color:var(--text)">Whisper</b><br>
    🌲 Classifier: <b style="color:var(--text)">Random Forest</b><br>
    📊 Features: <b style="color:var(--text)">TF-IDF + MFCC</b><br>
    🔤 Lemmatization: <b style="color:var(--text)">pymorphy3</b>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# UPLOAD ZONE
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Audio Upload</div>', unsafe_allow_html=True)

col_up, col_info = st.columns([2, 1])
with col_up:
    uploaded_audio = st.file_uploader(
        "Drag and drop file or click to browse",
        type=["wav", "mp3"],
        label_visibility="collapsed"
    )
with col_info:
    st.markdown("""
    <div class="panel" style="padding:1rem; font-size:0.78rem; color:var(--muted); height:100%;">
    <b style="color:var(--accent); font-family:var(--mono);">SUPPORTED FORMATS</b><br><br>
    🎵 WAV — Lossless (recommended)<br>
    🎧 MP3 — Compressed audio<br><br>
    <span style="color:var(--muted); font-size:0.72rem;">Recommended: 16 kHz, Mono, ≥ 10 sec</span>
    </div>
    """, unsafe_allow_html=True)

if uploaded_audio is not None:
    audio_bytes = uploaded_audio.getvalue()

    col_audio, col_meta = st.columns([3, 1])
    with col_audio:
        st.audio(audio_bytes, format='audio/wav')
    with col_meta:
        size_kb = len(audio_bytes) / 1024
        st.markdown(f"""
        <div class="panel" style="padding:0.9rem;">
            <div class="metric-label">File</div>
            <div style="font-size:0.82rem; margin-top:4px; color:var(--text); word-break:break-all;">{uploaded_audio.name}</div>
            <div class="metric-label" style="margin-top:0.7rem;">Size</div>
            <div style="font-size:0.82rem; color:var(--accent);">{size_kb:.1f} KB</div>
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("🎙️ Whisper is transcribing speech..."):
        segments_data, temp_audio_path = transcribe_audio(audio_bytes)

    total_duration = max((s['end'] for s in segments_data), default=0)
    total_words    = sum(len(s['text'].split()) for s in segments_data)
    avg_conf       = np.mean([np.exp(s.get('avg_logprob', -1)) for s in segments_data if s['text'].strip()])

    # Quick stats before analysis
    st.markdown('<div class="section-header">Call Metadata</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card accent">
            <div class="metric-label">Duration</div>
            <div class="metric-value accent">{total_duration:.0f}s</div>
            <div class="metric-sub">{total_duration/60:.1f} minutes</div>
        </div>
        <div class="metric-card safe">
            <div class="metric-label">Segments</div>
            <div class="metric-value safe">{len(segments_data)}</div>
            <div class="metric-sub">detected</div>
        </div>
        <div class="metric-card warn">
            <div class="metric-label">Words</div>
            <div class="metric-value warn">{total_words}</div>
            <div class="metric-sub">total in call</div>
        </div>
        <div class="metric-card accent">
            <div class="metric-label">STT Confidence</div>
            <div class="metric-value accent">{avg_conf:.0%}</div>
            <div class="metric-sub">segment average</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Transcript preview
    with st.expander("📄 View Transcription (Whisper)", expanded=False):
        transcript_lines = [f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments_data if s['text'].strip()]
        st.markdown(
            '<div style="font-family:var(--mono); font-size:0.78rem; line-height:1.9; color:var(--text); '
            'max-height:240px; overflow-y:auto;">' +
            "<br>".join(transcript_lines) +
            "</div>", unsafe_allow_html=True
        )

    st.markdown("---")

    # ─── ANALYSIS BUTTON ───
    if st.button("🔍 Run Analysis", type="primary", use_container_width=False):
        with st.spinner("⚙️ Analyzing..."):

            if model_id == "1":
                model, tfidf = data_grouping['rf_model'], data_grouping['tfidf']
            elif model_id == "2":
                model, tfidf = data_segments['rf_model'], data_segments['tfidf']
            elif model_id == "3":
                model = data_audio['rf_model']
            else:
                model, tfidf = data_hybrid['rf_model'], data_hybrid['tfidf']

            results_list = []
            prog = st.progress(0, text="Processing segments...")

            for i, seg in enumerate(segments_data):
                text = seg['text'].strip()
                if not text:
                    continue
                start, end = seg['start'], seg['end']
                dur    = end - start
                sr_rate = len(text.split()) / (dur + 0.001)
                asr    = np.exp(seg.get('avg_logprob', -1))
                clean_t = clean_and_lemmatize(text)

                if model_id == "1":
                    X_text  = tfidf.transform([clean_t])
                    X_num   = np.array([[sr_rate, dur, asr]])
                    X_final = hstack([X_text, X_num]).tocsr()
                elif model_id == "2":
                    X_text  = tfidf.transform([clean_t]).toarray()
                    X_num   = np.array([[sr_rate, dur, asr]])
                    X_final = np.hstack([X_text, X_num])
                elif model_id == "3":
                    X_final = extract_audio_features(temp_audio_path, start, end).reshape(1, -1)
                else:
                    X_text  = tfidf.transform([clean_t])
                    X_num   = np.array([[sr_rate, dur, asr]])
                    X_aud   = extract_audio_features(temp_audio_path, start, end).reshape(1, -1)
                    X_final = hstack([X_text, X_aud, X_num]).tocsr()

                prob = model.predict_proba(X_final)[0][1]
                results_list.append({
                    'start': start, 'end': end,
                    'Time': f"{start:.1f}s – {end:.1f}s",
                    'Phrase': text,
                    'Risk (Fraud)': prob,
                    'Level': risk_label(prob),
                    'Speech Rate': sr_rate,
                    'Segment Length': dur,
                    'ASR Confidence': asr,
                })
                prog.progress((i + 1) / len(segments_data),
                              text=f"Segment {i+1} / {len(segments_data)}")

        prog.empty()

        if not results_list:
            st.warning("Could not extract segments for analysis.")
            st.stop()

        df = pd.DataFrame(results_list)
        max_prob  = df['Risk (Fraud)'].max()
        mean_prob = df['Risk (Fraud)'].mean()
        n_high    = (df['Risk (Fraud)'] > threshold).sum()
        peak_idx  = df['Risk (Fraud)'].idxmax()
        peak_time = df.loc[peak_idx, 'Time']
        peak_phrase = df.loc[peak_idx, 'Phrase']

        # ── VERDICT BANNER ──
        st.markdown('<div class="section-header">Verdict</div>', unsafe_allow_html=True)
        if max_prob > threshold:
            st.markdown(f"""
            <div class="verdict-fraud">
                <div class="verdict-title">🚨 FRAUD DETECTED</div>
                <div class="verdict-sub">
                    Peak risk <b style="color:var(--danger);">{max_prob:.1%}</b> exceeds threshold {threshold:.0%} ·
                    Dangerous segments: <b>{n_high}</b> · Peak at [{peak_time}]
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-safe">
                <div class="verdict-title">✅ CALL IS SAFE</div>
                <div class="verdict-sub">
                    Maximum risk <b style="color:var(--safe);">{max_prob:.1%}</b> is below threshold {threshold:.0%} ·
                    Suspicious segments: <b>{n_high}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── KEY METRICS ROW ──
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card danger">
                <div class="metric-label">Peak Risk</div>
                <div class="metric-value danger">{max_prob:.1%}</div>
                <div class="metric-sub">segment maximum</div>
            </div>
            <div class="metric-card warn">
                <div class="metric-label">Average Risk</div>
                <div class="metric-value warn">{mean_prob:.1%}</div>
                <div class="metric-sub">call average</div>
            </div>
            <div class="metric-card {'danger' if n_high > 0 else 'safe'}">
                <div class="metric-label">Threats Found</div>
                <div class="metric-value {'danger' if n_high > 0 else 'safe'}">{n_high}</div>
                <div class="metric-sub">above {threshold:.0%} threshold</div>
            </div>
            <div class="metric-card accent">
                <div class="metric-label">Processed</div>
                <div class="metric-value accent">{len(df)}</div>
                <div class="metric-sub">segments</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── TABS ──
        tab_timeline, tab_phrases, tab_stats, tab_raw = st.tabs([
            "📈 Risk Timeline",
            "⚠️ High Risk Phrases",
            "📊 Analytics",
            "🗂 Raw Data"
        ])

        with tab_timeline:
            st.markdown('<div class="section-header">Risk Dynamics Over Time</div>', unsafe_allow_html=True)
            chart_df = df[['start', 'Risk (Fraud)']].copy()
            chart_df = chart_df.rename(columns={'start': 'Time (sec)', 'Risk (Fraud)': 'Fraud Risk'})
            st.line_chart(chart_df.set_index('Time (sec)'), use_container_width=True, height=220)

            # Threshold line note
            st.markdown(f"""
            <div style="font-size:0.72rem; color:var(--muted); font-family:var(--mono); margin-top:-0.5rem;">
            ⬆ Detection Threshold: {threshold:.0%} &nbsp;|&nbsp; Red zone is above this value
            </div>
            """, unsafe_allow_html=True)

            # Top risky moments
            top3 = df.nlargest(3, 'Risk (Fraud)')
            if not top3.empty:
                st.markdown('<div class="section-header">Peak Moments</div>', unsafe_allow_html=True)
                for rank, (_, row) in enumerate(top3.iterrows(), 1):
                    color = risk_color(row['Risk (Fraud)'])
                    st.markdown(f"""
                    <div class="phrase-row">
                        <div class="phrase-time">#{rank} &nbsp; {row['Time']}</div>
                        <div class="phrase-text">{row['Phrase']}</div>
                        <div class="risk-bar-wrap">
                            <div class="risk-bar-bg">
                                <div class="risk-bar-fill" style="width:{row['Risk (Fraud)']*100:.0f}%; background:{color};"></div>
                            </div>
                            <div class="risk-pct" style="color:{color};">{row['Risk (Fraud)']:.0%}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_phrases:
            high_risk_df = df[df['Risk (Fraud)'] > threshold].sort_values('Risk (Fraud)', ascending=False)
            if high_risk_df.empty:
                st.markdown("""
                <div style="text-align:center; padding:3rem; color:var(--safe);">
                    <div style="font-size:2rem;">✅</div>
                    <div style="font-family:var(--mono); margin-top:0.5rem;">No dangerous phrases detected</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="section-header">Detected {len(high_risk_df)} Dangerous Phrases</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                for _, row in high_risk_df.iterrows():
                    color = risk_color(row['Risk (Fraud)'])
                    st.markdown(f"""
                    <div class="phrase-row">
                        <div class="phrase-time">{row['Time']}</div>
                        <div class="phrase-text">{row['Phrase']}</div>
                        <div class="risk-bar-wrap">
                            <div class="risk-bar-bg">
                                <div class="risk-bar-fill" style="width:{row['Risk (Fraud)']*100:.0f}%; background:{color};"></div>
                            </div>
                            <div class="risk-pct" style="color:{color};">{row['Risk (Fraud)']:.0%} · {row['Level']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab_stats:
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
                bins = [0, 0.30, 0.50, 0.75, 1.01]
                labels = ["Normal (<30%)", "Low (30-50%)", "Medium (50-75%)", "High (>75%)"]
                counts = pd.cut(df['Risk (Fraud)'], bins=bins, labels=labels, right=False).value_counts()
                counts_df = counts.reset_index()
                counts_df.columns = ['Category', 'Count']
                st.bar_chart(counts_df.set_index('Category'), use_container_width=True, height=200)

            with col_right:
                st.markdown('<div class="section-header">Segment Statistics</div>', unsafe_allow_html=True)
                stats = {
                    "Min Risk":      f"{df['Risk (Fraud)'].min():.1%}",
                    "Max Risk":      f"{df['Risk (Fraud)'].max():.1%}",
                    "Median Risk":   f"{df['Risk (Fraud)'].median():.1%}",
                    "Avg Seg Length":f"{df['Segment Length'].mean():.1f}s",
                    "Avg Speed":     f"{df['Speech Rate'].mean():.1f} w/s",
                    "Avg STT Conf":  f"{df['ASR Confidence'].mean():.0%}",
                }
                for k, v in stats.items():
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; padding:0.45rem 0;
                                border-bottom:1px solid var(--border); font-size:0.82rem;">
                        <span style="color:var(--muted);">{k}</span>
                        <span style="font-family:var(--mono); color:var(--accent);">{v}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Speech Rate Over Time</div>', unsafe_allow_html=True)
            speed_chart = df[['start', 'Speech Rate']].rename(
                columns={'start': 'Time (sec)', 'Speech Rate': 'Words/sec'}
            )
            st.area_chart(speed_chart.set_index('Time (sec)'), use_container_width=True, height=160)

        with tab_raw:
            st.markdown('<div class="section-header">Full Analysis Report</div>', unsafe_allow_html=True)
            display_df = df[['Time', 'Phrase', 'Risk (Fraud)', 'Level', 'Speech Rate', 'Segment Length', 'ASR Confidence']].copy()

            def color_risk_cell(val):
                if isinstance(val, float):
                    if val > threshold: return 'background-color: rgba(255,71,87,0.2); color: #ff4757'
                    elif val > threshold * 0.7: return 'background-color: rgba(255,165,2,0.15); color: #ffa502'
                return ''

            styled = display_df.style \
                .map(color_risk_cell, subset=['Risk (Fraud)']) \
                .format({
                    'Risk (Fraud)': '{:.1%}',
                    'Speech Rate': '{:.2f}',
                    'Segment Length': '{:.1f}s',
                    'ASR Confidence': '{:.0%}'
                })
            st.dataframe(styled, use_container_width=True, height=420)

            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name=f"fraud_analysis_{uploaded_audio.name}.csv",
                mime="text/csv"
            )

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:var(--muted);">
        <div style="font-size:3.5rem; margin-bottom:1rem;">🎙️</div>
        <div style="font-size:1.1rem; font-weight:600; color:var(--text); margin-bottom:0.5rem;">
            Upload an audio recording to start analysis
        </div>
        <div style="font-size:0.82rem; max-width:380px; margin:0 auto; line-height:1.7;">
            The system will automatically transcribe speech, segment the call,
            and calculate fraud probability based on the selected AI model.
        </div>
    </div>
    """, unsafe_allow_html=True)

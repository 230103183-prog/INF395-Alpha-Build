# 🛡️ FraudGuard AI: Multimodal Audio Analysis
An intelligent real-time system designed to detect and prevent telephone fraud using AI-driven multimodal analysis.

# ⚠️ Problem Statement
Telephone fraud is a growing global threat where attackers use social engineering to manipulate victims into revealing sensitive data. Traditional detection methods often fail to catch subtle behavioral cues in real-time, making it necessary to have an automated defense layer that analyzes both linguistic content and acoustic patterns in calls.

# 📌 Project Overview
Course: INF 395 Final Project - Alpha Build

## FraudGuard AI is an intelligent system designed to detect telephone fraud in real-time. It uses a multimodal approach, analyzing both the content of the conversation (NLP) and acoustic metadata (speech rate, duration, and ASR confidence) to identify suspicious patterns.

# ✨ Key Features
Automated Transcription: Integrated OpenAI Whisper (tiny model) for instant speech-to-text.

Multimodal Prediction: Combines TF-IDF text analysis with acoustic features.

Machine Learning: Powered by a tuned Random Forest Classifier.

Explainable AI (XAI): Automatically highlights fraudulent trigger words (e.g., "код", "счёт", "продиктовать") in the transcription.

Interactive Dashboard: A clean Streamlit interface for easy file uploads and result visualization.

# 🛠 Technology Stack
Language: Python

Speech-to-Text: OpenAI Whisper

ML Frameworks: Scikit-learn, XGBoost

Dashboard: Streamlit

Audio Processing: FFmpeg, OpenSMILE

# 📂 Repository Structure
src/ — Contains app.py and core logic.

assets/ — Directory with test samples (formerly sample_audio/).

models/ — Pre-trained model weights (.pkl files).

docs/ — Project documentation.

tests/ — Automated tests.

requirements.txt — List of Python dependencies.

# 🚀 Installation & Setup
## 1. Prerequisites

The system requires FFmpeg to process audio files.

macOS: brew install ffmpeg

Windows: winget install ffmpeg

Linux: sudo apt install ffmpeg

## 2. Install Dependencies

Navigate to the project directory and run:

Bash
pip install -r requirements.txt
## 3. Run the Application

To launch the FraudGuard AI dashboard, execute the following command:

Bash
streamlit run src/app.py
# 🧪 How to Test
Launch the app and wait for the browser to open http://localhost:8501.

In the Audio Input section, upload a file from the assets/ folder.

Enter the audio duration (in seconds) as seen in the player.

Click Transcribe & Analyze Call.

View the automated transcription and the fraud probability score provided by the AI.

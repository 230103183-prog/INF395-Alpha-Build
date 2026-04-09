# 🛡️ FraudGuard AI: Multimodal Audio Analysis

## 📌 Project Overview
**Course:** INF 395 Final Project - Alpha Build

FraudGuard AI is an intelligent system designed to detect telephone fraud in real-time. It uses a multimodal approach, analyzing both the **content of the conversation** (NLP) and **acoustic metadata** (speech rate, duration, and ASR confidence) to identify suspicious patterns.

## ✨ Key Features
* **Automated Transcription:** Integrated OpenAI Whisper (`tiny` model) for instant speech-to-text.
* **Multimodal Prediction:** Combines TF-IDF text analysis with acoustic features.
* **Machine Learning:** Powered by a tuned Random Forest Classifier.
* **Explainable AI (XAI):** Automatically highlights fraudulent trigger words (e.g., "код", "счёт", "продиктовать") in the transcription.
* **Interactive Dashboard:** A clean Streamlit interface for easy file uploads and result visualization.

## 📂 Repository Structure
* `app.py` — The core Streamlit application.
* `requirements.txt` — List of Python dependencies.
* `models/` — Folder containing pre-trained model weights (`.pkl` files).
* `sample_audio/` — Directory with test samples (Fraud vs. Normal calls).
* `README.md` — Project documentation.

## 🚀 Installation & Setup

### 1. Prerequisites
The system requires **FFmpeg** to process audio files.
* **macOS:** `brew install ffmpeg`
* **Windows:** `winget install ffmpeg` or download manually.
* **Linux:** `sudo apt install ffmpeg`

### 2. Install Dependencies
Navigate to the project directory and run:
`pip install -r requirements.txt`

### 3. Run the Application
To launch the FraudGuard AI dashboard, execute the following command:
`streamlit run app.py`

## 🧪 How to Test
1. Launch the app and wait for the browser to open `http://localhost:8501`.
2. In the **Audio Input** section, upload a file from the `sample_audio/` folder.
3. Enter the audio duration (in seconds) as seen in the player.
4. Click **Transcribe & Analyze Call**.
5. View the automated transcription and the fraud probability score provided by the AI.


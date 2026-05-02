# FraudShield AI: Multimodal Audio Analysis
**Student IDs:** 230103183, 230103064

## Problem Statement
Telephone fraud is a global security threat causing billions in losses. Attackers use sophisticated social engineering to manipulate victims. Current solutions often lack real-time multimodal analysis—combining linguistic cues (what is said) with acoustic patterns (how it is said). FraudShield AI fills this gap by providing an automated, AI-driven defense layer to detect suspicious activities during live calls.

## Technology Stack
* **Language:** Python 3.9+
* **Interface:** Streamlit
* **Speech-to-Text:** OpenAI Whisper (Tiny)
* **Machine Learning:** Scikit-learn (Random Forest), Joblib
* **Audio Processing:** Librosa, FFmpeg
* **NLP:** Pymorphy3 (Morphological analysis)

## Repository Structure
* `src/` — Core application logic and `app.py`.
* `src/models/` — Pre-trained ML models (.pkl).
* `assets/` — Sample audio files for testing.
* `docs/` — Project documentation and manuals.
* `tests/` — Unit tests for fraud detection logic.
* `requirements.txt` — Python dependencies.
* `packages.txt` — System dependencies (FFmpeg).

## Installation & Usage
1. **Clone the repository:** ```bash
git clone https://github.com/your-repo/fraudshield-ai.git
cd fraudshield-ai```

2. **Virtual Environment:** ```bash
python -m venv venv

#Windows:
venv\Scripts\activate

#macOS/Linux:
source venv/bin/activate```

4. **System Requirements (FFmpeg):**
   
macOS: `brew install ffmpeg`

Ubuntu: `sudo apt install ffmpeg`

Windows: Download binaries from ffmpeg.org and add the bin folder to your system PATH.

4. **Install Dependencies:** ```bash
pip install --upgrade pip
pip install -r requirements.txt```

5. **Run Application:** ```bash
streamlit run src/app.py```

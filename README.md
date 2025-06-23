# 🖐️ Viberr – Feel the Words

**Viberr** is a real-time speech-to-vibration translation system for the deaf and hard-of-hearing. It transforms spoken language into finger-based vibration patterns using a unique 5-bit binary encoding system — enabling users to feel and interpret speech through touch.

> A new universal tactile language – powered by binary, driven by empathy.

---

## 🌟 Features

- 🎤 **Offline Voice Input** via [Vosk](https://github.com/alphacep/vosk-api)
- 🌍 **Multilingual Translation Support** (user toggleable)
- 🔡 **Binary Encoding** of letters into 5-bit vibration signals
- ✋ **Finger-Based Vibration Simulation** (console or UI)
- 🧠 **Interactive Trainer Web UI** to learn and practice patterns
- 🕒 **Manual Speed Control** for vibration pace

---

## 📁 Folder Structure

```
VIBRA_TRAINER_WEB/
├── app.py                     # Flask + Vosk + simulation logic
├── README.md                  # This file
│
├── data/
│   └── sample_words.json      # Words for training mode
│
├── mappings/
│   └── binary_map.json        # Binary letter mapping (A–Z)
│
├── models/
│
├── static/
│   ├── script.js              # JS logic (UI, speed, animation)
│   └── style.css              # CSS for finger animations
│
├── templates/
│   └── index.html             # Flask-rendered Trainer interface
│
├── viberr.git/                # Local Git repository metadata
└── __pycache__/               # Python build cache
```

---
```

---

## 🎮 How It Works

### 🎧 Stage 1: Voice-to-Vibration (Backend)
1. Live speech captured using microphone
2. Transcribed to English text using `vosk`
3. Text → 5-bit binary code → finger mapping
4. Console simulates finger vibrations (e.g., “Thumb, Pinky”)
5. User can adjust output speed

### 🌐 Stage 1.5: Trainer Web UI (Flask)
1. User clicks “Next” to get a letter
2. Binary pattern shown as finger animation
3. User can guess the letter and receive feedback
4. Speed slider adjusts animation delay
5. Also supports simple words (optional)

---

## 🔢 Binary Mapping Logic

Each alphabet letter is encoded into a unique 5-bit pattern:

- Bit 1 → Thumb  
- Bit 2 → Index  
- Bit 3 → Middle  
- Bit 4 → Ring  
- Bit 5 → Pinky

Example:
```text
Letter: H
Binary: 10101
Fingers Vibrated: Thumb, Middle, Pinky
```

- `11111` is reserved as a **break buzz** to indicate space or word separation.

Mapping is optimized using **Hamming distance** and **letter frequency**.

---

## ⚙️ Requirements

- Python 3.8+
- Flask
- vosk
- PyAudio (or sounddevice for some systems)

Install with:
```bash
pip install flask vosk pyaudio
```

---

## 🚀 Setup Instructions

1. Clone the repo:
```bash
git clone https://github.com/yourusername/viberr.git
cd viberr
```

2. Download a Vosk model (e.g., small English):
   https://alphacephei.com/vosk/models

3. Run the app:
```bash
python app.py
```

4. Open the web interface at:
```
http://localhost:5000
```

---

## 📌 Roadmap

| Stage | Description |
|-------|-------------|
| ✅ Stage 1 | Real-time console output |
| ✅ Stage 1.5 | Web UI for training |
| 🔜 Stage 2 | Hardware glove integration (vibration motors) |
| 🔜 Stage 3 | ML-powered adaptive speed and feedback |
| 🔜 Stage 4 | Mobile app with on-device inference (TinyML) |

---

## 📚 Use Cases

- Live communication aid for deaf individuals
- Tactile speech learning for assistive education
- Emergency response and safety systems
- Prototype base for wearable haptics

---

## 🧠 Future Enhancements

- Glove-based hardware support via Raspberry Pi / Arduino
- Predictive next-letter suggestions
- User performance tracking and feedback loop
- Support for sentence context and grammar cues

---

## 🪪 License

This project is licensed under the MIT License.  
Free to use, modify, and build upon.

---

## 🙌 Contributions

Have an idea or want to improve Viberr?  
Feel free to fork the repo, raise issues, or submit PRs.

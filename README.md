# 🐾 WildGuard AI – Wildlife Detection & Intelligent Alert System

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-dc2626?style=for-the-badge&logo=qdrant)](https://qdrant.tech)
[![Gemini 3.8 Flash](https://img.shields.io/badge/Google_Gemini-3.8_Flash-4285F4?style=for-the-badge&logo=google)](https://ai.google.dev)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)

An edge-optimized, real-time wildlife monitoring and early warning system designed to mitigate Human-Wildlife Conflict (HWC). **WildGuard AI** tracks apex predators and megaherbivores using YOLOv8, detects perimeter breaches via user-drawn virtual polygon geofences, logs forensic telemetry into **Qdrant Vector Database**, and assists forest rangers with an interactive **Gemini 3.8 Flash** intelligence assistant.

---

## 🌟 Key Capabilities

- **🎯 Real-Time YOLOv8 Detection & Tracking:** Multi-species identification (Tigers, Elephants, Deer, etc.) with ground-contact anchor tracking and persistent track IDs.
- **📐 Interactive Virtual Polygon Geofencing:** Draw custom danger zones directly onto the camera canvas. Animals crossing into the danger zone trigger instant audible alarms and visual siren alerts.
- **⚡ 100% Qdrant Vector Database:** Fully vector-native storage storing 512-dimensional dynamic quantized embeddings alongside rich situational incident payloads (GPS, timestamps, danger zone status, threat scores, snapshots). Zero SQL required.
- **💬 AI Ranger Operations Assistant:** Conversational RAG assistant grounded on Qdrant telemetry and powered by Gemini 3.8 Flash. Features multi-turn memory, tactical dispatch directives, and village broadcast advisories.
- **🛡️ Secure Client-Side Markdown Rendering:** Built-in Markdown parser with DOMPurify XSS sanitization, rendering responsive threat badges, incident details, timestamps, and zero-overflow containers.
- **🖥️ Responsive Command Center Dashboard:** Low-latency MJPEG video streaming, camera sensor telemetry, risk distribution charts, and forensic evidence galleries.

---

## 🏗️ System Architecture

```text
       [ Camera Trap / Video Stream / Webcam ]
                         │
                         ▼
             [ YOLOv8 Tracking Engine ]
                         │
                         ├─► Virtual Polygon Geofence Evaluator
                         │   └─► Instant Audible Siren & Visual Badge
                         │
                         ▼
        [ Multimodal INT8 Dynamic Quantizer ]
                         │
                         ▼
      [ Embedded Qdrant Local Vector Database ]
       (Vectors: 512-dim | Payloads: Full Forensic Logs)
                         │
                         ├────────────────────────┐
                         ▼                        ▼
               [ FastAPI Command Center ]   [ Gemini 3.8 Flash ]
                 • /video_feed (MJPEG)        • AI Ranger Assistant
                 • /api/latest_alert          • Multi-turn Dialogue
                 • /api/incidents             • Tactical Directives
```

---

## 📁 Repository Structure

```text
CV_wildlife/
├── data/
│   ├── cameras.json               # Camera sensors GPS & sector metadata
│   ├── geofence.json              # Active polygon geofence coordinates
│   └── sample_wildlife.mp4        # Default surveillance feed
├── models/
│   ├── wildlife_multi_yolov8.pt   # Multi-class wildlife YOLOv8 model
│   └── yolov8n.pt                 # YOLOv8 nano baseline
├── src/
│   ├── chatbot.py                 # Conversational AI Ranger & RAG engine
│   ├── incident_manager.py        # Frame aggregator & geofence evaluator
│   ├── live_stream.py             # VideoCapture stream & MJPEG generator
│   ├── llm_advisor.py             # Incident evaluator & threat scorer
│   ├── quantized_embedding.py     # CPU-optimized 8-bit embedding engine
│   └── vector_store.py            # Qdrant client & payload manager
├── static/
│   ├── js/
│   │   ├── marked.min.js          # Client-side Markdown parser
│   │   └── purify.min.js          # DOMPurify XSS sanitizer
│   ├── siren.wav                  # Audible warning alarm
│   └── *.jpg                      # Wildlife species assets
├── templates/
│   └── index.html                 # Command Center dashboard UI
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules (secrets, DB, cache)
├── requirements.txt               # Project dependencies
└── server.py                      # FastAPI server entry point
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Git

### 2. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/WildGuard-AI.git
cd WildGuard-AI
```

### 3. Create Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Keys
Copy the example environment file and add your Google Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: If no API key is provided, the AI Ranger automatically falls back to the deterministic local RAG engine).*

---

## 🏃 Running the Application

### Start Command Center Server
```bash
python server.py
```
The server will start on `http://127.0.0.1:8080`.

Open your browser and navigate to:
```
http://127.0.0.1:8080
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Main Command Center Dashboard (HTML) |
| `GET` | `/video_feed` | Continuous MJPEG stream with YOLO tracking & geofence overlay |
| `POST` | `/api/upload_video` | Upload custom MP4/AVI surveillance footage |
| `POST` | `/api/switch_source` | Switch feed between sample video and local webcam (`0`) |
| `GET` | `/api/first_frame` | Fetch first video frame for polygon drawing canvas |
| `GET` | `/api/get_geofence` | Retrieve active geofence polygon vertices |
| `POST` | `/api/set_geofence` | Update custom polygon geofence vertices |
| `POST` | `/api/reset_geofence`| Reset geofence to default perimeter |
| `GET` | `/api/latest_alert` | Real-time incident alert status & dashboard counters |
| `GET` | `/api/incidents` | Fetch all historical incident payloads from Qdrant |
| `POST` | `/api/chat` | Query the AI Ranger Assistant with conversation context |
| `POST` | `/api/chat/reset` | Clear session dialogue history |

---

## 🔒 Security & Privacy Notice
- Never commit your `.env` file or API keys.
- Local video uploads and temporary evidence snapshots are excluded from Git by `.gitignore`.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

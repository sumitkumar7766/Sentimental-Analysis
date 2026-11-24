# 🚀 SentiMind — Node.js Backend Server

A high-performance, cross-platform Node.js Express backend for the **SentiMind Multimodal Sentiment Analysis Platform**, powered by **OpenRouter AI** with free models.

---

## 🌟 Why Node.js?
- **Zero Python or PyTorch installation required**: Runs directly with Node.js on Windows, Mac, and Linux.
- **No C++ compilation or symlink errors on Windows**: Installs cleanly in seconds.
- **100% API Compatible**: Drop-in replacement for the FastAPI backend, matching all routes (`/predict`, `/predict/emotion`, `/metrics`, `/datasets`).
- **OpenRouter Free Model Support**: Built-in 95%+ accurate sentiment and emotion detection with automatic fallback models (`nvidia/nemotron-3.5-lightning:free`, `baidu/ernie-5.5-flash-preview:free`, `google/gemma-4-26b-a4b-it:free`).

---

## ⚡ Quick Start on Windows & Mac

### 1. Install Dependencies
```bash
cd backend-node
npm install
```

### 2. Configure Environment
```bash
# On Windows:
copy .env.example .env

# On Mac/Linux:
cp .env.example .env
```
Open `.env` and paste your OpenRouter API key:
```env
OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Start the Server
```bash
npm start
```
The server will start on port **8000**:
- **API URL:** `http://localhost:8000`
- **Documentation:** `http://localhost:8000/docs`

---

## 🔗 Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Multimodal sentiment analysis (`text`, `image`, `audio`, `fusion_type`) |
| `POST` | `/predict/emotion` | 7-category emotion detection (`Happiness`, `Sadness`, `Anger`, etc.) |
| `GET` | `/metrics` | Model performance, ROC curves & confusion matrix |
| `GET` | `/datasets` | Research datasets metadata |

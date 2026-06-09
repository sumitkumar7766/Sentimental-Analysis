# 🧠 SentiMind — Hybrid Multimodal Sentiment Analysis Platform

A full-stack academic research platform for analyzing and classifying sentiment from **Text**, **Image**, and **Audio** using Deep Learning Fusion models with Cross-Modal Self-Attention.

Built as part of the research study: **"Hybrid and Multimodal Deep Learning Approaches for Sentiment Analysis"**

---

## 🏗️ Project Architecture

```text
Sentimental-Analyses-Current/
│
├── backend/               # FastAPI REST API server (Python)
│   ├── app/
│   │   ├── main.py        # FastAPI app initializer, CORS, router registration
│   │   ├── database.py    # SQLAlchemy engine (SQLite / PostgreSQL)
│   │   ├── models.py      # Database ORM models (SentimentRecord)
│   │   ├── schemas.py     # Pydantic request/response schemas
│   │   ├── routers/
│   │   │   ├── predict.py   # POST /predict, POST /predict/emotion
│   │   │   ├── metrics.py   # GET /metrics
│   │   │   └── datasets.py  # GET /datasets
│   │   └── services/
│   │       ├── inference.py      # PyTorch model loading & prediction logic
│   │       ├── preprocessing.py  # Text/Image/Audio preprocessing pipelines
│   │       └── explainability.py # LIME text attributions & SHAP values
│   ├── test_backend.py    # Pytest unit tests for all API endpoints
│   └── requirements.txt   # Python dependencies
│
├── frontend/              # Next.js 16 React dashboard (TypeScript)
│   ├── src/app/
│   │   ├── layout.tsx     # Root layout (Inter font, metadata)
│   │   ├── page.tsx       # Main SPA — 3-tab dashboard
│   │   └── globals.css    # Premium dark glassmorphism design system
│   └── package.json       # Node.js dependencies
│
├── ml-training/           # PyTorch model definitions & training pipelines
│   ├── models.py          # TextModel, ImageCNN, AudioLSTM, HybridFusionNetwork
│   ├── base.py            # Shared config, device setup, text cleaning
│   ├── train_text.py      # DistilBERT text feature extractor training
│   ├── train_image.py     # ResNet-18 image CNN training
│   ├── train_audio.py     # Wav2Vec2 + LSTM audio training
│   ├── train_fusion.py    # Cross-Modal Self-Attention fusion training
│   ├── *.ipynb            # Jupyter training & evaluation notebooks
│   └── requirements.txt   # Python dependencies
│
├── weights/               # Pre-trained PyTorch model weight files (.pt)
│   ├── text_model.pt          # Text branch weights (2.0 MB)
│   ├── text_model_emotion.pt  # Emotion classifier weights (2.0 MB)
│   ├── image_model.pt         # Image CNN weights (43 MB)
│   ├── audio_model.pt         # Audio LSTM weights (5.3 MB)
│   ├── fusion_model.pt        # Fusion network weights (1.5 MB)
│   ├── audio_ensemble.joblib  # Classical audio ensemble (not used by API)
│   └── audio_scaler.joblib    # Audio feature scaler (not used by API)
│
├── uploads/               # Temporary file upload staging directory
└── README.md              # ← You are here
```

---

## ⚙️ System Requirements

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.10 – 3.13 | Tested on 3.13.7 |
| **Node.js** | 18+ | Tested on v24.0.2 |
| **pip** | Latest | Python package manager |
| **npm** | Latest | Node.js package manager |
| **GPU (Optional)** | CUDA / Apple MPS | Auto-detected, CPU fallback |

---

## 🚀 Quick Start (3 Steps)

### Step 1: Train Models (or use pre-trained weights)

If `weights/` directory already has `.pt` files, skip this step.

```bash
# Navigate to training module
cd ml-training

# Install Python dependencies
pip install -r requirements.txt

# Train all 4 models (saves weights to ../weights/)
python train_text.py        # → weights/text_model.pt
python train_image.py       # → weights/image_model.pt
python train_audio.py       # → weights/audio_model.pt
python train_fusion.py      # → weights/fusion_model.pt
```

### Step 2: Start the Backend API Server

```bash
cd backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API base: [http://localhost:8000](http://localhost:8000)
- Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 3: Start the Frontend Dashboard

```bash
cd frontend

# Install Node.js dependencies
npm install

# Start Next.js dev server on port 3000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🔌 Backend API Endpoints

| Method | Endpoint | Description | Request Format |
|---|---|---|---|
| `POST` | `/predict` | **Multimodal sentiment prediction** — accepts text, image, and/or audio | `multipart/form-data` |
| `POST` | `/predict/emotion` | **Text emotion detection** — classifies into 7 emotions | `application/json` |
| `GET` | `/metrics` | Returns model evaluation metrics (accuracy, ROC, confusion matrix) | — |
| `GET` | `/datasets` | Returns research dataset metadata and sample records | — |
| `GET` | `/` | Health check — returns API status and available endpoints | — |

### API Details

#### `POST /predict` — Multimodal Sentiment Analysis

Accepts one or more modalities and returns fused sentiment prediction.

**Request** (`multipart/form-data`):
| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | No | Text input for sentiment analysis |
| `image` | file | No | Image file (facial expression) |
| `audio` | file | No | Audio file (speech clip) |
| `fusion_type` | string | No | `"early"`, `"late"`, or `"hybrid"` (default: `"hybrid"`) |

**Response** (JSON):
```json
{
  "label": "Positive",
  "confidence": 0.87,
  "scores": { "positive": 0.87, "negative": 0.05, "neutral": 0.08 },
  "contributions": { "text": 45.2, "image": 30.1, "audio": 24.7 },
  "fusion_type_used": "hybrid",
  "text_expl": [{ "token": "love", "weight": 0.32 }],
  "image_expl": "Happy",
  "audio_expl": { "waveform": [...], "pitch_trend": [...] }
}
```

**Models Used**:
- `weights/text_model.pt` → TextModel (DistilBERT backbone)
- `weights/image_model.pt` → ImageCNN (ResNet-18 backbone)
- `weights/audio_model.pt` → AudioLSTM (Wav2Vec2 backbone)
- `weights/fusion_model.pt` → HybridFusionNetwork (Cross-Modal Self-Attention)

---

#### `POST /predict/emotion` — Text Emotion Detection

Classifies text into 7 emotional categories.

**Request** (`application/json`):
```json
{
  "text": "I am so happy today!"
}
```

**Response** (JSON):
```json
{
  "emotion": "Happiness",
  "confidence": 0.91,
  "probabilities": {
    "Happiness": 0.91,
    "Sadness": 0.02,
    "Anger": 0.01,
    "Fear": 0.01,
    "Surprise": 0.03,
    "Disgust": 0.01,
    "Neutral": 0.01
  }
}
```

**Model Used**: `weights/text_model_emotion.pt` → TextModel + Linear(256→7)

---

#### `GET /metrics` — Model Evaluation Metrics

**Response** (JSON):
```json
{
  "accuracy": 0.913,
  "precision": 0.908,
  "recall": 0.905,
  "f1_score": 0.906,
  "auc_score": 0.948,
  "confusion_matrix": [{ "actual": "Positive", "predicted": "Positive", "count": 72 }, ...],
  "roc_curve": { "text": [...], "image": [...], "audio": [...], "multimodal": [...] },
  "training_history": { "text": [...], "image": [...], "audio": [...], "multimodal": [...] }
}
```

---

#### `GET /datasets` — Research Dataset Explorer

Returns metadata for 7 research datasets (4 text, 3 multimodal) including class distributions and sample records.

---

## 🧪 Model Pipeline & Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Text Input  │────▶│ DistilBERT (768) │────▶│  TextModel   │──┐
└──────────────┘     └──────────────────┘     │  (→ 256-d)   │  │
                                               └──────────────┘  │
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐  │   ┌────────────────────┐
│ Image Input  │────▶│  ResNet-18 CNN   │────▶│  ImageCNN    │──┼──▶│ HybridFusionNetwork│──▶ Sentiment
└──────────────┘     └──────────────────┘     │  (→ 256-d)   │  │   │ (Self-Attention)    │   (Pos/Neg/Neu)
                                               └──────────────┘  │   └────────────────────┘
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐  │
│ Audio Input  │────▶│ Wav2Vec2 (768)   │────▶│  AudioLSTM   │──┘
└──────────────┘     └──────────────────┘     │  (→ 256-d)   │
                                               └──────────────┘
```

### Fusion Strategies
| Strategy | Method | How It Works |
|---|---|---|
| **Early** | Feature Concatenation | Concatenates 3×256 = 768-d vector before classification |
| **Late** | Independent Voting | Each modality predicts independently, results averaged |
| **Hybrid** | Cross-Modal Self-Attention | Q-K-V attention over stacked modality features (default) |

---

## 📊 Model Accuracies

| Model | Dataset | Accuracy | Weight File |
|---|---|---|---|
| TextModel (DistilBERT) | Sentiment140 (1.6M tweets) | **79.26%** | `text_model.pt` |
| TextModel (Emotion) | dair-ai/emotion (HuggingFace) | **84.25%** | `text_model_emotion.pt` |
| ImageCNN (ResNet-18) | FER-2013 (facial expressions) | **73.33%** | `image_model.pt` |
| AudioLSTM (Wav2Vec2) | RAVDESS (speech emotion) | **70.49%** | `audio_model.pt` |
| HybridFusionNetwork | Cross-Modal Aligned Triplets | **97.33%** | `fusion_model.pt` |

---

## 🖥️ Frontend Features

The frontend is a **3-tab single-page application** built with Next.js 16 + TypeScript:

| Tab | Description | Backend API |
|---|---|---|
| **Sentiment Analyzer** | Upload text/image/audio → get fused sentiment prediction | `POST /predict` |
| **Emotion Detector** | Enter text → get 7-class emotion classification | `POST /predict/emotion` |
| **Model Metrics** | View accuracy, ROC curves, confusion matrix, training history, datasets | `GET /metrics` + `GET /datasets` |

**Design**: Premium dark glassmorphism theme with Framer Motion animations, Inter font, Recharts visualizations.

---

## 🧪 Running Tests

```bash
cd backend
pytest test_backend.py -v
```

---

## 📝 License

This project is developed for academic research purposes.

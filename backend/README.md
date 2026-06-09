# 🔧 Backend — FastAPI Sentiment Analysis API

The Python backend hosts all REST API endpoints for multimodal sentiment prediction, emotion detection, model evaluation metrics, and dataset exploration. It loads pre-trained PyTorch model weights from the `weights/` directory and performs real-time inference.

---

## 📂 Directory Structure

```text
backend/
├── app/
│   ├── __init__.py            # Package initializer
│   ├── main.py                # FastAPI app creation, CORS setup, router registration
│   ├── database.py            # SQLAlchemy engine (SQLite default, PostgreSQL optional)
│   ├── models.py              # ORM model: SentimentRecord (audit logging)
│   ├── schemas.py             # Pydantic schemas for all API request/response contracts
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── predict.py         # POST /predict, POST /predict/emotion
│   │   ├── metrics.py         # GET /metrics
│   │   └── datasets.py        # GET /datasets
│   └── services/
│       ├── __init__.py
│       ├── inference.py       # Core: loads PyTorch models, runs predictions
│       ├── preprocessing.py   # Text tokenization, image transforms, audio MFCC extraction
│       └── explainability.py  # LIME text attributions, SHAP modality weights
├── test_backend.py            # Pytest unit tests (5 tests covering all endpoints)
├── requirements.txt           # Python dependencies
├── sentiment_platform.db      # SQLite database (auto-created on first run)
└── uploads/                   # Temporary storage for uploaded files
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 – 3.13
- Pre-trained weight files in `../weights/` directory (see ml-training README)

### Step-by-step

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Start the FastAPI development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Verify Installation

- **Health Check**: [http://localhost:8000](http://localhost:8000) — returns JSON with status and available APIs
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs) — interactive API documentation
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc) — alternative API documentation

---

## 🔌 API Endpoints

### `GET /` — Health Check

Returns server status and lists all available API routes.

**Response:**
```json
{
  "status": "online",
  "platform": "Hybrid Multimodal Sentiment Analysis Platform",
  "apis": ["/predict", "/predict/emotion", "/metrics", "/datasets"]
}
```

---

### `POST /predict` — Multimodal Sentiment Prediction

Accepts text, image, and/or audio inputs. Runs feature extraction through pre-trained backbone models, then predicts sentiment using the HybridFusionNetwork.

**Request Format**: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | `string` | No | Raw text input for sentiment analysis |
| `image` | `file` (binary) | No | Image file — facial expression (JPG, PNG) |
| `audio` | `file` (binary) | No | Audio file — speech clip (WAV, MP3) |
| `fusion_type` | `string` | No | Fusion strategy: `"early"`, `"late"`, or `"hybrid"` (default: `"hybrid"`) |

> **Note:** At least one of `text`, `image`, or `audio` must be provided.

**Response:**
```json
{
  "label": "Positive",
  "confidence": 0.87,
  "scores": {
    "positive": 0.87,
    "negative": 0.05,
    "neutral": 0.08
  },
  "contributions": {
    "text": 45.2,
    "image": 30.1,
    "audio": 24.7
  },
  "fusion_type_used": "hybrid",
  "text_expl": [
    { "token": "love", "weight": 0.32 },
    { "token": "great", "weight": 0.28 }
  ],
  "image_expl": "Happy",
  "audio_expl": {
    "waveform": [0.1, -0.3, 0.5, ...],
    "pitch_trend": [140.2, 145.1, 138.9, ...]
  }
}
```

**Models Loaded (from `../weights/`):**
| Weight File | Model Class | Backbone | Output |
|---|---|---|---|
| `text_model.pt` | `TextModel` | DistilBERT (768-d) | 256-d text feature |
| `image_model.pt` | `ImageCNN` | ResNet-18 | 256-d image feature |
| `audio_model.pt` | `AudioLSTM` | Wav2Vec2 (768-d) | 256-d audio feature |
| `fusion_model.pt` | `HybridFusionNetwork` | Cross-Modal Self-Attention | 3-class sentiment |

**cURL Example:**
```bash
# Text only
curl -X POST http://localhost:8000/predict \
  -F "text=I love this product, it works perfectly!"

# Text + Image
curl -X POST http://localhost:8000/predict \
  -F "text=What a beautiful day" \
  -F "image=@/path/to/face.jpg"

# All three modalities
curl -X POST http://localhost:8000/predict \
  -F "text=Great experience overall" \
  -F "image=@/path/to/face.jpg" \
  -F "audio=@/path/to/speech.wav" \
  -F "fusion_type=hybrid"
```

---

### `POST /predict/emotion` — Text Emotion Detection

Classifies input text into one of 7 emotional categories using the pre-trained emotion model.

**Request Format**: `application/json`

```json
{
  "text": "I am so happy today!"
}
```

**Response:**
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

**Model Used**: `text_model_emotion.pt` → `TextModel` (DistilBERT backbone) + `Linear(256→7)` classifier head

**Emotion Classes**: Happiness, Sadness, Anger, Fear, Surprise, Disgust, Neutral

**cURL Example:**
```bash
curl -X POST http://localhost:8000/predict/emotion \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so angry right now!"}'
```

---

### `GET /metrics` — Model Evaluation Metrics

Returns pre-computed evaluation metrics for all model branches and the multimodal fusion ensemble.

**Response:**
```json
{
  "accuracy": 0.913,
  "precision": 0.908,
  "recall": 0.905,
  "f1_score": 0.906,
  "auc_score": 0.948,
  "confusion_matrix": [
    { "actual": "Positive", "predicted": "Positive", "count": 72 },
    { "actual": "Positive", "predicted": "Neutral", "count": 6 },
    ...
  ],
  "roc_curve": {
    "text": [{ "fpr": 0.0, "tpr": 0.0, "threshold": 1.0 }, ...],
    "image": [...],
    "audio": [...],
    "multimodal": [...]
  },
  "training_history": {
    "text": [{ "epoch": 1, "train_loss": 0.85, "val_loss": 0.68, "train_acc": 0.61, "val_acc": 0.70 }, ...],
    "image": [...],
    "audio": [...],
    "multimodal": [...]
  }
}
```

**cURL Example:**
```bash
curl http://localhost:8000/metrics
```

---

### `GET /datasets` — Research Dataset Explorer

Returns metadata, class distributions, and sample records for 7 research datasets.

**Datasets Included:**

| Dataset | Category | Size | Description |
|---|---|---|---|
| IMDB Reviews | Text | 50,000 | Binary movie review classification |
| Amazon Reviews | Text | 142,000 | Multi-domain product feedback |
| Sentiment140 | Text | 1,600,000 | Twitter sentiment with emoticon labels |
| US Airlines Twitter | Text | 14,485 | Airline complaint sentiment |
| CMU-MOSI | Multimodal | 2,199 | Video clips with aligned text + audio |
| MuSe | Multimodal | 4,500 | In-the-wild conversational sentiment |
| GeoCoV19 | Multimodal | 20,000 | COVID-19 social media posts |

**cURL Example:**
```bash
curl http://localhost:8000/datasets
```

---

## 🧬 Model Loading Flow

When the backend starts, `InferenceService.__init__()` in `app/services/inference.py` loads:

1. **Hugging Face Backbones** (downloaded from internet on first run):
   - `distilbert-base-uncased` — text tokenizer + BERT embeddings
   - `facebook/wav2vec2-base-960h` — audio feature extraction

2. **PyTorch Weight Files** (from `../weights/` directory):
   - `text_model.pt` → `TextModel(pretrained_dim=768, feature_dim=256)`
   - `image_model.pt` → `ImageCNN(feature_dim=256)`
   - `audio_model.pt` → `AudioLSTM(input_dim=768, hidden_dim=128, num_layers=2, feature_dim=256)`
   - `fusion_model.pt` → `HybridFusionNetwork(feature_dim=256, num_classes=3)`
   - `text_model_emotion.pt` → `TextModel(pretrained_dim=768, feature_dim=256)` + `Linear(256, 7)`

3. **Model Architecture Definitions** (imported from `../ml-training/models.py`):
   - `TextModel`, `ImageCNN`, `AudioLSTM`, `HybridFusionNetwork`

**Device Selection**: Automatically uses CUDA → MPS (Apple Silicon) → CPU

---

## 🧪 Running Tests

```bash
cd backend
pytest test_backend.py -v
```

**Tests:**
- `test_root` — Health check endpoint
- `test_predict_text` — Text-only sentiment prediction
- `test_predict_emotion` — Emotion classification
- `test_metrics` — Metrics endpoint response validation
- `test_datasets` — Datasets endpoint response validation

---

## 📦 Dependencies

Key packages in `requirements.txt`:

| Package | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI web server |
| `pydantic` | Request/response validation |
| `sqlalchemy` | Database ORM |
| `torch`, `torchvision`, `torchaudio` | PyTorch model inference |
| `transformers` | Hugging Face DistilBERT, Wav2Vec2 |
| `librosa` | Audio signal processing (MFCC) |
| `pillow` | Image loading & transforms |
| `numpy`, `scipy` | Numerical computing |
| `python-multipart` | File upload handling |
| `pytest` | Unit testing |

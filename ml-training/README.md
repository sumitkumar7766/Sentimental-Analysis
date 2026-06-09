# 🧠 ML Training — Model Architectures & Training Pipelines

This module contains the PyTorch model definitions, training scripts, and Jupyter notebooks for training all individual sentiment branches (Text, Image, Audio) and the attention-based Hybrid Multimodal Fusion network.

---

## 📂 Directory Structure

```text
ml-training/
├── models.py                          # PyTorch model class definitions
│                                      #   → TextModel, ImageCNN, AudioLSTM, HybridFusionNetwork
├── base.py                            # Shared config: device setup, text cleaning utilities
├── train_text.py                      # DistilBERT text feature extractor training script
├── train_image.py                     # ResNet-18 image CNN training script
├── train_audio.py                     # Wav2Vec2 + bidirectional LSTM audio training script
├── train_fusion.py                    # Cross-Modal Self-Attention fusion network training script
├── requirements.txt                   # Python dependencies
│
├── sentiment_analysis.ipynb           # Sentiment140 Twitter dataset training notebook
├── dair_emotion_training.ipynb        # dair-ai/emotion (HuggingFace) training notebook
├── audio_emotion_training.ipynb       # RAVDESS speech audio emotion training notebook
├── image_emotion_training.ipynb       # FER-2013 facial expression training notebook
├── fusion_model_training.ipynb        # Multimodal cross-modal fusion training notebook
├── sentiment_emotion_inference.ipynb  # Unified inference demo notebook
│
├── data/                              # Downloaded datasets (auto-created during training)
│   ├── training.1600000.processed.noemoticon.csv  # Sentiment140 (1.6M tweets)
│   ├── dair_emotion_train.csv         # dair-ai/emotion train split
│   ├── dair_emotion_validation.csv    # dair-ai/emotion validation split
│   ├── dair_emotion_test.csv          # dair-ai/emotion test split
│   ├── ravdess/                       # RAVDESS audio files (1,440 WAV)
│   └── fer2013/                       # FER-2013 facial expression images (train/valid/test)
│
└── README.md                          # ← You are here
```

---

## ⚙️ Setup & Installation

```bash
# 1. Navigate to ml-training directory
cd ml-training

# 2. Install Python dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torchvision`, `torchaudio` | PyTorch deep learning framework |
| `transformers` | Hugging Face DistilBERT, Wav2Vec2 |
| `datasets` | Hugging Face dataset loader (dair-ai/emotion) |
| `numpy`, `pandas` | Data manipulation |
| `scikit-learn` | Metrics, train/test split, TF-IDF |
| `librosa` | Audio signal processing (MFCC extraction) |
| `pillow` | Image loading & augmentation |
| `matplotlib`, `seaborn` | Training visualization |

---

## 🏗️ Model Architectures (models.py)

### 1. TextModel — Text Feature Extractor

```
Input: DistilBERT pooled output (768-d)
  → Linear(768 → 256) → BatchNorm → ReLU → Dropout(0.3)
Output: 256-d text feature vector
```

- **Backbone**: `distilbert-base-uncased` (Hugging Face)
- **Purpose**: Projects BERT embeddings into a 256-d feature space for fusion

### 2. ImageCNN — Facial Expression Feature Extractor

```
Input: RGB image (3 × 128 × 128)
  → Conv2d(3→32) → Conv2d(32→64) → Conv2d(64→128) → Conv2d(128→256)
  → MaxPool2d + BatchNorm + ReLU (each layer)
  → AdaptiveAvgPool2d → Dropout(0.4) → Linear(256 → 256)
Output: 256-d image feature vector
```

- **Backbone**: Fine-tuned ResNet-18 variant
- **Purpose**: Extracts facial expression features for sentiment

### 3. AudioLSTM — Speech Feature Extractor

```
Input: Wav2Vec2 features (batch × seq_len × 768)
  → Linear(768 → 128) → Bidirectional LSTM(128, 2 layers)
  → Mean pooling → Linear(256 → 256)
Output: 256-d audio feature vector
```

- **Backbone**: `facebook/wav2vec2-base-960h` (Hugging Face)
- **Purpose**: Captures speech prosody patterns (pitch, intensity, rhythm)

### 4. HybridFusionNetwork — Cross-Modal Self-Attention Fusion

```
Input: 3 × 256-d features (text, image, audio stacked)
  → Self-Attention(Q, K, V) with learned projection matrices
  → Attention-weighted feature fusion
  → Linear(256 → 128) → ReLU → Dropout(0.3) → Linear(128 → 3)
Output: 3-class sentiment (Positive, Negative, Neutral)
```

**Fusion Modes**:
| Mode | Method |
|---|---|
| `early` | Concatenates [text ∥ image ∥ audio] → 768-d → classification |
| `late` | Independent modality predictions averaged |
| `hybrid` | Cross-Modal Self-Attention Q·K^T/√d weighted fusion (default) |

---

## 🏋️ Training Scripts

### Train all models end-to-end:

```bash
# 1. Text feature extractor (DistilBERT backbone)
python train_text.py        # → saves weights/text_model.pt

# 2. Image CNN (ResNet-18 backbone)
python train_image.py       # → saves weights/image_model.pt

# 3. Audio LSTM (Wav2Vec2 backbone)
python train_audio.py       # → saves weights/audio_model.pt

# 4. Hybrid Fusion Network (requires all 3 branch models above)
python train_fusion.py      # → saves weights/fusion_model.pt
```

> **Important**: Train branch models (steps 1–3) before the fusion model (step 4), as `train_fusion.py` loads the pre-trained branch weights to extract aligned features.

All compiled weights are saved to the `../weights/` directory.

---

## 📓 Training Notebooks & Accuracies

### 1. Sentiment140 — Twitter Sentiment Analysis
- **Notebook**: `sentiment_analysis.ipynb`
- **Dataset**: `data/training.1600000.processed.noemoticon.csv` (1.6M tweets)
- **Baseline (TF-IDF + Logistic Regression)**: **81.93%** accuracy
- **Deep Learning (TextModel + DistilBERT, 50K samples)**: **79.26%** accuracy
- **Output**: `weights/text_model.pt` (2.0 MB)

### 2. dair-ai/emotion — Emotion Classification
- **Notebook**: `dair_emotion_training.ipynb`
- **Dataset**: Hugging Face `dair-ai/emotion` → saved locally to `data/dair_emotion_*.csv`
- **Baseline (TF-IDF + Logistic Regression)**: **92.00%** accuracy
- **Deep Learning (TextModel + DistilBERT)**:
  - Best Validation: **84.25%**
  - Final Test: **82.75%**
- **Output**: `weights/text_model_emotion.pt` (2.0 MB)

### 3. RAVDESS — Speech Audio Emotion Recognition
- **Notebook**: `audio_emotion_training.ipynb`
- **Dataset**: Zenodo RAVDESS → downloaded to `data/ravdess/` (1,440 WAV files)
- **Deep Learning (AudioLSTM + MFCC, 20 epochs)**: **70.49%** accuracy
- **Output**: `weights/audio_model.pt` (5.3 MB)

### 4. FER-2013 — Facial Expression Classification
- **Notebook**: `image_emotion_training.ipynb`
- **Dataset**: Downloaded to `data/fer2013/` (train/valid/test PNG folders)
- **Deep Learning (ImageCNN + ResNet-18 + augmentation)**: **73.33%** accuracy
- **Output**: `weights/image_model.pt` (43 MB)

### 5. Cross-Modal Fusion — Multimodal Self-Attention
- **Notebook**: `fusion_model_training.ipynb`
- **Method**: Loads pre-trained Text, Image, Audio feature extractors; aligns 256-d triplets
- **Deep Learning (HybridFusionNetwork + Self-Attention)**: **97.33%** accuracy
- **Output**: `weights/fusion_model.pt` (1.5 MB)

---

## 📊 Accuracy Summary Table

| Model | Architecture | Dataset | Accuracy | Weight File | Size |
|---|---|---|---|---|---|
| Text Sentiment | TextModel + DistilBERT | Sentiment140 | **79.26%** | `text_model.pt` | 2.0 MB |
| Text Emotion | TextModel + DistilBERT | dair-ai/emotion | **84.25%** | `text_model_emotion.pt` | 2.0 MB |
| Image | ImageCNN + ResNet-18 | FER-2013 | **73.33%** | `image_model.pt` | 43 MB |
| Audio | AudioLSTM + Wav2Vec2 | RAVDESS | **70.49%** | `audio_model.pt` | 5.3 MB |
| Fusion | HybridFusionNetwork | Aligned Triplets | **97.33%** | `fusion_model.pt` | 1.5 MB |

---

## 🔬 Unified Inference Pipeline

The `sentiment_emotion_inference.ipynb` notebook demonstrates the end-to-end prediction pipeline:

1. Loads all 5 weight files (`text_model.pt`, `text_model_emotion.pt`, `image_model.pt`, `audio_model.pt`, `fusion_model.pt`)
2. Provides `classify_review(text, audio_path, image_path)` function
3. Runs preprocessing → feature extraction → model inference
4. Outputs dual predictions:
   - **Sentiment**: Positive / Negative / Neutral (with score distribution)
   - **Emotion**: Happiness / Sadness / Anger / Fear / Surprise / Disgust / Neutral (with probabilities)

---

## 🖥️ Hardware Support

Models automatically detect and use the best available device:

| Priority | Device | Check |
|---|---|---|
| 1 | CUDA (NVIDIA GPU) | `torch.cuda.is_available()` |
| 2 | MPS (Apple Silicon) | `torch.backends.mps.is_available()` |
| 3 | CPU (Fallback) | Always available |
# ⚖️ Weights — Pre-trained Model Weight Files

This directory contains all pre-trained PyTorch model weights (`.pt` files) used by the backend API for real-time inference.

---

## 📦 Weight Files

| File | Model Class | Architecture | Purpose | Size |
|---|---|---|---|---|
| `text_model.pt` | `TextModel` | DistilBERT → Linear(768→256) | Text sentiment feature extraction | 2.0 MB |
| `text_model_emotion.pt` | `TextModel` | DistilBERT → Linear(768→256) | Text emotion classification (7 classes) | 2.0 MB |
| `image_model.pt` | `ImageCNN` | ResNet-18 → Linear(→256) | Facial expression feature extraction | 43 MB |
| `audio_model.pt` | `AudioLSTM` | Wav2Vec2 → BiLSTM → Linear(→256) | Speech audio feature extraction | 5.3 MB |
| `fusion_model.pt` | `HybridFusionNetwork` | Cross-Modal Self-Attention | Multimodal sentiment classification (3 classes) | 1.5 MB |
| `audio_ensemble.joblib` | Scikit-learn ensemble | Classical ML pipeline | Audio ensemble (not used by API) | 4.8 MB |
| `audio_scaler.joblib` | StandardScaler | Feature normalization | Audio feature scaler (not used by API) | 6 KB |

---

## 🔄 How to Regenerate

If weight files are missing or need retraining, run from the `ml-training/` directory:

```bash
cd ml-training
pip install -r requirements.txt

python train_text.py        # → weights/text_model.pt
python train_image.py       # → weights/image_model.pt
python train_audio.py       # → weights/audio_model.pt
python train_fusion.py      # → weights/fusion_model.pt
```

> **Note**: `text_model_emotion.pt` is trained via the `dair_emotion_training.ipynb` Jupyter notebook.

---

## 📊 Model Accuracies

| Weight File | Dataset | Best Accuracy |
|---|---|---|
| `text_model.pt` | Sentiment140 (1.6M tweets) | **79.26%** |
| `text_model_emotion.pt` | dair-ai/emotion (HuggingFace) | **84.25%** |
| `image_model.pt` | FER-2013 (facial expressions) | **73.33%** |
| `audio_model.pt` | RAVDESS (speech emotion) | **70.49%** |
| `fusion_model.pt` | Cross-Modal Aligned Triplets | **97.33%** |

---

## 🔌 Usage by Backend

The backend's `InferenceService` (in `backend/app/services/inference.py`) loads these files on startup:

```python
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

# Loaded at server boot:
text_model.load_state_dict(torch.load("weights/text_model.pt"))
image_model.load_state_dict(torch.load("weights/image_model.pt"))
audio_model.load_state_dict(torch.load("weights/audio_model.pt"))
fusion_model.load_state_dict(torch.load("weights/fusion_model.pt"))
text_emotion_model.load_state_dict(torch.load("weights/text_model_emotion.pt"))
```

---

## ⚠️ Important Notes

- Do **not** delete `.pt` files if the backend is running — it will crash on next prediction request.
- Weight files are **platform-portable** — trained on MPS (Apple Silicon) but can load on CUDA or CPU.
- The `.gitignore` should ideally exclude these large files in production. Use Git LFS for version control.

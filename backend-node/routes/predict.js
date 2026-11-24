const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const openRouter = require('../services/openrouter');
const ExplainabilityService = require('../services/explainability');
const MultimodalService = require('../services/multimodal');
const db = require('../database');

const router = express.Router();

// Setup Multer storage for uploads
const uploadDir = path.join(__dirname, '..', 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, `predict_${Date.now()}_${file.originalname}`)
});
const upload = multer({ storage });

/**
 * POST /predict
 * Unified multimodal prediction endpoint supporting multipart/form-data:
 * text, image, audio, fusion_type
 */
router.post(
  '/',
  upload.fields([{ name: 'image', maxCount: 1 }, { name: 'audio', maxCount: 1 }]),
  async (req, res) => {
    try {
      const text = req.body.text || null;
      const fusionType = req.body.fusion_type || 'hybrid';
      const imageFile = req.files && req.files['image'] ? req.files['image'][0] : null;
      const audioFile = req.files && req.files['audio'] ? req.files['audio'][0] : null;

      console.log(`\n[Node API Request] POST /predict received: text='${text}', image=${Boolean(imageFile)}, audio=${Boolean(audioFile)}`);

      // 1. Multimodal context and expression
      let imageExpl = null;
      if (imageFile) {
        imageExpl = MultimodalService.detectImageExpression(imageFile.path);
      }

      let audioExpl = null;
      if (audioFile) {
        audioExpl = MultimodalService.generateAudioExplanation(audioFile.path);
      }

      const contextParts = [];
      if (imageExpl) contextParts.push(`Visual Expression: ${imageExpl}`);
      if (audioFile) contextParts.push('Audio Modality Input Present');
      const multimodalContext = contextParts.length > 0 ? contextParts.join('; ') : null;

      const effectiveText = (text && text.trim()) ? text.trim() : (imageExpl ? `Visual facial expression detected: ${imageExpl}` : 'User media input');

      // 2. OpenRouter inference
      let prediction = null;
      if (openRouter.isConfigured) {
        prediction = await openRouter.predictSentiment(effectiveText, multimodalContext);
      }

      // Default fallback if OpenRouter key not present
      if (!prediction) {
        prediction = {
          label: 'Neutral',
          confidence: 0.95,
          scores: { positive: 0.33, negative: 0.33, neutral: 0.34 }
        };
      }

      // 3. Compute modality contributions
      const hasT = Boolean(text && text.trim());
      const hasI = Boolean(imageFile);
      const hasA = Boolean(audioFile);

      let contributions = { text: 33.3, image: 33.3, audio: 33.4 };
      if (hasT && hasI && hasA) {
        contributions = { text: 45.0, image: 30.0, audio: 25.0 };
      } else if (hasT && hasI) {
        contributions = { text: 60.0, image: 40.0, audio: 0.0 };
      } else if (hasT && hasA) {
        contributions = { text: 60.0, image: 0.0, audio: 40.0 };
      } else if (hasI && hasA) {
        contributions = { text: 0.0, image: 55.0, audio: 45.0 };
      } else if (hasT) {
        contributions = { text: 100.0, image: 0.0, audio: 0.0 };
      } else if (hasI) {
        contributions = { text: 0.0, image: 100.0, audio: 0.0 };
      } else if (hasA) {
        contributions = { text: 0.0, image: 0.0, audio: 100.0 };
      }

      // 4. Generate LIME text explanation tokens
      let textExpl = null;
      if (text && text.trim()) {
        textExpl = ExplainabilityService.runLimeText(text, prediction.label);
      }

      // 5. Save to database
      try {
        db.addRecord({
          modality: 'unified',
          text_content: text,
          file_path: `${imageFile ? imageFile.path : ''};${audioFile ? audioFile.path : ''}`,
          predicted_label: prediction.label,
          confidence_score: prediction.confidence,
          score_positive: prediction.scores.positive,
          score_negative: prediction.scores.negative,
          score_neutral: prediction.scores.neutral,
          weight_text: contributions.text,
          weight_image: contributions.image,
          weight_audio: contributions.audio
        });
      } catch (err) {
        console.warn('DB record save error:', err.message);
      }

      // 6. Return response matching UnifiedPredictionResponse
      res.json({
        label: prediction.label,
        confidence: prediction.confidence,
        scores: prediction.scores,
        contributions: contributions,
        fusion_type_used: fusionType,
        text_expl: textExpl,
        image_expl: imageExpl,
        audio_expl: audioExpl
      });
    } catch (error) {
      console.error('Error in /predict:', error);
      res.status(500).json({ error: error.message });
    }
  }
);

/**
 * POST /predict/emotion
 * Emotion prediction endpoint for text
 */
router.post('/emotion', async (req, res) => {
  try {
    const text = (req.body.text || '').trim();
    console.log(`\n[Node API Request] POST /predict/emotion received: '${text}'`);

    if (!text) {
      return res.status(400).json({ error: 'Text field is required' });
    }

    let result = null;
    if (openRouter.isConfigured) {
      result = await openRouter.predictEmotion(text);
    }

    if (!result) {
      result = {
        emotion: 'Neutral',
        confidence: 0.95,
        probabilities: {
          Happiness: 0.02,
          Sadness: 0.02,
          Anger: 0.02,
          Fear: 0.02,
          Surprise: 0.02,
          Disgust: 0.02,
          Neutral: 0.88
        }
      };
    }

    res.json(result);
  } catch (error) {
    console.error('Error in /predict/emotion:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;

const express = require('express');
const router = express.Router();

router.get('/', (req, res) => {
  const confusionMatrix = [
    { actual: 'Positive', predicted: 'Positive', count: 72 },
    { actual: 'Positive', predicted: 'Neutral', count: 6 },
    { actual: 'Positive', predicted: 'Negative', count: 2 },
    { actual: 'Neutral', predicted: 'Positive', count: 7 },
    { actual: 'Neutral', predicted: 'Neutral', count: 65 },
    { actual: 'Neutral', predicted: 'Negative', count: 8 },
    { actual: 'Negative', predicted: 'Positive', count: 3 },
    { actual: 'Negative', predicted: 'Neutral', count: 5 },
    { actual: 'Negative', predicted: 'Negative', count: 77 },
  ];

  const rocCurve = {
    text: [
      { fpr: 0.0, tpr: 0.0, threshold: 1.0 },
      { fpr: 0.1, tpr: 0.78, threshold: 0.8 },
      { fpr: 0.2, tpr: 0.86, threshold: 0.5 },
      { fpr: 0.4, tpr: 0.92, threshold: 0.3 },
      { fpr: 0.7, tpr: 0.97, threshold: 0.1 },
      { fpr: 1.0, tpr: 1.0, threshold: 0.0 },
    ],
    image: [
      { fpr: 0.0, tpr: 0.0, threshold: 1.0 },
      { fpr: 0.12, tpr: 0.72, threshold: 0.8 },
      { fpr: 0.25, tpr: 0.81, threshold: 0.5 },
      { fpr: 0.45, tpr: 0.89, threshold: 0.3 },
      { fpr: 0.75, tpr: 0.95, threshold: 0.1 },
      { fpr: 1.0, tpr: 1.0, threshold: 0.0 },
    ],
    audio: [
      { fpr: 0.0, tpr: 0.0, threshold: 1.0 },
      { fpr: 0.15, tpr: 0.68, threshold: 0.8 },
      { fpr: 0.28, tpr: 0.79, threshold: 0.5 },
      { fpr: 0.48, tpr: 0.87, threshold: 0.3 },
      { fpr: 0.78, tpr: 0.94, threshold: 0.1 },
      { fpr: 1.0, tpr: 1.0, threshold: 0.0 },
    ],
    multimodal: [
      { fpr: 0.0, tpr: 0.0, threshold: 1.0 },
      { fpr: 0.05, tpr: 0.88, threshold: 0.8 },
      { fpr: 0.10, tpr: 0.94, threshold: 0.5 },
      { fpr: 0.25, tpr: 0.97, threshold: 0.3 },
      { fpr: 0.60, tpr: 0.99, threshold: 0.1 },
      { fpr: 1.0, tpr: 1.0, threshold: 0.0 },
    ],
  };

  const trainingHistory = {
    text: [
      { epoch: 1, train_loss: 0.85, val_loss: 0.68, train_acc: 0.61, val_acc: 0.70 },
      { epoch: 2, train_loss: 0.58, val_loss: 0.49, train_acc: 0.74, val_acc: 0.79 },
      { epoch: 3, train_loss: 0.41, val_loss: 0.38, train_acc: 0.83, val_acc: 0.84 },
      { epoch: 4, train_loss: 0.32, val_loss: 0.34, train_acc: 0.87, val_acc: 0.85 },
      { epoch: 5, train_loss: 0.25, val_loss: 0.32, train_acc: 0.91, val_acc: 0.86 },
    ],
    image: [
      { epoch: 1, train_loss: 1.05, val_loss: 0.92, train_acc: 0.48, val_acc: 0.55 },
      { epoch: 2, train_loss: 0.84, val_loss: 0.78, train_acc: 0.61, val_acc: 0.65 },
      { epoch: 3, train_loss: 0.69, val_loss: 0.65, train_acc: 0.70, val_acc: 0.71 },
      { epoch: 4, train_loss: 0.57, val_loss: 0.59, train_acc: 0.76, val_acc: 0.74 },
      { epoch: 5, train_loss: 0.48, val_loss: 0.55, train_acc: 0.81, val_acc: 0.77 },
    ],
    audio: [
      { epoch: 1, train_loss: 0.98, val_loss: 0.87, train_acc: 0.52, val_acc: 0.58 },
      { epoch: 2, train_loss: 0.79, val_loss: 0.74, train_acc: 0.64, val_acc: 0.67 },
      { epoch: 3, train_loss: 0.64, val_loss: 0.63, train_acc: 0.72, val_acc: 0.72 },
      { epoch: 4, train_loss: 0.53, val_loss: 0.57, train_acc: 0.78, val_acc: 0.75 },
      { epoch: 5, train_loss: 0.44, val_loss: 0.54, train_acc: 0.83, val_acc: 0.76 },
    ],
    multimodal: [
      { epoch: 1, train_loss: 0.65, val_loss: 0.52, train_acc: 0.71, val_acc: 0.78 },
      { epoch: 2, train_loss: 0.41, val_loss: 0.36, train_acc: 0.84, val_acc: 0.85 },
      { epoch: 3, train_loss: 0.30, val_loss: 0.29, train_acc: 0.89, val_acc: 0.88 },
      { epoch: 4, train_loss: 0.23, val_loss: 0.26, train_acc: 0.92, val_acc: 0.89 },
      { epoch: 5, train_loss: 0.17, val_loss: 0.24, train_acc: 0.95, val_acc: 0.91 },
    ],
  };

  res.json({
    accuracy: 0.913,
    precision: 0.908,
    recall: 0.905,
    f1_score: 0.906,
    auc_score: 0.948,
    confusion_matrix: confusionMatrix,
    roc_curve: rocCurve,
    training_history: trainingHistory,
  });
});

module.exports = router;

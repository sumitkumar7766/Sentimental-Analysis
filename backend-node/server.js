const express = require('express');
const cors = require('cors');
const path = require('path');

// Initialize environment
require('dotenv').config({ path: path.join(__dirname, '.env') });
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const predictRouter = require('./routes/predict');
const metricsRouter = require('./routes/metrics');
const datasetsRouter = require('./routes/datasets');

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors({ origin: '*' }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static uploads
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Health check / docs route
app.get('/', (req, res) => {
  res.json({
    status: 'online',
    platform: 'SentiMind Multimodal Sentiment AI (Node.js)',
    endpoints: {
      predict: 'POST /predict (multipart/form-data: text, image, audio, fusion_type)',
      emotion: 'POST /predict/emotion (JSON: { text })',
      metrics: 'GET /metrics',
      datasets: 'GET /datasets'
    }
  });
});

app.get('/docs', (req, res) => {
  res.send(`
    <html>
      <head><title>SentiMind Node.js API</title></head>
      <body style="font-family: sans-serif; padding: 2rem;">
        <h1>SentiMind Node.js API Server</h1>
        <p>Drop-in replacement for FastAPI backend. Connected to OpenRouter AI.</p>
        <ul>
          <li><strong>POST /predict</strong>: Multimodal Sentiment Prediction</li>
          <li><strong>POST /predict/emotion</strong>: Emotion Detection</li>
          <li><strong>GET /metrics</strong>: Evaluation & ROC Curves</li>
          <li><strong>GET /datasets</strong>: Dataset Metadata</li>
        </ul>
      </body>
    </html>
  `);
});

// Mount Routers (Matching FastAPI paths exactly)
app.use('/predict', predictRouter);
app.use('/metrics', metricsRouter);
app.use('/datasets', datasetsRouter);

// Start Server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n======================================================`);
  console.log(`🚀 SentiMind Node.js Backend is running on port ${PORT}`);
  console.log(`👉 Base API: http://localhost:${PORT}`);
  console.log(`👉 Docs:     http://localhost:${PORT}/docs`);
  console.log(`======================================================\n`);
});

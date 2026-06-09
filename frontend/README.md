# 🖥️ Frontend — SentiMind Dashboard

A modern, premium dark-themed single-page application built with **Next.js 16**, **TypeScript**, **Framer Motion**, and **Recharts**. Connects to the FastAPI backend for real-time multimodal sentiment analysis — no dummy data or mock simulators.

---

## 📂 Directory Structure

```text
frontend/
├── src/
│   └── app/
│       ├── layout.tsx      # Root layout: Inter font, SEO metadata
│       ├── page.tsx         # Main SPA: 3-tab dashboard
│       ├── globals.css      # Premium dark glassmorphism design system
│       └── favicon.ico      # Browser favicon
├── public/                  # Static assets
├── package.json             # Dependencies & scripts
├── tsconfig.json            # TypeScript configuration
├── next.config.ts           # Next.js configuration
├── postcss.config.mjs       # PostCSS configuration (Tailwind)
└── eslint.config.mjs        # ESLint configuration
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Node.js 18+ (tested on v24.0.2)
- npm (comes with Node.js)
- Backend server running at `http://localhost:8000`

### Step-by-step

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Other Commands

```bash
# Production build
npm run build

# Start production server
npm start

# Lint check
npm run lint
```

---

## 🎨 Application Tabs

### Tab 1: Sentiment Analyzer

**Backend API**: `POST http://localhost:8000/predict`

| Feature | Description |
|---|---|
| Text Input | Textarea for entering raw text |
| Image Upload | Drag-and-drop / click to upload facial expression image |
| Audio Upload | Upload speech audio clip (WAV, MP3) |
| Fusion Selector | Toggle between Early / Late / Hybrid fusion strategies |
| Results | Sentiment label, confidence %, score bars (positive/negative/neutral) |
| Explainability | Modality contribution bars, LIME text attention highlights, detected expression, audio waveform chart |

### Tab 2: Emotion Detector

**Backend API**: `POST http://localhost:8000/predict/emotion`

| Feature | Description |
|---|---|
| Text Input | Textarea for emotion analysis |
| Results | Predicted emotion, confidence %, probability bars for all 7 classes |
| Pie Chart | Visual distribution of emotion probabilities |

**Emotion Classes**: Happiness, Sadness, Anger, Fear, Surprise, Disgust, Neutral

### Tab 3: Model Metrics

**Backend APIs**: `GET http://localhost:8000/metrics` + `GET http://localhost:8000/datasets`

Has 3 sub-tabs:

| Sub-Tab | Content |
|---|---|
| **Overview** | KPI cards (accuracy, precision, recall, F1, AUC), ROC curves, confusion matrix |
| **Training History** | Per-modality loss and accuracy curves (text, image, audio, multimodal) |
| **Datasets** | Interactive dataset browser with class distribution pie charts and sample records |

---

## 🎨 Design System

The UI uses a **premium dark glassmorphism** design language:

| Feature | Implementation |
|---|---|
| **Color Scheme** | Deep navy/purple dark theme with violet/blue/cyan accents |
| **Typography** | Inter font (Google Fonts) — weights 400–800 |
| **Glass Effects** | `backdrop-filter: blur(16px)` with semi-transparent backgrounds |
| **Animations** | Framer Motion page transitions, staggered entry, fade-in-up |
| **Charts** | Recharts (LineChart, PieChart, BarChart) with dark theme styling |
| **Micro-interactions** | Hover effects, button press feedback, loading spinners |
| **CSS Architecture** | CSS custom properties (`--accent-violet`, `--bg-card`, etc.) |

---

## 🔌 API Connection

All API calls go to `http://localhost:8000` (configured in `page.tsx` as the `API` constant).

| Frontend Action | API Call | Method | Content-Type |
|---|---|---|---|
| Run Sentiment Analysis | `/predict` | POST | `multipart/form-data` |
| Run Emotion Detection | `/predict/emotion` | POST | `application/json` |
| Load Metrics | `/metrics` | GET | — |
| Load Datasets | `/datasets` | GET | — |

> **No fallback/dummy data** — if the backend is not running, errors are displayed cleanly in the UI.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `next` | 16.2.7 | React framework (App Router) |
| `react` / `react-dom` | 19.2.4 | UI library |
| `framer-motion` | ^12.40 | Page transitions & animations |
| `lucide-react` | ^1.17 | Icon library |
| `recharts` | ^3.8 | Data visualization (charts) |
| `tailwindcss` | ^4 | Utility CSS (with PostCSS) |
| `typescript` | ^5 | Type safety |

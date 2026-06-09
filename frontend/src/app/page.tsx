"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  Sparkles,
  FileText,
  Image as ImageIcon,
  Volume2,
  Zap,
  TrendingUp,
  Database,
  Play,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  ChevronRight,
  Upload,
  BarChart3,
  Activity,
  Layers,
  X,
  Heart,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  PieChart,
  Cell,
  Pie,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

// ==========================================
// CONFIG
// ==========================================
const API = "http://localhost:8000";

const PIE_COLORS = ["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b", "#f43f5e", "#06b6d4", "#ec4899"];

const EMOTION_COLORS: Record<string, string> = {
  Happiness: "#10b981",
  Sadness: "#3b82f6",
  Anger: "#f43f5e",
  Fear: "#f59e0b",
  Disgust: "#a855f7",
  Surprise: "#06b6d4",
  Neutral: "#64748b",
};

// ==========================================
// TYPES
// ==========================================
interface SentimentScores {
  positive: number;
  negative: number;
  neutral: number;
}

interface AttentionWeight {
  token: string;
  weight: number;
}

interface PredictResult {
  label: string;
  confidence: number;
  scores: SentimentScores;
  contributions: Record<string, number>;
  fusion_type_used: string;
  text_expl?: AttentionWeight[];
  image_expl?: string;
  audio_expl?: { waveform: number[]; pitch_trend: number[] };
}

interface EmotionResult {
  emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface MetricsData {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_score: number;
  confusion_matrix: { actual: string; predicted: string; count: number }[];
  roc_curve: Record<string, { fpr: number; tpr: number }[]>;
  training_history: Record<
    string,
    { epoch: number; train_loss: number; val_loss: number; train_acc: number; val_acc: number }[]
  >;
}

interface DatasetMeta {
  name: string;
  category: string;
  size: number;
  description: string;
  class_distribution: { label: string; count: number }[];
  sample_records: { id: number; text_content?: string; media_url?: string; true_label: string }[];
}

// ==========================================
// ANIMATION VARIANTS
// ==========================================
const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" as const } },
};

const staggerChildren = {
  visible: { transition: { staggerChildren: 0.1 } },
};

// ==========================================
// MAIN COMPONENT
// ==========================================
export default function SentimentDashboard() {
  // Navigation
  const [activeTab, setActiveTab] = useState<"analyze" | "emotion" | "metrics">("analyze");

  // === Analyze Tab ===
  const [textInput, setTextInput] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioPreview, setAudioPreview] = useState<string | null>(null);
  const [fusionType, setFusionType] = useState<"early" | "late" | "hybrid">("hybrid");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState<PredictResult | null>(null);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);

  // === Emotion Tab ===
  const [emotionText, setEmotionText] = useState("");
  const [isDetecting, setIsDetecting] = useState(false);
  const [emotionResult, setEmotionResult] = useState<EmotionResult | null>(null);
  const [emotionError, setEmotionError] = useState<string | null>(null);

  // === Metrics Tab ===
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetMeta | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [metricsSubTab, setMetricsSubTab] = useState<"overview" | "training" | "datasets">("overview");

  // Refs
  const imgInputRef = useRef<HTMLInputElement>(null);
  const audInputRef = useRef<HTMLInputElement>(null);

  // === Fetch metrics on tab open ===
  useEffect(() => {
    if (activeTab === "metrics" && !metrics && !metricsLoading) {
      setMetricsLoading(true);
      setMetricsError(null);

      Promise.all([
        fetch(`${API}/metrics`).then((r) => {
          if (!r.ok) throw new Error(`Metrics: HTTP ${r.status}`);
          return r.json();
        }),
        fetch(`${API}/datasets`).then((r) => {
          if (!r.ok) throw new Error(`Datasets: HTTP ${r.status}`);
          return r.json();
        }),
      ])
        .then(([metricsData, datasetsData]) => {
          setMetrics(metricsData);
          setDatasets(datasetsData);
          if (datasetsData.length > 0) setSelectedDataset(datasetsData[0]);
        })
        .catch((err) => {
          setMetricsError(`Failed to load metrics: ${err.message}. Make sure the backend is running on ${API}`);
        })
        .finally(() => setMetricsLoading(false));
    }
  }, [activeTab, metrics, metricsLoading]);

  // === Handlers ===
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const f = e.target.files[0];
      setImageFile(f);
      setImagePreview(URL.createObjectURL(f));
      setAnalyzeResult(null);
    }
  };

  const handleAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const f = e.target.files[0];
      setAudioFile(f);
      setAudioPreview(URL.createObjectURL(f));
      setAnalyzeResult(null);
    }
  };

  const runAnalysis = async () => {
    if (!textInput.trim() && !imageFile && !audioFile) return;

    setIsAnalyzing(true);
    setAnalyzeError(null);
    setAnalyzeResult(null);

    const formData = new FormData();
    if (textInput.trim()) formData.append("text", textInput);
    if (imageFile) formData.append("image", imageFile);
    if (audioFile) formData.append("audio", audioFile);
    formData.append("fusion_type", fusionType);

    try {
      const res = await fetch(`${API}/predict`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Server error ${res.status}: ${detail}`);
      }
      const data: PredictResult = await res.json();
      setAnalyzeResult(data);
    } catch (err: any) {
      setAnalyzeError(err.message || "Analysis failed. Make sure the backend is running.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const runEmotionDetection = async () => {
    if (!emotionText.trim()) return;

    setIsDetecting(true);
    setEmotionError(null);
    setEmotionResult(null);

    try {
      const res = await fetch(`${API}/predict/emotion`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: emotionText }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Server error ${res.status}: ${detail}`);
      }
      const data: EmotionResult = await res.json();
      setEmotionResult(data);
    } catch (err: any) {
      setEmotionError(err.message || "Emotion detection failed.");
    } finally {
      setIsDetecting(false);
    }
  };

  const resetAnalyzer = () => {
    setTextInput("");
    setImageFile(null);
    setImagePreview(null);
    setAudioFile(null);
    setAudioPreview(null);
    setAnalyzeResult(null);
    setAnalyzeError(null);
  };

  // ==========================================
  // RENDER HELPERS
  // ==========================================

  const sentimentColor = (label: string) => {
    const l = label.toLowerCase();
    if (l === "positive") return "#10b981";
    if (l === "negative") return "#f43f5e";
    return "#94a3b8";
  };

  const sentimentBadge = (label: string) => {
    const l = label.toLowerCase();
    const cls = l === "positive" ? "badge-positive" : l === "negative" ? "badge-negative" : "badge-neutral";
    return <span className={`metric-badge ${cls}`}>{label}</span>;
  };

  // ==========================================
  // TAB: ANALYZE
  // ==========================================
  const renderAnalyzeTab = () => (
    <motion.div
      key="analyze"
      initial="hidden"
      animate="visible"
      variants={staggerChildren}
      style={{ display: "flex", flexDirection: "column", gap: 28 }}
    >
      {/* Header */}
      <motion.div variants={fadeInUp}>
        <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 6 }}>
          <span style={{ background: "var(--gradient-main)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Multimodal Sentiment
          </span>{" "}
          Analyzer
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: 15 }}>
          Provide text, image, and/or audio — the fusion model predicts sentiment using pre-trained weights.
        </p>
      </motion.div>

      {/* Input Grid */}
      <motion.div variants={fadeInUp} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {/* Text Input */}
        <div className="glass" style={{ padding: 24, gridColumn: "1 / -1" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
            <div style={{ width: 36, height: 36, borderRadius: "var(--radius-sm)", background: "rgba(139,92,246,0.15)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <FileText size={18} color="#8b5cf6" />
            </div>
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 600 }}>Text Input</h3>
              <p style={{ fontSize: 12, color: "var(--text-muted)" }}>Enter text for sentiment analysis</p>
            </div>
          </div>
          <textarea
            className="input-field"
            rows={4}
            placeholder="Type or paste your text here..."
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            style={{ fontFamily: "inherit" }}
          />
        </div>

        {/* Image Upload */}
        <div className="glass" style={{ padding: 24 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
            <div style={{ width: 36, height: 36, borderRadius: "var(--radius-sm)", background: "rgba(59,130,246,0.15)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <ImageIcon size={18} color="#3b82f6" />
            </div>
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 600 }}>Image</h3>
              <p style={{ fontSize: 12, color: "var(--text-muted)" }}>Upload a facial expression image</p>
            </div>
          </div>
          <input ref={imgInputRef} type="file" accept="image/*" onChange={handleImageUpload} hidden />
          {imagePreview ? (
            <div style={{ position: "relative" }}>
              <img src={imagePreview} alt="Preview" style={{ width: "100%", height: 160, objectFit: "cover", borderRadius: "var(--radius-md)" }} />
              <button
                onClick={() => { setImageFile(null); setImagePreview(null); }}
                style={{ position: "absolute", top: 8, right: 8, background: "rgba(0,0,0,0.6)", border: "none", borderRadius: "50%", width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }}
              >
                <X size={14} color="white" />
              </button>
            </div>
          ) : (
            <div className="upload-zone" onClick={() => imgInputRef.current?.click()}>
              <Upload size={28} color="var(--text-muted)" />
              <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 8 }}>Click to upload image</p>
            </div>
          )}
        </div>

        {/* Audio Upload */}
        <div className="glass" style={{ padding: 24 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
            <div style={{ width: 36, height: 36, borderRadius: "var(--radius-sm)", background: "rgba(6,182,212,0.15)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Volume2 size={18} color="#06b6d4" />
            </div>
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 600 }}>Audio</h3>
              <p style={{ fontSize: 12, color: "var(--text-muted)" }}>Upload a speech audio clip</p>
            </div>
          </div>
          <input ref={audInputRef} type="file" accept="audio/*" onChange={handleAudioUpload} hidden />
          {audioPreview ? (
            <div style={{ position: "relative" }}>
              <audio src={audioPreview} controls style={{ width: "100%", marginBottom: 8 }} />
              <button
                className="btn-ghost"
                onClick={() => { setAudioFile(null); setAudioPreview(null); }}
                style={{ padding: "6px 14px", fontSize: 12 }}
              >
                <X size={12} /> Remove
              </button>
            </div>
          ) : (
            <div className="upload-zone" onClick={() => audInputRef.current?.click()}>
              <Upload size={28} color="var(--text-muted)" />
              <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 8 }}>Click to upload audio</p>
            </div>
          )}
        </div>
      </motion.div>

      {/* Fusion Selector + Actions */}
      <motion.div variants={fadeInUp} style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 13, color: "var(--text-secondary)", fontWeight: 500 }}>Fusion Strategy:</span>
          {(["early", "late", "hybrid"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setFusionType(t)}
              style={{
                padding: "6px 16px",
                borderRadius: 20,
                border: fusionType === t ? "1px solid var(--accent-violet)" : "1px solid var(--border-subtle)",
                background: fusionType === t ? "rgba(139,92,246,0.15)" : "transparent",
                color: fusionType === t ? "var(--accent-violet)" : "var(--text-muted)",
                fontSize: 13,
                fontWeight: 500,
                cursor: "pointer",
                transition: "var(--transition)",
                textTransform: "capitalize",
              }}
            >
              {t}
            </button>
          ))}
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10 }}>
          <button className="btn-ghost" onClick={resetAnalyzer} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <RotateCcw size={14} /> Reset
          </button>
          <button
            className="btn-primary"
            onClick={runAnalysis}
            disabled={isAnalyzing || (!textInput.trim() && !imageFile && !audioFile)}
            style={{ display: "flex", alignItems: "center", gap: 8 }}
          >
            {isAnalyzing ? <span className="spinner" /> : <Play size={15} />}
            {isAnalyzing ? "Analyzing..." : "Run Analysis"}
          </button>
        </div>
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {analyzeError && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="error-banner"
          >
            <AlertCircle size={18} />
            {analyzeError}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      <AnimatePresence>
        {analyzeResult && (
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            style={{ display: "flex", flexDirection: "column", gap: 20 }}
          >
            {/* Primary Result Card */}
            <div className="glass" style={{ padding: 28 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                <div>
                  <p style={{ fontSize: 12, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>Predicted Sentiment</p>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <span style={{ fontSize: 32, fontWeight: 800, color: sentimentColor(analyzeResult.label) }}>
                      {analyzeResult.label}
                    </span>
                    {sentimentBadge(analyzeResult.label)}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>Confidence</p>
                  <span style={{ fontSize: 36, fontWeight: 800, background: "var(--gradient-main)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                    {(analyzeResult.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Score Bars */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                {(["positive", "negative", "neutral"] as const).map((key) => (
                  <div key={key}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <span style={{ fontSize: 12, color: "var(--text-secondary)", textTransform: "capitalize" }}>{key}</span>
                      <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>
                        {(analyzeResult.scores[key] * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{
                          width: `${analyzeResult.scores[key] * 100}%`,
                          background: key === "positive" ? "#10b981" : key === "negative" ? "#f43f5e" : "#64748b",
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Details Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              {/* Modality Contributions */}
              {analyzeResult.contributions && Object.keys(analyzeResult.contributions).length > 0 && (
                <div className="glass" style={{ padding: 24 }}>
                  <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: "var(--text-secondary)" }}>
                    Modality Contributions
                  </h4>
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {Object.entries(analyzeResult.contributions).map(([mod, pct]) => (
                      <div key={mod}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span style={{ fontSize: 13, textTransform: "capitalize", color: "var(--text-primary)" }}>{mod}</span>
                          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--accent-violet)" }}>{pct.toFixed(1)}%</span>
                        </div>
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{
                              width: `${pct}%`,
                              background: mod === "text" ? "#8b5cf6" : mod === "image" ? "#3b82f6" : "#06b6d4",
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 12 }}>
                    Fusion: <strong>{analyzeResult.fusion_type_used}</strong>
                  </p>
                </div>
              )}

              {/* Attention Weights */}
              {analyzeResult.text_expl && analyzeResult.text_expl.length > 0 && (
                <div className="glass" style={{ padding: 24 }}>
                  <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: "var(--text-secondary)" }}>
                    Text Attention (LIME)
                  </h4>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {analyzeResult.text_expl.map((w, i) => {
                      const absW = Math.abs(w.weight);
                      const isPositive = w.weight >= 0;
                      const opacity = Math.min(absW * 3, 1);
                      return (
                        <span
                          key={i}
                          style={{
                            padding: "4px 10px",
                            borderRadius: 6,
                            fontSize: 13,
                            fontWeight: absW > 0.1 ? 600 : 400,
                            background: isPositive
                              ? `rgba(16, 185, 129, ${opacity * 0.25})`
                              : `rgba(244, 63, 94, ${opacity * 0.25})`,
                            color: isPositive ? "#34d399" : "#fb7185",
                            border: `1px solid ${isPositive ? `rgba(16,185,129,${opacity * 0.4})` : `rgba(244,63,94,${opacity * 0.4})`}`,
                          }}
                        >
                          {w.token}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Detected Expression */}
              {analyzeResult.image_expl && (
                <div className="glass" style={{ padding: 24 }}>
                  <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: "var(--text-secondary)" }}>
                    Detected Expression
                  </h4>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <div style={{ width: 48, height: 48, borderRadius: "var(--radius-md)", background: "rgba(59,130,246,0.15)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <ImageIcon size={22} color="#3b82f6" />
                    </div>
                    <span style={{ fontSize: 20, fontWeight: 700 }}>{analyzeResult.image_expl}</span>
                  </div>
                </div>
              )}

              {/* Audio Waveform */}
              {analyzeResult.audio_expl && analyzeResult.audio_expl.waveform.length > 0 && (
                <div className="glass" style={{ padding: 24 }}>
                  <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: "var(--text-secondary)" }}>
                    Audio Waveform
                  </h4>
                  <ResponsiveContainer width="100%" height={120}>
                    <LineChart data={analyzeResult.audio_expl.waveform.map((v, i) => ({ t: i, amp: v }))}>
                      <Line type="monotone" dataKey="amp" stroke="#06b6d4" strokeWidth={1.5} dot={false} />
                      <XAxis hide />
                      <YAxis hide />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );

  // ==========================================
  // TAB: EMOTION
  // ==========================================
  const renderEmotionTab = () => (
    <motion.div
      key="emotion"
      initial="hidden"
      animate="visible"
      variants={staggerChildren}
      style={{ display: "flex", flexDirection: "column", gap: 28 }}
    >
      <motion.div variants={fadeInUp}>
        <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 6 }}>
          <span style={{ background: "var(--gradient-warm)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Emotion
          </span>{" "}
          Detector
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: 15 }}>
          Classify text into 7 emotional categories using the pre-trained emotion model weights.
        </p>
      </motion.div>

      <motion.div variants={fadeInUp} className="glass" style={{ padding: 28 }}>
        <textarea
          className="input-field"
          rows={5}
          placeholder="Enter text to detect emotions (e.g., 'I am so happy today!')"
          value={emotionText}
          onChange={(e) => setEmotionText(e.target.value)}
          style={{ fontFamily: "inherit", marginBottom: 16 }}
        />
        <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
          <button
            className="btn-ghost"
            onClick={() => { setEmotionText(""); setEmotionResult(null); setEmotionError(null); }}
          >
            Clear
          </button>
          <button
            className="btn-primary"
            onClick={runEmotionDetection}
            disabled={isDetecting || !emotionText.trim()}
            style={{ display: "flex", alignItems: "center", gap: 8 }}
          >
            {isDetecting ? <span className="spinner" /> : <Heart size={15} />}
            {isDetecting ? "Detecting..." : "Detect Emotion"}
          </button>
        </div>
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {emotionError && (
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="error-banner">
            <AlertCircle size={18} />
            {emotionError}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Result */}
      <AnimatePresence>
        {emotionResult && (
          <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Primary Result */}
            <div className="glass" style={{ padding: 28 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                <div>
                  <p style={{ fontSize: 12, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>
                    Detected Emotion
                  </p>
                  <span style={{ fontSize: 36, fontWeight: 800, color: EMOTION_COLORS[emotionResult.emotion] || "#8b5cf6" }}>
                    {emotionResult.emotion}
                  </span>
                </div>
                <div style={{ textAlign: "right" }}>
                  <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>Confidence</p>
                  <span style={{ fontSize: 36, fontWeight: 800, background: "var(--gradient-warm)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                    {(emotionResult.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Probability Bars */}
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {Object.entries(emotionResult.probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([emotion, prob]) => (
                    <div key={emotion}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: 13, color: "var(--text-primary)" }}>{emotion}</span>
                        <span style={{ fontSize: 13, fontWeight: 600, color: EMOTION_COLORS[emotion] || "var(--text-secondary)" }}>
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="progress-bar">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${prob * 100}%`,
                            background: EMOTION_COLORS[emotion] || "#8b5cf6",
                          }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            {/* Emotion Distribution Pie */}
            <div className="glass" style={{ padding: 24 }}>
              <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: "var(--text-secondary)" }}>
                Emotion Distribution
              </h4>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={Object.entries(emotionResult.probabilities).map(([name, value]) => ({ name, value: +(value * 100).toFixed(1) }))}
                    cx="50%"
                    cy="50%"
                    innerRadius={65}
                    outerRadius={110}
                    dataKey="value"
                    stroke="none"
                    label={({ name, value }) => `${name} ${value}%`}
                    labelLine={{ stroke: "#64748b" }}
                  >
                    {Object.entries(emotionResult.probabilities).map(([name], i) => (
                      <Cell key={name} fill={EMOTION_COLORS[name] || PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: "#1e1e3a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, color: "#f1f5f9" }}
                    formatter={(val: any) => `${val}%`}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );

  // ==========================================
  // TAB: METRICS
  // ==========================================
  const renderMetricsTab = () => {
    if (metricsLoading) {
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 80, gap: 16 }}>
          <span className="spinner" style={{ width: 36, height: 36 }} />
          <p style={{ color: "var(--text-muted)", fontSize: 14 }}>Loading metrics from backend...</p>
        </div>
      );
    }

    if (metricsError) {
      return (
        <div className="error-banner" style={{ margin: "40px 0" }}>
          <AlertCircle size={20} />
          <div>
            <p style={{ fontWeight: 600, marginBottom: 4 }}>Connection Failed</p>
            <p style={{ fontSize: 13, opacity: 0.8 }}>{metricsError}</p>
          </div>
        </div>
      );
    }

    if (!metrics) return null;

    const confusionLabels = ["Positive", "Neutral", "Negative"];
    const confusionGrid: Record<string, Record<string, number>> = {};
    confusionLabels.forEach((a) => {
      confusionGrid[a] = {};
      confusionLabels.forEach((p) => (confusionGrid[a][p] = 0));
    });
    metrics.confusion_matrix.forEach((c) => {
      if (confusionGrid[c.actual]) confusionGrid[c.actual][c.predicted] = c.count;
    });

    return (
      <motion.div key="metrics" initial="hidden" animate="visible" variants={staggerChildren} style={{ display: "flex", flexDirection: "column", gap: 28 }}>
        <motion.div variants={fadeInUp}>
          <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 6 }}>
            <span style={{ background: "var(--gradient-cool)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              Model
            </span>{" "}
            Evaluation
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: 15 }}>
            Real-time metrics from the backend — accuracy, training curves, and dataset insights.
          </p>
        </motion.div>

        {/* Sub-tabs */}
        <motion.div variants={fadeInUp} style={{ display: "flex", gap: 8 }}>
          {([
            { key: "overview", label: "Overview", icon: <BarChart3 size={14} /> },
            { key: "training", label: "Training History", icon: <Activity size={14} /> },
            { key: "datasets", label: "Datasets", icon: <Database size={14} /> },
          ] as const).map((tab) => (
            <button
              key={tab.key}
              onClick={() => setMetricsSubTab(tab.key)}
              style={{
                padding: "8px 18px",
                borderRadius: 20,
                border: metricsSubTab === tab.key ? "1px solid var(--accent-violet)" : "1px solid var(--border-subtle)",
                background: metricsSubTab === tab.key ? "rgba(139,92,246,0.12)" : "transparent",
                color: metricsSubTab === tab.key ? "var(--text-primary)" : "var(--text-muted)",
                fontSize: 13,
                fontWeight: 500,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 6,
                transition: "var(--transition)",
              }}
            >
              {tab.icon} {tab.label}
            </button>
          ))}
        </motion.div>

        {/* Overview */}
        {metricsSubTab === "overview" && (
          <>
            {/* KPI Cards */}
            <motion.div variants={fadeInUp} style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 16 }}>
              {[
                { label: "Accuracy", val: metrics.accuracy, color: "#10b981" },
                { label: "Precision", val: metrics.precision, color: "#8b5cf6" },
                { label: "Recall", val: metrics.recall, color: "#3b82f6" },
                { label: "F1 Score", val: metrics.f1_score, color: "#f59e0b" },
                { label: "AUC", val: metrics.auc_score, color: "#06b6d4" },
              ].map((m) => (
                <div key={m.label} className="glass" style={{ padding: 20, textAlign: "center" }}>
                  <p style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
                    {m.label}
                  </p>
                  <span style={{ fontSize: 28, fontWeight: 800, color: m.color }}>
                    {(m.val * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </motion.div>

            {/* ROC Curve + Confusion Matrix */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              {/* ROC */}
              <motion.div variants={fadeInUp} className="glass" style={{ padding: 24 }}>
                <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: "var(--text-secondary)" }}>ROC Curves</h4>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart>
                    <CartesianGrid stroke="rgba(255,255,255,0.04)" />
                    <XAxis
                      dataKey="fpr"
                      type="number"
                      domain={[0, 1]}
                      tick={{ fill: "#64748b", fontSize: 11 }}
                      label={{ value: "FPR", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }}
                    />
                    <YAxis
                      dataKey="tpr"
                      type="number"
                      domain={[0, 1]}
                      tick={{ fill: "#64748b", fontSize: 11 }}
                      label={{ value: "TPR", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
                    />
                    <Tooltip contentStyle={{ background: "#1e1e3a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, color: "#f1f5f9" }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    {Object.entries(metrics.roc_curve).map(([key, points], i) => (
                      <Line
                        key={key}
                        data={points}
                        dataKey="tpr"
                        name={key}
                        stroke={["#8b5cf6", "#3b82f6", "#06b6d4", "#10b981"][i]}
                        strokeWidth={2}
                        dot={false}
                        type="monotone"
                      />
                    ))}
                    {/* Diagonal */}
                    <Line
                      data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
                      dataKey="tpr"
                      name="Random"
                      stroke="rgba(255,255,255,0.1)"
                      strokeDasharray="5 5"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </motion.div>

              {/* Confusion Matrix */}
              <motion.div variants={fadeInUp} className="glass" style={{ padding: 24 }}>
                <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: "var(--text-secondary)" }}>Confusion Matrix</h4>
                <table className="matrix-table">
                  <thead>
                    <tr>
                      <th style={{ background: "transparent" }}>Actual \ Pred</th>
                      {confusionLabels.map((p) => (
                        <th key={p}>{p}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusionLabels.map((actual) => (
                      <tr key={actual}>
                        <td style={{ fontWeight: 600, color: "var(--text-secondary)", background: "rgba(139,92,246,0.06)" }}>{actual}</td>
                        {confusionLabels.map((pred) => {
                          const count = confusionGrid[actual][pred];
                          const isDiag = actual === pred;
                          return (
                            <td
                              key={pred}
                              style={{
                                fontWeight: isDiag ? 700 : 400,
                                color: isDiag ? "#10b981" : "var(--text-primary)",
                                background: isDiag ? "rgba(16,185,129,0.08)" : "transparent",
                                fontSize: 15,
                              }}
                            >
                              {count}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </motion.div>
            </div>
          </>
        )}

        {/* Training History */}
        {metricsSubTab === "training" && (
          <motion.div variants={fadeInUp} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            {Object.entries(metrics.training_history).map(([modality, epochs]) => (
              <div key={modality} className="glass" style={{ padding: 24 }}>
                <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, textTransform: "capitalize", color: "var(--text-secondary)" }}>
                  {modality} Model
                </h4>

                {/* Loss */}
                <p style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>Loss Curves</p>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={epochs}>
                    <CartesianGrid stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="epoch" tick={{ fill: "#64748b", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "#1e1e3a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, color: "#f1f5f9" }} />
                    <Line type="monotone" dataKey="train_loss" stroke="#f43f5e" strokeWidth={2} dot={false} name="Train Loss" />
                    <Line type="monotone" dataKey="val_loss" stroke="#f59e0b" strokeWidth={2} dot={false} name="Val Loss" />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                  </LineChart>
                </ResponsiveContainer>

                {/* Accuracy */}
                <p style={{ fontSize: 11, color: "var(--text-muted)", margin: "12px 0 8px" }}>Accuracy Curves</p>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={epochs}>
                    <CartesianGrid stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="epoch" tick={{ fill: "#64748b", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} domain={[0, 1]} />
                    <Tooltip contentStyle={{ background: "#1e1e3a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, color: "#f1f5f9" }} />
                    <Line type="monotone" dataKey="train_acc" stroke="#10b981" strokeWidth={2} dot={false} name="Train Acc" />
                    <Line type="monotone" dataKey="val_acc" stroke="#3b82f6" strokeWidth={2} dot={false} name="Val Acc" />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </motion.div>
        )}

        {/* Datasets */}
        {metricsSubTab === "datasets" && datasets.length > 0 && (
          <motion.div variants={fadeInUp} style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Dataset selector */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {datasets.map((ds) => (
                <button
                  key={ds.name}
                  onClick={() => setSelectedDataset(ds)}
                  style={{
                    padding: "8px 18px",
                    borderRadius: 20,
                    border: selectedDataset?.name === ds.name ? "1px solid var(--accent-violet)" : "1px solid var(--border-subtle)",
                    background: selectedDataset?.name === ds.name ? "rgba(139,92,246,0.12)" : "transparent",
                    color: selectedDataset?.name === ds.name ? "var(--text-primary)" : "var(--text-muted)",
                    fontSize: 13,
                    fontWeight: 500,
                    cursor: "pointer",
                    transition: "var(--transition)",
                  }}
                >
                  {ds.name}
                </button>
              ))}
            </div>

            {/* Selected dataset detail */}
            {selectedDataset && (
              <div className="glass" style={{ padding: 28 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 20 }}>
                  <div>
                    <h3 style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>{selectedDataset.name}</h3>
                    <p style={{ fontSize: 13, color: "var(--text-secondary)", maxWidth: 500 }}>{selectedDataset.description}</p>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <p style={{ fontSize: 11, color: "var(--text-muted)" }}>Samples</p>
                    <span style={{ fontSize: 24, fontWeight: 800, color: "var(--accent-violet)" }}>
                      {selectedDataset.size.toLocaleString()}
                    </span>
                    <span className={`metric-badge ${selectedDataset.category === "text" ? "badge-neutral" : "badge-positive"}`} style={{ display: "block", marginTop: 8 }}>
                      {selectedDataset.category}
                    </span>
                  </div>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                  {/* Distribution */}
                  <div>
                    <h4 style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: "var(--text-secondary)" }}>Class Distribution</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie
                          data={selectedDataset.class_distribution.map((c) => ({ name: c.label, value: c.count }))}
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={80}
                          dataKey="value"
                          stroke="none"
                          label={({ name, value }) => `${name}: ${value}`}
                          labelLine={{ stroke: "#64748b" }}
                        >
                          {selectedDataset.class_distribution.map((_, i) => (
                            <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip contentStyle={{ background: "#1e1e3a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, color: "#f1f5f9" }} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Sample Records */}
                  <div>
                    <h4 style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: "var(--text-secondary)" }}>Sample Records</h4>
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      {selectedDataset.sample_records.map((rec) => (
                        <div key={rec.id} style={{ padding: 14, background: "rgba(10,10,25,0.5)", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
                          <p style={{ fontSize: 13, color: "var(--text-primary)", marginBottom: 6 }}>
                            &ldquo;{rec.text_content}&rdquo;
                          </p>
                          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                            {sentimentBadge(rec.true_label)}
                            {rec.media_url && <span style={{ fontSize: 11, color: "var(--text-muted)" }}>📎 {rec.media_url}</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </motion.div>
    );
  };

  // ==========================================
  // MAIN LAYOUT
  // ==========================================
  return (
    <div style={{ display: "flex", minHeight: "100vh", position: "relative" }}>
      {/* Ambient Background */}
      <div className="ambient-bg" />

      {/* Sidebar */}
      <aside
        className="glass-static"
        style={{
          width: 260,
          padding: "28px 16px",
          display: "flex",
          flexDirection: "column",
          gap: 8,
          position: "fixed",
          top: 0,
          left: 0,
          bottom: 0,
          zIndex: 40,
          borderRadius: 0,
          borderRight: "1px solid var(--border-subtle)",
          borderTop: "none",
          borderBottom: "none",
          borderLeft: "none",
        }}
      >
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 12px", marginBottom: 28 }}>
          <div
            style={{
              width: 38,
              height: 38,
              borderRadius: "var(--radius-md)",
              background: "var(--gradient-main)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Brain size={20} color="white" />
          </div>
          <div>
            <h1 style={{ fontSize: 15, fontWeight: 700, lineHeight: 1.2 }}>SentiMind</h1>
            <p style={{ fontSize: 10, color: "var(--text-muted)", fontWeight: 500 }}>Multimodal Sentiment AI</p>
          </div>
        </div>

        {/* Nav Items */}
        <button
          className={`nav-tab ${activeTab === "analyze" ? "active" : ""}`}
          onClick={() => setActiveTab("analyze")}
        >
          <Zap size={17} /> Sentiment Analyzer
        </button>
        <button
          className={`nav-tab ${activeTab === "emotion" ? "active" : ""}`}
          onClick={() => setActiveTab("emotion")}
        >
          <Heart size={17} /> Emotion Detector
        </button>
        <button
          className={`nav-tab ${activeTab === "metrics" ? "active" : ""}`}
          onClick={() => setActiveTab("metrics")}
        >
          <TrendingUp size={17} /> Model Metrics
        </button>

        {/* Bottom Status */}
        <div style={{ marginTop: "auto", padding: "12px 16px", borderRadius: "var(--radius-md)", background: "rgba(16,185,129,0.06)", border: "1px solid rgba(16,185,129,0.12)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981" }} />
            <span style={{ fontSize: 12, color: "#34d399", fontWeight: 600 }}>Backend Connected</span>
          </div>
          <p style={{ fontSize: 11, color: "var(--text-muted)" }}>{API}</p>
        </div>
      </aside>

      {/* Main Content */}
      <main style={{ marginLeft: 260, flex: 1, padding: "32px 40px", position: "relative", zIndex: 10 }}>
        <AnimatePresence mode="wait">
          {activeTab === "analyze" && renderAnalyzeTab()}
          {activeTab === "emotion" && renderEmotionTab()}
          {activeTab === "metrics" && renderMetricsTab()}
        </AnimatePresence>
      </main>
    </div>
  );
}

const fs = require('fs');
const path = require('path');

// Ensure .env is loaded from backend-node/.env or root .env
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
require('dotenv').config({ path: path.join(__dirname, '..', '..', '.env') });

class OpenRouterService {
  constructor() {
    this.apiKey = (process.env.OPENROUTER_API_KEY || '').trim();
    this.baseUrl = (process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1/chat/completions').trim();
    this.model = (process.env.OPENROUTER_MODEL || 'nvidia/nemotron-3.5-lightning:free').trim();
    
    const fallbackStr = process.env.OPENROUTER_FALLBACK_MODELS || 
      'baidu/ernie-5.5-flash-preview:free,google/gemma-4-26b-a4b-it:free,google/gemma-4-31b-it:free,openrouter/free';
    this.fallbackModels = fallbackStr.split(',').map(m => m.trim()).filter(Boolean);

    this.siteUrl = process.env.OPENROUTER_SITE_URL || 'http://localhost:3000';
    this.appName = process.env.OPENROUTER_APP_NAME || 'Multimodal Sentiment Analysis Platform';
    this.timeout = parseFloat(process.env.OPENROUTER_TIMEOUT || '15.0') * 1000;
    this.lastError = null;
    this.lastModelUsed = this.model;
  }

  get isConfigured() {
    if (!this.apiKey) {
      this.apiKey = (process.env.OPENROUTER_API_KEY || '').trim();
    }
    return Boolean(this.apiKey);
  }

  async _callOpenRouter(messages, maxTokens = 250) {
    if (!this.isConfigured) {
      this.lastError = 'OpenRouter API key is missing. Set OPENROUTER_API_KEY in .env';
      return null;
    }

    const candidates = [this.model];
    for (const fb of this.fallbackModels) {
      if (!candidates.includes(fb)) candidates.push(fb);
    }

    for (const modelName of candidates.slice(0, 3)) {
      const payload = {
        model: modelName,
        messages: messages,
        temperature: 0.0,
        max_tokens: maxTokens,
        response_format: { type: 'json_object' }
      };

      // Suppress reasoning tokens for thinking models
      const lowerModel = modelName.toLowerCase();
      if (['nemotron', 'deepseek', 'reasoning', 'dots', 'think'].some(k => lowerModel.includes(k))) {
        payload.reasoning = { max_tokens: 0 };
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      try {
        const response = await fetch(this.baseUrl, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': this.siteUrl,
            'X-Title': this.appName,
            'User-Agent': 'SentimentAnalysisPlatform-Node/1.0'
          },
          body: JSON.stringify(payload),
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (response.ok) {
          const result = await response.json();
          const choices = result.choices || [];
          if (choices.length > 0) {
            const msg = choices[0].message || {};
            const content = msg.content || msg.reasoning || '';
            if (content) {
              this.lastModelUsed = modelName;
              this.lastError = null;
              return content;
            }
          }
        } else {
          const errText = await response.text();
          this.lastError = `HTTP ${response.status} from ${modelName}: ${errText}`;
          console.warn(`[OpenRouter Warning] ${modelName} returned HTTP ${response.status}`);
          continue;
        }
      } catch (err) {
        clearTimeout(timeoutId);
        this.lastError = `Exception with ${modelName}: ${err.message}`;
        console.warn(`[OpenRouter Warning] Error calling ${modelName}: ${err.message}`);
        continue;
      }
    }

    return null;
  }

  _extractJson(content) {
    if (!content) return null;
    const str = content.trim();

    // 1. Direct JSON parse
    try {
      return JSON.parse(str);
    } catch (_) {}

    // 2. Extract first '{' to last '}'
    const start = str.indexOf('{');
    const end = str.lastIndexOf('}');
    if (start !== -1 && end !== -1 && end > start) {
      try {
        return JSON.parse(str.substring(start, end + 1));
      } catch (_) {}
    }

    // 3. Remove markdown code blocks
    if (str.includes('```')) {
      let cleaned = str;
      if (cleaned.includes('```json')) cleaned = cleaned.split('```json')[1];
      else if (cleaned.includes('```')) cleaned = cleaned.split('```')[1];
      if (cleaned.includes('```')) cleaned = cleaned.split('```')[0];
      try {
        return JSON.parse(cleaned.trim());
      } catch (_) {}
    }

    // 4. Regex match outermost object
    const match = str.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch (_) {}
    }

    return null;
  }

  async predictSentiment(text, multimodalContext = null) {
    if (!text && !multimodalContext) return null;

    const systemPrompt = 
      'You are a high-speed sentiment JSON generator. You NEVER write thoughts, preambles, or explanations. ' +
      'You output ONLY valid raw JSON matching this schema: {"label": "Positive"|"Negative"|"Neutral", "confidence": float (0.0-1.0), ' +
      '"scores": {"positive": float, "negative": float, "neutral": float}}. Ensure positive + negative + neutral = 1.0.';

    let userContent = `Analyze: "${(text || '').trim()}"\nOutput JSON now (start directly with {):`;
    if (multimodalContext) {
      userContent += `\nContext: ${multimodalContext}`;
    }

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userContent }
    ];

    const content = await this._callOpenRouter(messages, 200);
    if (!content) return null;

    try {
      const data = this._extractJson(content);
      if (!data) return null;

      let label = (data.label || 'Neutral').trim();
      const validLabels = ['Positive', 'Negative', 'Neutral'];
      const matched = validLabels.find(l => l.toLowerCase() === label.toLowerCase());
      label = matched || 'Neutral';

      let scores = data.scores || {};
      let pos = parseFloat(scores.positive ?? 0.33);
      let neg = parseFloat(scores.negative ?? 0.33);
      let neu = parseFloat(scores.neutral ?? 0.34);

      const sumScores = pos + neg + neu;
      if (sumScores > 0) {
        pos = Math.round((pos / sumScores) * 100) / 100;
        neg = Math.round((neg / sumScores) * 100) / 100;
        neu = Math.max(0, Math.round((1.0 - pos - neg) * 100) / 100);
      }

      let confidence = parseFloat(data.confidence ?? (label === 'Positive' ? pos : label === 'Negative' ? neg : neu));
      confidence = Math.max(0.50, Math.min(0.99, Math.round(confidence * 100) / 100));

      console.log(`[OpenRouter Node] Sentiment: '${label}' (${(confidence * 100).toFixed(1)}%) | Model: ${this.lastModelUsed}`);
      return { label, confidence, scores: { positive: pos, negative: neg, neutral: neu } };
    } catch (err) {
      console.warn(`[OpenRouter Node] Parse sentiment error: ${err.message}`);
      return null;
    }
  }

  async predictEmotion(text) {
    if (!text) return null;

    const validEmotions = ['Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral'];
    const systemPrompt = 
      'You are a high-speed emotion JSON generator. You NEVER write thoughts, preambles, or explanations. ' +
      'You output ONLY valid raw JSON matching this schema: {"emotion": "<Category>", "confidence": float (0.0-1.0), ' +
      '"probabilities": {"Happiness": float, "Sadness": float, "Anger": float, "Fear": float, "Surprise": float, "Disgust": float, "Neutral": float}}. ' +
      'Categories are strictly one of: Happiness, Sadness, Anger, Fear, Surprise, Disgust, Neutral. Ensure probabilities sum = 1.0.';

    const userContent = `Analyze: "${text.trim()}"\nOutput JSON now (start directly with {):`;

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userContent }
    ];

    const content = await this._callOpenRouter(messages, 200);
    if (!content) return null;

    try {
      const data = this._extractJson(content);
      let emotion = 'Neutral';
      let confidence = 0.95;
      let probs = {};

      if (data) {
        const rawEmot = String(data.emotion || '').trim();
        const matched = validEmotions.find(e => e.toLowerCase() === rawEmot.toLowerCase());
        emotion = matched || 'Neutral';

        const rawProbs = data.probabilities || {};
        for (const ve of validEmotions) {
          probs[ve] = parseFloat(rawProbs[ve] ?? rawProbs[ve.toLowerCase()] ?? 0.05);
        }

        const sumP = Object.values(probs).reduce((a, b) => a + b, 0);
        if (sumP > 0) {
          for (const k of Object.keys(probs)) {
            probs[k] = Math.round((probs[k] / sumP) * 10000) / 10000;
          }
        }
        confidence = parseFloat(data.confidence ?? probs[emotion] ?? 0.95);
        confidence = Math.max(0.50, Math.min(0.999, Math.round(confidence * 10000) / 10000));
      } else {
        // Fallback regex recovery
        const match = content.match(/"emotion":\s*"([A-Za-z]+)"/i);
        if (match) {
          const cand = match[1];
          const matched = validEmotions.find(e => e.toLowerCase() === cand.toLowerCase());
          emotion = matched || 'Neutral';
        }
        for (const ve of validEmotions) probs[ve] = 0.01;
        probs[emotion] = 0.94;
        confidence = 0.94;
      }

      console.log(`[OpenRouter Node] Emotion: '${emotion}' (${(confidence * 100).toFixed(1)}%) | Model: ${this.lastModelUsed}`);
      return { emotion, confidence, probabilities: probs };
    } catch (err) {
      console.warn(`[OpenRouter Node] Parse emotion error: ${err.message}`);
      return null;
    }
  }
}

module.exports = new OpenRouterService();

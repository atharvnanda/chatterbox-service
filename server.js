/**
 * Node.js + Express — API Gateway
 *
 * Responsibilities:
 *   1. Serve the static frontend (public/)
 *   2. Accept POST /api/synthesise from the browser
 *   3. Proxy the request to the Python FastAPI microservice
 *   4. Stream the WAV audio response straight back to the browser
 */

const express = require("express");
const axios = require("axios");
const path = require("path");

const app = express();

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PORT = process.env.PORT || 3000;
const ML_SERVICE_URL =
  process.env.ML_SERVICE_URL || "http://localhost:8000/synthesise";

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------
app.get("/health", (_req, res) => res.json({ status: "ok" }));

// ---------------------------------------------------------------------------
// Main proxy endpoint
// ---------------------------------------------------------------------------
app.post("/api/synthesise", async (req, res) => {
  const { text, audio_prompt_path, language_id, max_chars, exaggeration, cfg_weight, temperature } =
    req.body;

  if (!text || !text.trim()) {
    return res.status(400).json({ error: "text field is required and must not be empty." });
  }

  // Build the payload for the ML service (only forward non-undefined fields)
  const mlPayload = { text: text.trim() };
  if (audio_prompt_path !== undefined) mlPayload.audio_prompt_path = audio_prompt_path;
  if (language_id !== undefined)       mlPayload.language_id       = language_id;
  if (max_chars !== undefined)         mlPayload.max_chars          = max_chars;
  if (exaggeration !== undefined)      mlPayload.exaggeration       = exaggeration;
  if (cfg_weight !== undefined)        mlPayload.cfg_weight         = cfg_weight;
  if (temperature !== undefined)       mlPayload.temperature        = temperature;

  console.log(`[Gateway] → ML service  text_len=${text.length}`);

  try {
    const mlResponse = await axios({
      method: "POST",
      url: ML_SERVICE_URL,
      data: mlPayload,
      responseType: "stream",        // <-- key: don't buffer the WAV in Node
      timeout: 5 * 60 * 1000,       // 5-minute timeout for long texts
    });

    // Forward headers so the browser knows it's receiving audio
    res.setHeader("Content-Type", mlResponse.headers["content-type"] || "audio/wav");
    res.setHeader("Content-Disposition", 'inline; filename="synthesis.wav"');

    // Pipe the stream directly — no intermediate buffering
    mlResponse.data.pipe(res);

    mlResponse.data.on("end", () => {
      console.log("[Gateway] Stream complete.");
    });

    mlResponse.data.on("error", (streamErr) => {
      console.error("[Gateway] Stream error:", streamErr.message);
      if (!res.headersSent) {
        res.status(500).json({ error: "Audio stream error." });
      }
    });
  } catch (err) {
    console.error("[Gateway] ML service error:", err.message);

    if (err.response) {
      // The ML service returned an HTTP error — forward it
      const status = err.response.status || 502;

      // err.response.data may be a stream when responseType='stream'
      if (err.response.data && typeof err.response.data.on === "function") {
        let body = "";
        err.response.data.on("data", (chunk) => (body += chunk));
        err.response.data.on("end", () => {
          try {
            const parsed = JSON.parse(body);
            res.status(status).json({ error: parsed.detail || body });
          } catch {
            res.status(status).json({ error: body || "ML service error" });
          }
        });
      } else {
        res.status(status).json({ error: "ML service returned an error." });
      }
    } else if (err.code === "ECONNREFUSED") {
      res
        .status(503)
        .json({ error: "ML service is unavailable. Is the Python server running?" });
    } else {
      res.status(500).json({ error: err.message });
    }
  }
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
app.listen(PORT, () => {
  console.log(`[Gateway] Listening on http://localhost:${PORT}`);
  console.log(`[Gateway] ML service URL: ${ML_SERVICE_URL}`);
});

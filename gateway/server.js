const express = require("express");
const axios = require("axios");
const path = require("path");

const app = express();
const ML_URL = process.env.ML_SERVICE_URL || "http://localhost:8000/synthesise";

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.get("/health", (_req, res) => res.json({ status: "ok" }));

app.post("/api/synthesise", async (req, res) => {
  if (!req.body.text?.trim()) {
    return res.status(400).json({ error: "text is required" });
  }

  try {
    const mlRes = await axios.post(ML_URL, req.body, {
      responseType: "stream",
      timeout: 5 * 60 * 1000,
    });
    res.setHeader("Content-Type", "audio/wav");
    mlRes.data.pipe(res);
  } catch (err) {
    const status = err.response?.status || 502;
    res.status(status).json({ error: err.message });
  }
});

app.listen(3000, () => console.log("Gateway on http://localhost:3000"));
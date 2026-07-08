// This is the whole backend. It has ONE job:
// receive an image from the iPhone app, send it to Sightengine,
// and reply with a simple yes/no-ish answer.

require('dotenv').config();
const express = require('express');
const multer = require('multer');
const fetch = require('node-fetch');
const FormData = require('form-data');

const app = express();

// This handles receiving an uploaded image file from the app.
// 10MB limit is plenty for a single screen frame.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
});

// These come from environment variables (set in a .env file locally,
// or in Render's dashboard when deployed) — never hard-coded here.
const SIGHTENGINE_API_USER = process.env.SIGHTENGINE_API_USER;
const SIGHTENGINE_API_SECRET = process.env.SIGHTENGINE_API_SECRET;

// A simple "is it alive" check — visiting this URL in a browser
// should just say the server is running.
app.get('/', (req, res) => {
  res.send('AI Screen Detector backend is running.');
});

// The one real endpoint. The app will POST an image here.
app.post('/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No image received. Send it as form-data under the field name "image".',
      });
    }

    if (!SIGHTENGINE_API_USER || !SIGHTENGINE_API_SECRET) {
      return res.status(500).json({
        error: 'Server is missing its Sightengine credentials (check environment variables).',
      });
    }

    // Build the request Sightengine expects: the image file, which
    // model to run ("genai" = AI-generated image detection), and your credentials.
    const form = new FormData();
    form.append('media', req.file.buffer, { filename: 'frame.jpg' });
    form.append('models', 'genai');
    form.append('api_user', SIGHTENGINE_API_USER);
    form.append('api_secret', SIGHTENGINE_API_SECRET);

    const sightengineResponse = await fetch('https://api.sightengine.com/1.0/check.json', {
      method: 'POST',
      body: form,
    });

    const data = await sightengineResponse.json();

    if (data.status !== 'success') {
      return res.status(502).json({ error: 'Sightengine returned an error', details: data });
    }

    // Sightengine gives a score from 0 to 1: how confident it is
    // the image is AI-generated. We turn that into a simple label.
    const aiScore = data.type && typeof data.type.ai_generated === 'number'
      ? data.type.ai_generated
      : null;

    let verdict = 'uncertain';
    if (aiScore !== null) {
      if (aiScore >= 0.7) verdict = 'likely_ai';
      else if (aiScore <= 0.3) verdict = 'likely_real';
    }

    res.json({
      verdict,          // "likely_ai" | "likely_real" | "uncertain"
      confidence: aiScore, // raw 0-1 score from Sightengine
      source: 'sightengine',
    });
  } catch (err) {
    console.error('Error in /analyze:', err);
    res.status(500).json({ error: 'Something went wrong analyzing the image.' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});

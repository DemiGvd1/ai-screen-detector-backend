// This is the whole backend. It has ONE job:
// receive a photo, video, or audio clip from the iPhone app, send it
// to Sightengine, and reply with a simple verdict + confidence.

require('dotenv').config();
const express = require('express');
const multer = require('multer');
const fetch = require('node-fetch');
const FormData = require('form-data');
const { pool, initDB } = require('./db');

const app = express();
app.use(express.json());

initDB().catch((err) => console.error('Database setup failed:', err));

const ADMIN_SECRET = process.env.ADMIN_SECRET;

// Only requests that include the correct secret get through.
// This is your "admin only" gate — no accounts, no login screen,
// just a password only you know, sent by whatever tool you use to post.
function requireAdmin(req, res, next) {
  const providedSecret = req.headers['x-admin-secret'];
  if (!ADMIN_SECRET || providedSecret !== ADMIN_SECRET) {
    return res.status(401).json({ error: 'Unauthorized: missing or incorrect admin secret.' });
  }
  next();
}

// Photos are small. Videos and audio clips are bigger, so they get
// their own upload limit.
const uploadImage = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
});
const uploadMedia = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
});

const SIGHTENGINE_API_USER = process.env.SIGHTENGINE_API_USER;
const SIGHTENGINE_API_SECRET = process.env.SIGHTENGINE_API_SECRET;

app.get('/', (req, res) => {
  res.send('AI Screen Detector backend is running.');
});

// Turns a raw 0-1 "how AI does this look" score into a verdict, and a
// confidence number that reads HIGH whenever we're sure — whether
// we're sure it's AI, or sure it's real. This is what fixes the
// "likely real, 0% confidence" confusion.
const MAX_CONFIDENCE = 0.9; // never claim more than 90% certainty, either direction

function scoreToVerdict(score) {
  if (score === null || score === undefined) {
    return { verdict: 'uncertain', confidence: null };
  }
  if (score >= 0.6) {
    return { verdict: 'likely_ai', confidence: Math.min(score, MAX_CONFIDENCE) };
  }
  if (score <= 0.4) {
    return { verdict: 'likely_real', confidence: Math.min(1 - score, MAX_CONFIDENCE) };
  }
  return { verdict: 'uncertain', confidence: Math.min(score, MAX_CONFIDENCE) };
}

// Given several detector scores, picks whichever one is most confident.
function pickTopScore(entries) {
  const valid = entries.filter((e) => typeof e.value === 'number');
  if (valid.length === 0) return null;
  return valid.reduce((a, b) => (b.value > a.value ? b : a));
}

// ---------- PHOTOS ----------
app.post('/analyze', uploadImage.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No image received. Send it as form-data under the field name "image".',
      });
    }
    if (!SIGHTENGINE_API_USER || !SIGHTENGINE_API_SECRET) {
      return res.status(500).json({ error: 'Server is missing its Sightengine credentials.' });
    }

    const form = new FormData();
    form.append('media', req.file.buffer, { filename: 'photo.jpg' });
    form.append('models', 'genai,deepfake');
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

    const aiScore = data.type && typeof data.type.ai_generated === 'number' ? data.type.ai_generated : null;
    const deepfakeScore = data.deepfake && typeof data.deepfake.score === 'number' ? data.deepfake.score : null;

    const top = pickTopScore([
      { label: 'ai_generated', value: aiScore },
      { label: 'deepfake', value: deepfakeScore },
    ]);

    const { verdict, confidence } = scoreToVerdict(top ? top.value : null);

    res.json({
      verdict,
      confidence,
      flagged_by: top ? top.label : null,
      scores: { ai_generated: aiScore, deepfake: deepfakeScore },
      media_type: 'image',
      source: 'sightengine',
    });
  } catch (err) {
    console.error('Error in /analyze:', err);
    res.status(500).json({ error: 'Something went wrong analyzing the image.' });
  }
});

// ---------- VIDEO ----------
app.post('/analyze-video', uploadMedia.single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No video received. Send it as form-data under the field name "video".',
      });
    }
    if (!SIGHTENGINE_API_USER || !SIGHTENGINE_API_SECRET) {
      return res.status(500).json({ error: 'Server is missing its Sightengine credentials.' });
    }

    const form = new FormData();
    form.append('media', req.file.buffer, { filename: 'clip.mp4' });
    form.append('models', 'genai,deepfake');
    form.append('api_user', SIGHTENGINE_API_USER);
    form.append('api_secret', SIGHTENGINE_API_SECRET);

    const sightengineResponse = await fetch('https://api.sightengine.com/1.0/video/check-sync.json', {
      method: 'POST',
      body: form,
    });
    const data = await sightengineResponse.json();

    if (data.status !== 'success') {
      return res.status(502).json({ error: 'Sightengine returned an error', details: data });
    }

    // Video results come back frame-by-frame. We take the single
    // highest score seen across all frames as the overall verdict.
    const frames = (data.data && data.data.frames) || [];
    let aiScore = null;
    let deepfakeScore = null;

    for (const frame of frames) {
      const frameAi = frame.type && frame.type.ai_generated;
      const frameDeepfake = frame.deepfake && frame.deepfake.score;
      if (typeof frameAi === 'number' && (aiScore === null || frameAi > aiScore)) aiScore = frameAi;
      if (typeof frameDeepfake === 'number' && (deepfakeScore === null || frameDeepfake > deepfakeScore)) deepfakeScore = frameDeepfake;
    }

    // Fallback in case Sightengine returns a single top-level score instead.
    if (aiScore === null && data.type) aiScore = data.type.ai_generated ?? null;
    if (deepfakeScore === null && data.deepfake) deepfakeScore = data.deepfake.score ?? null;

    const top = pickTopScore([
      { label: 'ai_generated', value: aiScore },
      { label: 'deepfake', value: deepfakeScore },
    ]);

    const { verdict, confidence } = scoreToVerdict(top ? top.value : null);

    res.json({
      verdict,
      confidence,
      flagged_by: top ? top.label : null,
      scores: { ai_generated: aiScore, deepfake: deepfakeScore },
      media_type: 'video',
      source: 'sightengine',
    });
  } catch (err) {
    console.error('Error in /analyze-video:', err);
    res.status(500).json({ error: 'Something went wrong analyzing the video.' });
  }
});

// ---------- AUDIO ----------
app.post('/analyze-audio', uploadMedia.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No audio received. Send it as form-data under the field name "audio".',
      });
    }
    if (!SIGHTENGINE_API_USER || !SIGHTENGINE_API_SECRET) {
      return res.status(500).json({ error: 'Server is missing its Sightengine credentials.' });
    }

    const form = new FormData();
    form.append('audio', req.file.buffer, { filename: 'clip.mp3' });
    // genai catches AI-generated music, ai_speech catches AI-generated voice.
    form.append('models', 'genai,ai_speech');
    form.append('api_user', SIGHTENGINE_API_USER);
    form.append('api_secret', SIGHTENGINE_API_SECRET);

    const sightengineResponse = await fetch('https://api.sightengine.com/1.0/audio/check.json', {
      method: 'POST',
      body: form,
    });
    const data = await sightengineResponse.json();

    if (data.status !== 'success') {
      return res.status(502).json({ error: 'Sightengine returned an error', details: data });
    }

    const musicScore = data.type && typeof data.type.ai_generated === 'number' ? data.type.ai_generated : null;
    const speechScore = data.ai_speech && typeof data.ai_speech.score === 'number' ? data.ai_speech.score : null;

    const top = pickTopScore([
      { label: 'ai_music', value: musicScore },
      { label: 'ai_speech', value: speechScore },
    ]);

    const { verdict, confidence } = scoreToVerdict(top ? top.value : null);

    res.json({
      verdict,
      confidence,
      flagged_by: top ? top.label : null,
      scores: { ai_music: musicScore, ai_speech: speechScore },
      media_type: 'audio',
      source: 'sightengine',
    });
  } catch (err) {
    console.error('Error in /analyze-audio:', err);
    res.status(500).json({ error: 'Something went wrong analyzing the audio.' });
  }
});

// ---------- TRENDING FEED ----------

// Public — the app calls this to load the scrollable feed.
app.get('/trending', async (req, res) => {
  try {
    const result = await pool.query(
      'SELECT * FROM trending_posts ORDER BY created_at DESC LIMIT 50'
    );
    res.json(result.rows);
  } catch (err) {
    console.error('Error in /trending:', err);
    res.status(500).json({ error: 'Could not load trending posts.' });
  }
});

// Admin only — adds a new post to the feed.
app.post('/admin/trending', requireAdmin, async (req, res) => {
  const { title, video_url, thumbnail_url, ai_score, category, source_platform } = req.body;
  if (!title || !video_url) {
    return res.status(400).json({ error: 'title and video_url are required.' });
  }
  try {
    const result = await pool.query(
      `INSERT INTO trending_posts (title, video_url, thumbnail_url, ai_score, category, source_platform)
       VALUES ($1, $2, $3, $4, $5, $6) RETURNING *`,
      [title, video_url, thumbnail_url || null, ai_score || null, category || null, source_platform || null]
    );
    res.json(result.rows[0]);
  } catch (err) {
    console.error('Error in /admin/trending:', err);
    res.status(500).json({ error: 'Could not add post.' });
  }
});

// Admin only — removes a post from the feed.
app.delete('/admin/trending/:id', requireAdmin, async (req, res) => {
  try {
    await pool.query('DELETE FROM trending_posts WHERE id = $1', [req.params.id]);
    res.json({ success: true });
  } catch (err) {
    console.error('Error in /admin/trending delete:', err);
    res.status(500).json({ error: 'Could not delete post.' });
  }
});

// Admin only — given a TikTok link, fetches its official thumbnail
// and title using TikTok's public oEmbed feature (the same thing
// used for legitimate link previews, not a workaround).
app.get('/admin/fetch-preview', requireAdmin, async (req, res) => {
  const videoUrl = req.query.url;
  if (!videoUrl) {
    return res.status(400).json({ error: 'Provide ?url=... pointing to a TikTok video.' });
  }
  try {
    const oembedUrl = `https://www.tiktok.com/oembed?url=${encodeURIComponent(videoUrl)}`;
    const response = await fetch(oembedUrl);
    if (!response.ok) {
      return res.status(502).json({ error: 'TikTok did not recognize that link.' });
    }
    const data = await response.json();
    res.json({
      title: data.title || null,
      thumbnail_url: data.thumbnail_url || null,
      author_name: data.author_name || null,
    });
  } catch (err) {
    console.error('Error in /admin/fetch-preview:', err);
    res.status(500).json({ error: 'Could not fetch preview.' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});

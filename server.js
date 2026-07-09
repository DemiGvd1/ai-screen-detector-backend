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
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY;
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');

// Sends text to a free, open-source AI-text-detector model hosted on
// Hugging Face, and returns a 0-1 "probability this is AI-written" score.
async function detectAIText(text) {
  const response = await fetch(
    'https://api-inference.huggingface.co/models/fakespot-ai/roberta-base-ai-text-detection-v1',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${HUGGINGFACE_API_KEY}`,
      },
      body: JSON.stringify({ inputs: text }),
    }
  );

  const data = await response.json();

  // Free-tier models sometimes need a few seconds to "wake up" the
  // first time they're called. If that happens, wait and try once more.
  if (data.error && data.estimated_time) {
    await new Promise((resolve) => setTimeout(resolve, Math.min(data.estimated_time * 1000, 15000)));
    return detectAIText(text);
  }

  // The response is a list of {label, score} pairs. We look for
  // whichever label clearly means "AI-generated" or "human", since
  // exact label wording can vary between models.
  const results = Array.isArray(data[0]) ? data[0] : data;
  if (!Array.isArray(results)) return null;

  const aiEntry = results.find((r) => /ai|generated|fake|machine/i.test(r.label));
  if (aiEntry) return aiEntry.score;

  const humanEntry = results.find((r) => /human|real/i.test(r.label));
  if (humanEntry) return 1 - humanEntry.score;

  return results[0] ? results[0].score : null;
}

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

// ---------- TEXT ----------
app.post('/analyze-text', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || text.trim().length < 50) {
      return res.status(400).json({ error: 'Please provide at least a few sentences (50+ characters) to analyze.' });
    }
    if (!HUGGINGFACE_API_KEY) {
      return res.status(500).json({ error: 'Server is missing its GPTZero credentials.' });
    }

    const score = await detectAIText(text);
    const { verdict, confidence } = scoreToVerdict(score);

    res.json({
      verdict,
      confidence,
      media_type: 'text',
      source: 'huggingface',
    });
  } catch (err) {
    console.error('Error in /analyze-text:', err);
    res.status(500).json({ error: 'Something went wrong analyzing the text.' });
  }
});

// ---------- DOCUMENTS (PDF / Word) ----------
app.post('/analyze-document', uploadMedia.single('document'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No document received. Send it as form-data under the field name "document".' });
    }
    if (!HUGGINGFACE_API_KEY) {
      return res.status(500).json({ error: 'Server is missing its GPTZero credentials.' });
    }

    const filename = (req.file.originalname || '').toLowerCase();
    let extractedText = '';

    if (filename.endsWith('.pdf')) {
      const parsed = await pdfParse(req.file.buffer);
      extractedText = parsed.text;
    } else if (filename.endsWith('.docx')) {
      const result = await mammoth.extractRawText({ buffer: req.file.buffer });
      extractedText = result.value;
    } else {
      extractedText = req.file.buffer.toString('utf-8');
    }

    if (!extractedText || extractedText.trim().length < 50) {
      return res.status(400).json({ error: 'Could not find enough readable text in that document.' });
    }

    // GPTZero has a length limit, so trim very long documents.
    const trimmedText = extractedText.slice(0, 5000);
    const score = await detectAIText(trimmedText);
    const { verdict, confidence } = scoreToVerdict(score);

    res.json({
      verdict,
      confidence,
      media_type: 'document',
      source: 'huggingface',
    });
  } catch (err) {
    console.error('Error in /analyze-document:', err);
    res.status(500).json({ error: 'Something went wrong analyzing the document.' });
  }
});

// ---------- LINK (TikTok / Instagram paste-a-link) ----------
const { execFile } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

// Only these platforms are allowed — this stops people from pasting
// random/unknown links, which could otherwise let someone misuse your
// server to fetch arbitrary web addresses.
const ALLOWED_LINK_DOMAINS = [
  'tiktok.com',
  'vt.tiktok.com',
  'vm.tiktok.com',
  'instagram.com',
  'facebook.com',
  'fb.watch',
  'twitter.com',
  'x.com',
];

function isAllowedSocialLink(rawUrl) {
  try {
    const hostname = new URL(rawUrl).hostname.toLowerCase();
    return ALLOWED_LINK_DOMAINS.some((domain) => hostname === domain || hostname.endsWith(`.${domain}`));
  } catch {
    return false;
  }
}

app.post('/analyze-link', async (req, res) => {
  const { url } = req.body;
  if (!url) {
    return res.status(400).json({ error: 'Provide a "url" to analyze.' });
  }
  if (!isAllowedSocialLink(url)) {
    return res.status(400).json({
      error: 'Please paste a link from TikTok, Instagram, Facebook, or Twitter/X.',
    });
  }
  if (!SIGHTENGINE_API_USER || !SIGHTENGINE_API_SECRET) {
    return res.status(500).json({ error: 'Server is missing its Sightengine credentials.' });
  }

  const tempPath = path.join(os.tmpdir(), `link-${Date.now()}.mp4`);

  execFile('./yt-dlp', ['-f', 'best[ext=mp4]/best', '-o', tempPath, url], { timeout: 60000 }, async (err) => {
    if (err) {
      console.error('yt-dlp error:', err);
      return res.status(502).json({ error: 'Could not download that link. It may be private, region-locked, or an unsupported platform.' });
    }

    try {
      const videoBuffer = fs.readFileSync(tempPath);

      const form = new FormData();
      form.append('media', videoBuffer, { filename: 'link-video.mp4' });
      form.append('models', 'genai,deepfake');
      form.append('api_user', SIGHTENGINE_API_USER);
      form.append('api_secret', SIGHTENGINE_API_SECRET);

      const sightengineResponse = await fetch('https://api.sightengine.com/1.0/video/check-sync.json', {
        method: 'POST',
        body: form,
      });
      const data = await sightengineResponse.json();

      fs.unlink(tempPath, () => { }); // clean up the temp file either way

      if (data.status !== 'success') {
        return res.status(502).json({ error: 'Sightengine returned an error', details: data });
      }

      const frames = (data.data && data.data.frames) || [];
      let aiScore = null;
      let deepfakeScore = null;
      for (const frame of frames) {
        const frameAi = frame.type && frame.type.ai_generated;
        const frameDeepfake = frame.deepfake && frame.deepfake.score;
        if (typeof frameAi === 'number' && (aiScore === null || frameAi > aiScore)) aiScore = frameAi;
        if (typeof frameDeepfake === 'number' && (deepfakeScore === null || frameDeepfake > deepfakeScore)) deepfakeScore = frameDeepfake;
      }

      const top = pickTopScore([
        { label: 'ai_generated', value: aiScore },
        { label: 'deepfake', value: deepfakeScore },
      ]);
      const { verdict, confidence } = scoreToVerdict(top ? top.value : null);

      res.json({
        verdict,
        confidence,
        flagged_by: top ? top.label : null,
        media_type: 'video',
        source: 'sightengine',
        from_link: true,
      });
    } catch (processErr) {
      console.error('Error processing downloaded video:', processErr);
      fs.unlink(tempPath, () => { });
      res.status(500).json({ error: 'Downloaded the video but could not analyze it.' });
    }
  });
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
  const rawUrl = req.query.url;
  if (!rawUrl) {
    return res.status(400).json({ error: 'Provide ?url=... pointing to a TikTok video.' });
  }
  try {
    // TikTok's "Copy Link" button gives short links like vt.tiktok.com/xxx,
    // which oEmbed can't read directly. Follow the redirect first to get
    // the full, expanded link (tiktok.com/@user/video/123...).
    let resolvedUrl = rawUrl;
    try {
      const redirectCheck = await fetch(rawUrl, { method: 'GET', redirect: 'follow' });
      if (redirectCheck.url) resolvedUrl = redirectCheck.url;
    } catch (redirectErr) {
      console.error('Redirect resolution failed:', redirectErr.message);
    }

    const oembedUrl = `https://www.tiktok.com/oembed?url=${encodeURIComponent(resolvedUrl)}`;
    const response = await fetch(oembedUrl);
    if (!response.ok) {
      return res.status(502).json({ error: 'TikTok did not recognize that link.', resolved_url: resolvedUrl });
    }
    const data = await response.json();
    res.json({
      title: data.title || null,
      thumbnail_url: data.thumbnail_url || null,
      author_name: data.author_name || null,
      resolved_url: resolvedUrl,
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
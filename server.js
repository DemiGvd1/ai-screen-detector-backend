// This is the whole backend. It has ONE job:
// receive a photo, video, or audio clip from the iPhone app, send it
// to Sightengine, and reply with a simple verdict + confidence.

require('dotenv').config();
const express = require('express');
const multer = require('multer');
const fetch = require('node-fetch');
const FormData = require('form-data');
const rateLimit = require('express-rate-limit');
const { pool, initDB } = require('./db');

const app = express();
app.use(express.json());

initDB().catch((err) => console.error('Database setup failed:', err));

// Scans hit paid/rate-limited third-party APIs (Sightengine, Hugging Face)
// and, for video/link, spawn ffmpeg/yt-dlp — cap how often one client can
// trigger them so a single caller can't burn through quota or CPU.
const scanLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many scans from this device. Please wait a few minutes and try again.' },
});

// Lighter cap for cheap, frequently-polled public routes.
const publicLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 300,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests. Please try again shortly.' },
});

const ADMIN_SECRET = process.env.ADMIN_SECRET;
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY;
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const ffmpegPath = require('ffmpeg-static');
const { execFile } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

// All visual content (video uploads AND links) goes through Sightengine's
// IMAGE detector, not their video detector — this keeps everything on
// one provider (Sightengine) and avoids their paid-only video API.
// We pull up to 5 still frames out of the video ourselves using ffmpeg
// (a free tool), then run each frame through the same image detector
// used by Photo scan, taking the highest score seen across frames.
async function analyzeVideoFrames(videoBuffer) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'frames-'));
  const videoPath = path.join(tempDir, 'input.mp4');
  fs.writeFileSync(videoPath, videoBuffer);
  const framePattern = path.join(tempDir, 'frame-%02d.jpg');

  try {
    await new Promise((resolve, reject) => {
      execFile(
        ffmpegPath,
        ['-i', videoPath, '-vf', 'fps=1/2,scale=480:-1', '-vframes', '5', framePattern],
        (err) => (err ? reject(err) : resolve())
      );
    });

    const frameFiles = fs.readdirSync(tempDir).filter((f) => f.startsWith('frame-'));
    let aiScore = null;
    let deepfakeScore = null;

    // Grab the first frame as a reusable thumbnail (base64) so callers
    // like Link scan can save it to history without a second download —
    // and also caption it, same as Photo scan, so video results can say
    // what's actually in the clip instead of just a verdict.
    let thumbnailBase64 = null;
    let firstFrameBuffer = null;
    if (frameFiles.length > 0) {
      firstFrameBuffer = fs.readFileSync(path.join(tempDir, frameFiles[0]));
      thumbnailBase64 = firstFrameBuffer.toString('base64');
    }

    // Check all frames at the same time instead of one after another —
    // this is the main thing that was making video scans slow. Captioning
    // the first frame happens alongside it, not after, so it doesn't add
    // extra wait time.
    const [frameResults, caption] = await Promise.all([
      Promise.all(
        frameFiles.map(async (frameFile) => {
          const frameBuffer = fs.readFileSync(path.join(tempDir, frameFile));
          const form = new FormData();
          form.append('media', frameBuffer, { filename: frameFile });
          form.append('models', 'genai,deepfake');
          form.append('api_user', SIGHTENGINE_API_USER);
          form.append('api_secret', SIGHTENGINE_API_SECRET);

          const response = await fetch('https://api.sightengine.com/1.0/check.json', {
            method: 'POST',
            body: form,
          });
          const data = await response.json();

          if (data.status !== 'success') {
            console.error('Sightengine rejected a frame:', data);
            return { ai: null, deepfake: null };
          }

          return {
            ai: data.type && typeof data.type.ai_generated === 'number' ? data.type.ai_generated : null,
            deepfake: data.deepfake && typeof data.deepfake.score === 'number' ? data.deepfake.score : null,
          };
        })
      ),
      firstFrameBuffer ? generateImageCaption(firstFrameBuffer) : Promise.resolve(null),
    ]);

    for (const result of frameResults) {
      if (result.ai !== null && (aiScore === null || result.ai > aiScore)) aiScore = result.ai;
      if (result.deepfake !== null && (deepfakeScore === null || result.deepfake > deepfakeScore)) deepfakeScore = result.deepfake;
    }

    return { aiScore, deepfakeScore, frameCount: frameFiles.length, thumbnailBase64, caption };
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

// Sends text to a free, open-source AI-text-detector model hosted on
// Hugging Face. Tries a primary model first, and automatically falls
// back to a second model if the first one is unavailable, instead of
// failing outright.
const TEXT_DETECTION_MODELS = [
  'fakespot-ai/roberta-base-ai-text-detection-v1',
  'openai-community/roberta-base-openai-detector',
];

async function detectAIText(text, modelIndex = 0, attempt = 1) {
  if (modelIndex >= TEXT_DETECTION_MODELS.length) {
    throw new Error('All text detection models are currently unavailable.');
  }
  const model = TEXT_DETECTION_MODELS[modelIndex];

  try {
    const response = await fetch(`https://router.huggingface.co/hf-inference/models/${model}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${HUGGINGFACE_API_KEY}`,
      },
      body: JSON.stringify({ inputs: text }),
    });

    const data = await response.json();

    if (data.error && data.estimated_time) {
      await new Promise((resolve) => setTimeout(resolve, Math.min(data.estimated_time * 1000, 15000)));
      return detectAIText(text, modelIndex, attempt);
    }

    // If this model errored out for any other reason, try the next model.
    if (data.error) {
      console.error(`Text model ${model} failed:`, data.error);
      return detectAIText(text, modelIndex + 1, 1);
    }

    const results = Array.isArray(data[0]) ? data[0] : data;
    if (!Array.isArray(results)) return detectAIText(text, modelIndex + 1, 1);

    const aiEntry = results.find((r) => /ai|generated|fake|machine/i.test(r.label));
    if (aiEntry) return aiEntry.score;

    const humanEntry = results.find((r) => /human|real/i.test(r.label));
    if (humanEntry) return 1 - humanEntry.score;

    return results[0] ? results[0].score : null;
  } catch (err) {
    if (attempt < 2) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      return detectAIText(text, modelIndex, attempt + 1);
    }
    // This model is fully down — move to the next one.
    return detectAIText(text, modelIndex + 1, 1);
  }
}

function reasonForTextVerdict(verdict, confidence) {
  if (verdict === 'likely_ai') {
    return `Traced as AI-written. The style closely matches patterns common in AI-generated text (${Math.round(confidence * 100)}% match).`;
  }
  if (verdict === 'likely_real') {
    return `Traced as human-written. The style matches typical human writing patterns (${Math.round(confidence * 100)}% confidence).`;
  }
  return "Signals were mixed — Trace isn't confident enough to call this either way.";
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

// Same gate, but for the dashboard's initial page load — browsers can't
// attach a custom x-admin-secret header to a plain navigation, so this
// uses HTTP Basic Auth instead (the browser prompts for it natively).
// Username is ignored; the password is checked against ADMIN_SECRET.
function requireAdminBasicAuth(req, res, next) {
  const authHeader = req.headers['authorization'];
  if (ADMIN_SECRET && authHeader && authHeader.startsWith('Basic ')) {
    const decoded = Buffer.from(authHeader.slice('Basic '.length), 'base64').toString('utf-8');
    const password = decoded.slice(decoded.indexOf(':') + 1);
    if (password === ADMIN_SECRET) return next();
  }
  res.set('WWW-Authenticate', 'Basic realm="Trace Admin"');
  return res.status(401).send('Unauthorized');
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
// Generates a real, honest description of what's actually in an image
// (e.g. "a woman dancing in a red dress"), using a free image-captioning
// model. This is separate from AI-detection — it just describes the
// scene, same as a person would.
async function generateImageCaption(imageBuffer, attempt = 1) {
  try {
    const response = await fetch(
      'https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base',
      {
        method: 'POST',
        headers: { Authorization: `Bearer ${HUGGINGFACE_API_KEY}` },
        body: imageBuffer,
      }
    );
    const data = await response.json();

    if (data.error && data.estimated_time) {
      await new Promise((resolve) => setTimeout(resolve, Math.min(data.estimated_time * 1000, 15000)));
      return generateImageCaption(imageBuffer);
    }

    if (Array.isArray(data) && data[0] && data[0].generated_text) {
      return data[0].generated_text;
    }
    return null;
  } catch (err) {
    if (attempt < 2) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      return generateImageCaption(imageBuffer, attempt + 1);
    }
    console.error('Caption generation failed:', err.message);
    return null; // Non-critical — the scan still works without a caption.
  }
}

function withSubject(caption, sentence) {
  if (!caption) return sentence;
  const capped = caption.charAt(0).toUpperCase() + caption.slice(1);
  return `${capped}. ${sentence}`;
}

// Turns real detector signals into a plain-English explanation. This
// never invents specific visual details about the actual image/video (we
// don't get that from Sightengine — it only gives us a score, not which
// pixels triggered it). What it does do: name what's actually in frame
// (from the captioning model, when available) and explain, generically
// but honestly, what these detectors are known to key on — not "this
// image has fake skin" (a claim we can't back up), but "these models
// typically catch things like unnatural texture, warped fine detail,
// or inconsistent lighting" (true of the detector, not asserted of this
// specific frame).
function reasonForVerdict(verdict, flaggedBy, caption) {
  if (verdict === 'likely_ai') {
    if (flaggedBy === 'ai_generated') {
      return withSubject(
        caption,
        "Traced as AI-generated — flagged strongly by our AI-generation detector, the kind of match these models make when they pick up on unnatural skin or hair texture, inconsistent lighting and shadows, warped background detail, or irregular hands and fine detail that generators still struggle to get right."
      );
    }
    if (flaggedBy === 'deepfake') {
      return withSubject(
        caption,
        "Traced as manipulated — signs of facial editing or a face-swap were found, the kind of thing that usually shows up as a mismatched skin tone at the edge of the face, unnatural blinking or mouth movement, or soft blurring where the swapped face meets the original footage."
      );
    }
    if (flaggedBy === 'ai_music' || flaggedBy === 'ai_speech') {
      return "Traced as AI-generated audio based on synthetic voice patterns — a flatter pitch range and unnaturally even pacing than real speech typically has.";
    }
    return withSubject(caption, "Traced as AI-generated based on multiple signals.");
  }
  if (verdict === 'likely_real') {
    return withSubject(caption, "Traced as real — no strong AI-generation or manipulation signals were detected.");
  }
  return "Signals were mixed — Trace isn't confident enough to call this either way.";
}

function pickTopScore(entries) {
  const valid = entries.filter((e) => typeof e.value === 'number');
  if (valid.length === 0) return null;
  return valid.reduce((a, b) => (b.value > a.value ? b : a));
}

// ---------- PHOTOS ----------
app.post('/analyze', scanLimiter, uploadImage.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No image received. Send it as form-data under the field name "image".',
      });
    }
    if (!SIGHTENGINE_API_USER || !SIGHTENGINE_API_SECRET) {
      return res.status(500).json({ error: 'Server is missing its detection service credentials.' });
    }

    const form = new FormData();
    form.append('media', req.file.buffer, { filename: 'photo.jpg' });
    form.append('models', 'genai,deepfake');
    form.append('api_user', SIGHTENGINE_API_USER);
    form.append('api_secret', SIGHTENGINE_API_SECRET);

    // Run detection and captioning at the same time — captioning is
    // optional/non-critical, so if it fails, the scan still succeeds.
    const [sightengineResponse, caption] = await Promise.all([
      fetch('https://api.sightengine.com/1.0/check.json', { method: 'POST', body: form }),
      generateImageCaption(req.file.buffer),
    ]);
    const data = await sightengineResponse.json();

    if (data.status !== 'success') {
      return res.status(502).json({ error: 'Oops! Trace ran into a problem analyzing that. Please try again.', details: data });
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
      caption,
      reason: reasonForVerdict(verdict, top ? top.label : null, caption),
      flagged_by: top ? top.label : null,
      scores: { ai_generated: aiScore, deepfake: deepfakeScore },
      media_type: 'image',
      source: 'internal',
    });
  } catch (err) {
    console.error('Error in /analyze:', err);
    res.status(500).json({ error: "Oops! Trace couldn't finish that scan. Please try again." });
  }
});

// ---------- VIDEO ----------
app.post('/analyze-video', scanLimiter, uploadMedia.single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No video received. Send it as form-data under the field name "video".',
      });
    }
    if (!SIGHTENGINE_API_USER || !SIGHTENGINE_API_SECRET) {
      return res.status(500).json({ error: 'Server is missing its detection service credentials.' });
    }

    const { aiScore, deepfakeScore, caption } = await analyzeVideoFrames(req.file.buffer);

    const top = pickTopScore([
      { label: 'ai_generated', value: aiScore },
      { label: 'deepfake', value: deepfakeScore },
    ]);

    const { verdict, confidence } = scoreToVerdict(top ? top.value : null);

    res.json({
      verdict,
      confidence,
      caption,
      reason: reasonForVerdict(verdict, top ? top.label : null, caption),
      flagged_by: top ? top.label : null,
      scores: { ai_generated: aiScore, deepfake: deepfakeScore },
      media_type: 'video',
      source: 'internal',
    });
  } catch (err) {
    console.error('Error in /analyze-video:', err);
    res.status(500).json({ error: "Oops! Trace couldn't finish that scan. Please try again." });
  }
});

// ---------- AUDIO ----------
const AUDIO_DETECTION_MODELS = [
  'MelodyMachine/Deepfake-audio-detection-V2',
  'mo-thecreator/Deepfake-audio-detection',
];

async function detectAIAudio(audioBuffer, modelIndex = 0, attempt = 1) {
  if (modelIndex >= AUDIO_DETECTION_MODELS.length) {
    return null;
  }
  const model = AUDIO_DETECTION_MODELS[modelIndex];
  try {
    const response = await fetch(`https://router.huggingface.co/hf-inference/models/${model}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${HUGGINGFACE_API_KEY}` },
      body: audioBuffer,
    });
    const data = await response.json();

    if (data.error && data.estimated_time) {
      await new Promise((resolve) => setTimeout(resolve, Math.min(data.estimated_time * 1000, 15000)));
      return detectAIAudio(audioBuffer, modelIndex, attempt);
    }
    if (data.error) {
      console.error(`Audio model ${model} failed:`, data.error);
      return detectAIAudio(audioBuffer, modelIndex + 1, 1);
    }

    const results = Array.isArray(data[0]) ? data[0] : data;
    if (!Array.isArray(results)) return detectAIAudio(audioBuffer, modelIndex + 1, 1);

    const fakeEntry = results.find((r) => /fake|spoof|synthetic|ai/i.test(r.label));
    if (fakeEntry) return fakeEntry.score;

    const realEntry = results.find((r) => /real|bonafide|genuine|human/i.test(r.label));
    if (realEntry) return 1 - realEntry.score;

    return results[0] ? results[0].score : null;
  } catch (err) {
    if (attempt < 2) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      return detectAIAudio(audioBuffer, modelIndex, attempt + 1);
    }
    return detectAIAudio(audioBuffer, modelIndex + 1, 1);
  }
}

app.post('/analyze-audio', scanLimiter, uploadMedia.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No audio received. Send it as form-data under the field name "audio".',
      });
    }
    if (!HUGGINGFACE_API_KEY) {
      return res.status(500).json({ error: 'Server is missing its detection service credentials.' });
    }

    const score = await detectAIAudio(req.file.buffer);
    const { verdict, confidence } = scoreToVerdict(score);

    res.json({
      verdict,
      confidence,
      reason:
        verdict === 'likely_ai'
          ? 'Traced as AI-generated audio based on synthetic voice patterns.'
          : verdict === 'likely_real'
          ? 'Traced as real. No strong synthetic-voice signals were detected.'
          : "Signals were mixed. Trace isn't confident enough to call this either way.",
      media_type: 'audio',
      source: 'internal',
    });
  } catch (err) {
    console.error('Error in /analyze-audio:', err);
    res.status(500).json({ error: "Oops! Trace couldn't finish that scan. Please try again." });
  }
});

// ---------- TEXT ----------
app.post('/analyze-text', scanLimiter, async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || text.trim().length < 50) {
      return res.status(400).json({ error: 'Please provide at least a few sentences (50+ characters) to analyze.' });
    }
    if (!HUGGINGFACE_API_KEY) {
      return res.status(500).json({ error: 'Server is missing its detection service credentials.' });
    }

    const score = await detectAIText(text);
    const { verdict, confidence } = scoreToVerdict(score);

    res.json({
      verdict,
      confidence,
      reason: confidence !== null ? reasonForTextVerdict(verdict, confidence) : null,
      media_type: 'text',
      source: 'internal',
    });
  } catch (err) {
    console.error('Error in /analyze-text:', err.message, err.stack);
    res.status(500).json({ error: "Oops! Trace couldn't finish that scan. Please try again." });
  }
});

// ---------- DOCUMENTS (PDF / Word) ----------
app.post('/analyze-document', scanLimiter, uploadMedia.single('document'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No document received. Send it as form-data under the field name "document".' });
    }
    if (!HUGGINGFACE_API_KEY) {
      return res.status(500).json({ error: 'Server is missing its detection service credentials.' });
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
      source: 'internal',
    });
  } catch (err) {
    console.error('Error in /analyze-document:', err);
    res.status(500).json({ error: "Oops! Trace couldn't finish that scan. Please try again." });
  }
});

// ---------- LINK (TikTok / Instagram paste-a-link) ----------

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

// Short links (vt.tiktok.com/xxx) redirect before yt-dlp ever sees them.
// Follow that redirect ourselves first and re-check the allowlist against
// where it actually lands, so a link that starts on an allowed domain but
// redirects elsewhere can't reach yt-dlp (which runs on our own server).
// Best-effort: if resolution itself fails, fall through and let yt-dlp
// (which only recognizes a handful of extractors) be the final gate.
async function resolveLinkForDownload(rawUrl) {
  try {
    const response = await fetch(rawUrl, { method: 'GET', redirect: 'follow', timeout: 8000 });
    const resolvedUrl = response.url || rawUrl;
    if (!isAllowedSocialLink(resolvedUrl)) return null;
    return resolvedUrl;
  } catch (err) {
    console.error('Link redirect resolution failed, proceeding with original URL:', err.message);
    return rawUrl;
  }
}

// Admin only — quick way to check if the yt-dlp tool actually got
// installed during the build, without running the full link analysis.
app.get('/admin/check-ytdlp', requireAdmin, (req, res) => {
  execFile('./yt-dlp', ['--version'], (err, stdout, stderr) => {
    if (err) {
      return res.status(500).json({
        installed: false,
        error: err.message,
      });
    }
    res.json({
      installed: true,
      version: stdout.trim(),
    });
  });
});

app.post('/analyze-link', scanLimiter, async (req, res) => {
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
    return res.status(500).json({ error: 'Server is missing its detection service credentials.' });
  }

  const resolvedUrl = await resolveLinkForDownload(url);
  if (!resolvedUrl) {
    return res.status(400).json({
      error: 'That link redirects somewhere Trace doesn\'t recognize as a supported platform.',
    });
  }

  const tempPath = path.join(os.tmpdir(), `link-${Date.now()}.mp4`);

  execFile('./yt-dlp', ['-f', 'best[height<=480][ext=mp4]/worst[ext=mp4]/worst', '-o', tempPath, resolvedUrl], { timeout: 60000 }, async (err) => {
    if (err) {
      console.error('yt-dlp error:', err);
      return res.status(502).json({ error: 'Could not download that link. It may be private, region-locked, or an unsupported platform.' });
    }

    try {
      const videoBuffer = fs.readFileSync(tempPath);
      fs.unlink(tempPath, () => {}); // clean up the temp file either way

      const { aiScore, deepfakeScore, thumbnailBase64, caption } = await analyzeVideoFrames(videoBuffer);

      const top = pickTopScore([
        { label: 'ai_generated', value: aiScore },
        { label: 'deepfake', value: deepfakeScore },
      ]);
      const { verdict, confidence } = scoreToVerdict(top ? top.value : null);

      res.json({
        verdict,
        confidence,
        caption,
        reason: reasonForVerdict(verdict, top ? top.label : null, caption),
        flagged_by: top ? top.label : null,
        media_type: 'video',
        source: 'internal',
        from_link: true,
        thumbnail_base64: thumbnailBase64,
      });
    } catch (processErr) {
      console.error('Error processing downloaded video:', processErr);
      fs.unlink(tempPath, () => {});
      res.status(500).json({ error: 'Downloaded the video but could not analyze it.' });
    }
  });
});

// ---------- TRENDING FEED ----------

// Public — the app calls this to load the scrollable feed.
app.get('/trending', publicLimiter, async (req, res) => {
  try {
    const result = await pool.query(
      'SELECT id, title, video_url, thumbnail_url, ai_score, category, source_platform, post_type, description, view_count, share_count, created_at, (thumbnail_image IS NOT NULL) AS has_generated_thumbnail FROM trending_posts ORDER BY created_at DESC LIMIT 50'
    );
    const posts = result.rows.map((row) => {
      const post = { ...row };
      if (row.has_generated_thumbnail) {
        post.thumbnail_url = `https://ai-screen-detector-backend.onrender.com/trending/${row.id}/thumbnail`;
      }
      delete post.has_generated_thumbnail;
      return post;
    });
    res.json(posts);
  } catch (err) {
    console.error('Error in /trending:', err);
    res.status(500).json({ error: 'Could not load trending posts.' });
  }
});

// Public — serves the actual thumbnail image generated from the video.
app.get('/trending/:id/thumbnail', publicLimiter, async (req, res) => {
  try {
    const result = await pool.query('SELECT thumbnail_image FROM trending_posts WHERE id = $1', [req.params.id]);
    if (result.rows.length === 0 || !result.rows[0].thumbnail_image) {
      return res.status(404).send('No thumbnail available.');
    }
    res.set('Content-Type', 'image/jpeg');
    res.send(result.rows[0].thumbnail_image);
  } catch (err) {
    console.error('Error in /trending/:id/thumbnail:', err);
    res.status(500).send('Could not load thumbnail.');
  }
});

// Admin only — grabs a real frame from the post's video and stores it
// as the thumbnail, using the same yt-dlp + ffmpeg tools as Link scan.
app.post('/admin/trending/:id/generate-thumbnail', requireAdmin, async (req, res) => {
  try {
    const postResult = await pool.query('SELECT video_url FROM trending_posts WHERE id = $1', [req.params.id]);
    if (postResult.rows.length === 0) {
      return res.status(404).json({ error: 'Post not found.' });
    }
    const videoUrl = postResult.rows[0].video_url;
    const tempVideoPath = path.join(os.tmpdir(), `thumb-src-${Date.now()}.mp4`);
    const tempFramePath = path.join(os.tmpdir(), `thumb-frame-${Date.now()}.jpg`);

    await new Promise((resolve, reject) => {
      execFile('./yt-dlp', ['-f', 'best[height<=480][ext=mp4]/worst[ext=mp4]/worst', '-o', tempVideoPath, videoUrl], { timeout: 60000 }, (err) =>
        err ? reject(err) : resolve()
      );
    });

    await new Promise((resolve, reject) => {
      execFile(ffmpegPath, ['-i', tempVideoPath, '-vframes', '1', '-q:v', '3', tempFramePath], (err) =>
        err ? reject(err) : resolve()
      );
    });

    const frameBuffer = fs.readFileSync(tempFramePath);
    await pool.query('UPDATE trending_posts SET thumbnail_image = $1 WHERE id = $2', [frameBuffer, req.params.id]);

    fs.unlink(tempVideoPath, () => {});
    fs.unlink(tempFramePath, () => {});

    res.json({
      success: true,
      thumbnail_url: `https://ai-screen-detector-backend.onrender.com/trending/${req.params.id}/thumbnail`,
    });
  } catch (err) {
    console.error('Error in generate-thumbnail:', err);
    res.status(500).json({ error: 'Could not generate thumbnail from that video.' });
  }
});

// Public — call when a post is opened, to count it as a view.
app.post('/trending/:id/view', publicLimiter, async (req, res) => {
  try {
    const result = await pool.query(
      'UPDATE trending_posts SET view_count = view_count + 1 WHERE id = $1 RETURNING view_count',
      [req.params.id]
    );
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Post not found.' });
    }
    res.json({ view_count: result.rows[0].view_count });
  } catch (err) {
    console.error('Error in /trending/:id/view:', err);
    res.status(500).json({ error: 'Could not record view.' });
  }
});

// Public — call when a post is shared, to count it as a share.
app.post('/trending/:id/share', publicLimiter, async (req, res) => {
  try {
    const result = await pool.query(
      'UPDATE trending_posts SET share_count = share_count + 1 WHERE id = $1 RETURNING share_count',
      [req.params.id]
    );
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Post not found.' });
    }
    res.json({ share_count: result.rows[0].share_count });
  } catch (err) {
    console.error('Error in /trending/:id/share:', err);
    res.status(500).json({ error: 'Could not record share.' });
  }
});

// Admin only — adds a new video post to the feed.
app.post('/admin/trending', requireAdmin, async (req, res) => {
  const { title, video_url, thumbnail_url, ai_score, category, source_platform, description } = req.body;
  if (!title || !video_url) {
    return res.status(400).json({ error: 'title and video_url are required.' });
  }
  try {
    const result = await pool.query(
      `INSERT INTO trending_posts (title, video_url, thumbnail_url, ai_score, category, source_platform, description, post_type)
       VALUES ($1, $2, $3, $4, $5, $6, $7, 'video') RETURNING *`,
      [title, video_url, thumbnail_url || null, ai_score || null, category || null, source_platform || null, description || null]
    );
    res.json(result.rows[0]);
  } catch (err) {
    console.error('Error in /admin/trending:', err);
    res.status(500).json({ error: 'Could not add post.' });
  }
});

// Admin only — adds a social media *profile* post (e.g. a suspected AI
// influencer account) to the feed: a profile link, description, and an
// admin-uploaded screenshot, instead of a video that gets frame-extracted.
app.post('/admin/trending/profile', requireAdmin, uploadImage.single('screenshot'), async (req, res) => {
  const { title, profile_url, description, ai_score, category, source_platform } = req.body;
  if (!title || !profile_url) {
    return res.status(400).json({ error: 'title and profile_url are required.' });
  }
  if (!req.file) {
    return res.status(400).json({ error: 'A screenshot is required. Send it as form-data under the field name "screenshot".' });
  }
  try {
    const result = await pool.query(
      `INSERT INTO trending_posts (title, video_url, ai_score, category, source_platform, description, post_type, thumbnail_image)
       VALUES ($1, $2, $3, $4, $5, $6, 'profile', $7)
       RETURNING id, title, video_url, ai_score, category, source_platform, description, post_type, view_count, share_count, created_at`,
      [title, profile_url, ai_score || null, category || null, source_platform || null, description || null, req.file.buffer]
    );
    res.json({ ...result.rows[0], thumbnail_url: `https://ai-screen-detector-backend.onrender.com/trending/${result.rows[0].id}/thumbnail` });
  } catch (err) {
    console.error('Error in /admin/trending/profile:', err);
    res.status(500).json({ error: 'Could not add profile post.' });
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

// ---------- ADMIN DASHBOARD (simple webpage, no curl needed) ----------
app.get('/admin', requireAdminBasicAuth, (req, res) => {
  res.send(`<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace Admin</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.js"></script>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 16px; background: #f5f5f5; }
  h1 { font-size: 20px; }
  input, select, button, textarea { width: 100%; padding: 10px; margin: 6px 0; box-sizing: border-box; border-radius: 8px; border: 1px solid #ccc; font-size: 15px; }
  button { background: #007AFF; color: white; border: none; font-weight: 600; cursor: pointer; }
  button:disabled { background: #aaa; }
  .card { background: white; padding: 16px; border-radius: 12px; margin-bottom: 16px; }
  .post { display: flex; gap: 10px; align-items: center; border-bottom: 1px solid #eee; padding: 10px 0; }
  .post img { width: 60px; height: 60px; object-fit: cover; border-radius: 8px; background: #eee; }
  .post-info { flex: 1; font-size: 13px; }
  .delete-btn { background: #ff3b30; width: auto; padding: 6px 10px; font-size: 12px; }
  #status, #profileStatus { font-size: 13px; color: #666; margin-top: 6px; }
  .type-tag { display: inline-block; font-size: 10px; font-weight: 700; text-transform: uppercase; padding: 2px 6px; border-radius: 4px; background: #eee; color: #666; margin-left: 6px; }
  h2 { font-size: 16px; margin-top: 0; }
  #cropperContainer { display: none; margin-top: 10px; }
  #cropperImageWrap { max-height: 320px; overflow: hidden; background: #000; border-radius: 8px; }
  #cropperImage { display: block; max-width: 100%; }
  .crop-actions { display: flex; gap: 8px; }
  .crop-actions button { width: auto; flex: 1; }
  #cropPreviewWrap { display: none; align-items: center; gap: 10px; margin-top: 8px; }
  #cropPreview { width: 50px; height: 50px; object-fit: cover; border-radius: 8px; }
</style>
</head>
<body>
<h1>Trace Admin Dashboard</h1>

<div class="card">
  <h2>Add Video Post</h2>
  <label>Admin Password</label>
  <input id="secret" type="password" placeholder="Your admin secret">

  <label>TikTok Link</label>
  <input id="linkUrl" type="text" placeholder="https://vt.tiktok.com/...">
  <button onclick="fetchPreview()">1. Fetch Preview</button>

  <label>Title</label>
  <input id="title" type="text">
  <label>Category</label>
  <input id="category" type="text" placeholder="celebrity, scam, music...">
  <label>Source Platform</label>
  <select id="platform">
    <option>TikTok</option>
    <option>Instagram</option>
    <option>Facebook</option>
    <option>Twitter</option>
  </select>
  <label>AI Score (0.0 - 1.0)</label>
  <input id="aiScore" type="text" placeholder="0.9">

  <button onclick="createPost()">2. Create Post + Generate Thumbnail</button>
  <div id="status"></div>
</div>

<div class="card">
  <h2>Add Profile Post (AI influencer accounts, etc.)</h2>
  <label>Profile Link</label>
  <input id="profileUrl" type="text" placeholder="https://www.instagram.com/username or TikTok profile URL">

  <label>Title / Handle</label>
  <input id="profileTitle" type="text" placeholder="@username or account name">

  <label>Description</label>
  <textarea id="profileDescription" rows="3" placeholder="What makes this account notable — fully AI-generated, deepfake, etc."></textarea>

  <label>Category</label>
  <input id="profileCategory" type="text" placeholder="AI influencer, deepfake, brand...">
  <label>Source Platform</label>
  <select id="profilePlatform">
    <option>TikTok</option>
    <option>Instagram</option>
    <option>Facebook</option>
    <option>Twitter</option>
  </select>
  <label>AI Score (0.0 - 1.0)</label>
  <input id="profileAiScore" type="text" placeholder="0.95">

  <label>Screenshot of the profile</label>
  <input id="profileScreenshot" type="file" accept="image/*">

  <div id="cropperContainer">
    <div id="cropperImageWrap">
      <img id="cropperImage">
    </div>
    <div class="crop-actions">
      <button onclick="applyCrop()">Apply Crop</button>
      <button onclick="cancelCrop()" style="background:#8e8e93;">Cancel</button>
    </div>
  </div>

  <div id="cropPreviewWrap">
    <img id="cropPreview">
    <span style="font-size:13px;color:#666;">Cropped — this is what will upload.</span>
  </div>

  <button onclick="createProfilePost()">Create Profile Post</button>
  <div id="profileStatus"></div>
</div>

<div class="card">
  <h2>Existing Posts</h2>
  <div id="postList"></div>
  <button onclick="loadPosts()" style="margin-top:10px;">Refresh List</button>
</div>

<script>
function secret() { return document.getElementById('secret').value; }
function setStatus(msg) { document.getElementById('status').innerText = msg; }

async function fetchPreview() {
  const url = document.getElementById('linkUrl').value;
  if (!url) return alert('Paste a link first');
  setStatus('Fetching preview...');
  const res = await fetch('/admin/fetch-preview?url=' + encodeURIComponent(url), {
    headers: { 'x-admin-secret': secret() }
  });
  const data = await res.json();
  if (data.error) return setStatus('Error: ' + data.error);
  document.getElementById('title').value = data.title || '';
  document.getElementById('linkUrl').value = data.resolved_url || url;
  setStatus('Preview loaded. Fill in the rest and create the post.');
}

async function createPost() {
  setStatus('Creating post...');
  const body = {
    title: document.getElementById('title').value,
    video_url: document.getElementById('linkUrl').value,
    category: document.getElementById('category').value,
    source_platform: document.getElementById('platform').value,
    ai_score: parseFloat(document.getElementById('aiScore').value) || null
  };
  const res = await fetch('/admin/trending', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-admin-secret': secret() },
    body: JSON.stringify(body)
  });
  const data = await res.json();
  if (data.error) return setStatus('Error: ' + data.error);

  setStatus('Post created. Generating thumbnail (can take up to a minute)...');
  const thumbRes = await fetch('/admin/trending/' + data.id + '/generate-thumbnail', {
    method: 'POST',
    headers: { 'x-admin-secret': secret() }
  });
  const thumbData = await thumbRes.json();
  if (thumbData.error) {
    setStatus('Post created, but thumbnail failed: ' + thumbData.error);
  } else {
    setStatus('Done! Post and thumbnail created successfully.');
  }
  loadPosts();
}

let cropper = null;
let croppedScreenshotBlob = null;

document.getElementById('profileScreenshot').addEventListener('change', (e) => {
  const file = e.target.files[0];
  croppedScreenshotBlob = null;
  document.getElementById('cropPreviewWrap').style.display = 'none';
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (evt) => {
    const img = document.getElementById('cropperImage');
    img.src = evt.target.result;
    document.getElementById('cropperContainer').style.display = 'block';
    if (cropper) cropper.destroy();
    cropper = new Cropper(img, { viewMode: 1, autoCropArea: 1, background: false });
  };
  reader.readAsDataURL(file);
});

function applyCrop() {
  if (!cropper) return;
  cropper.getCroppedCanvas().toBlob((blob) => {
    croppedScreenshotBlob = blob;
    document.getElementById('cropPreview').src = URL.createObjectURL(blob);
    document.getElementById('cropPreviewWrap').style.display = 'flex';
    cropper.destroy();
    cropper = null;
    document.getElementById('cropperContainer').style.display = 'none';
  }, 'image/jpeg', 0.92);
}

function cancelCrop() {
  if (cropper) { cropper.destroy(); cropper = null; }
  document.getElementById('cropperContainer').style.display = 'none';
  document.getElementById('profileScreenshot').value = '';
  croppedScreenshotBlob = null;
  document.getElementById('cropPreviewWrap').style.display = 'none';
}

async function createProfilePost() {
  const setProfileStatus = (msg) => document.getElementById('profileStatus').innerText = msg;
  const fileInput = document.getElementById('profileScreenshot');
  const screenshotToUpload = croppedScreenshotBlob || fileInput.files[0];
  if (!screenshotToUpload) return alert('Choose a screenshot image first');
  if (cropper) return alert('Click "Apply Crop" (or "Cancel") before creating the post.');

  setProfileStatus('Creating profile post...');
  const form = new FormData();
  form.append('title', document.getElementById('profileTitle').value);
  form.append('profile_url', document.getElementById('profileUrl').value);
  form.append('description', document.getElementById('profileDescription').value);
  form.append('category', document.getElementById('profileCategory').value);
  form.append('source_platform', document.getElementById('profilePlatform').value);
  form.append('ai_score', parseFloat(document.getElementById('profileAiScore').value) || '');
  form.append('screenshot', screenshotToUpload, 'screenshot.jpg');

  const res = await fetch('/admin/trending/profile', {
    method: 'POST',
    headers: { 'x-admin-secret': secret() },
    body: form
  });
  const data = await res.json();
  if (data.error) return setProfileStatus('Error: ' + data.error);

  setProfileStatus('Done! Profile post created.');
  document.getElementById('profileTitle').value = '';
  document.getElementById('profileUrl').value = '';
  document.getElementById('profileDescription').value = '';
  document.getElementById('profileCategory').value = '';
  document.getElementById('profileAiScore').value = '';
  fileInput.value = '';
  croppedScreenshotBlob = null;
  document.getElementById('cropPreviewWrap').style.display = 'none';
  loadPosts();
}

async function loadPosts() {
  const res = await fetch('/trending');
  const posts = await res.json();
  const list = document.getElementById('postList');
  list.innerHTML = posts.map(p => \`
    <div class="post">
      <img src="\${p.thumbnail_url || ''}" onerror="this.style.display='none'">
      <div class="post-info">
        <strong>\${p.title}</strong><span class="type-tag">\${p.post_type || 'video'}</span><br>
        \${p.source_platform || ''} · \${p.category || ''} · \${Math.round((p.ai_score||0)*100)}% AI
        \${p.description ? '<br>' + p.description : ''}
      </div>
      <button class="delete-btn" onclick="deletePost(\${p.id})">Delete</button>
    </div>
  \`).join('');
}

async function deletePost(id) {
  if (!confirm('Delete this post?')) return;
  await fetch('/admin/trending/' + id, {
    method: 'DELETE',
    headers: { 'x-admin-secret': secret() }
  });
  loadPosts();
}

loadPosts();
</script>
</body>
</html>`);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
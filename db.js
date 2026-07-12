// Connects to your Neon Postgres database, and makes sure the
// "trending_posts" table exists (creates it automatically the first
// time the server starts, does nothing if it's already there).

const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: true },
});

async function initDB() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS trending_posts (
      id SERIAL PRIMARY KEY,
      title TEXT NOT NULL,
      video_url TEXT NOT NULL,
      thumbnail_url TEXT,
      ai_score DOUBLE PRECISION,
      category TEXT,
      source_platform TEXT,
      view_count INTEGER DEFAULT 0,
      share_count INTEGER DEFAULT 0,
      created_at TIMESTAMPTZ DEFAULT now()
    );
  `);
  // Adds columns automatically if the table already existed from
  // before, without losing any existing data.
  await pool.query(`ALTER TABLE trending_posts ADD COLUMN IF NOT EXISTS source_platform TEXT;`);
  await pool.query(`ALTER TABLE trending_posts ADD COLUMN IF NOT EXISTS view_count INTEGER DEFAULT 0;`);
  await pool.query(`ALTER TABLE trending_posts ADD COLUMN IF NOT EXISTS share_count INTEGER DEFAULT 0;`);
  await pool.query(`ALTER TABLE trending_posts ADD COLUMN IF NOT EXISTS thumbnail_image BYTEA;`);
  // 'video' (existing link-to-a-video posts) or 'profile' (an Instagram/
  // TikTok account, e.g. a suspected AI influencer, linked by profile URL
  // with an admin-uploaded screenshot instead of an extracted video frame).
  await pool.query(`ALTER TABLE trending_posts ADD COLUMN IF NOT EXISTS post_type TEXT NOT NULL DEFAULT 'video';`);
  await pool.query(`ALTER TABLE trending_posts ADD COLUMN IF NOT EXISTS description TEXT;`);

  // Perceptual-hash cache for Photo/Video/Link scans. Lets a repeat scan of
  // the same (or a re-compressed/re-encoded copy of the same) image or video
  // skip the paid Sightengine call entirely and return instantly.
  await pool.query(`
    CREATE TABLE IF NOT EXISTS scan_cache (
      id SERIAL PRIMARY KEY,
      hash TEXT NOT NULL,
      media_type TEXT NOT NULL,
      verdict TEXT NOT NULL,
      confidence DOUBLE PRECISION,
      reason TEXT,
      caption TEXT,
      ai_score DOUBLE PRECISION,
      deepfake_score DOUBLE PRECISION,
      flagged_by TEXT,
      provider TEXT NOT NULL DEFAULT 'sightengine',
      hit_count INTEGER DEFAULT 0,
      created_at TIMESTAMPTZ DEFAULT now(),
      last_hit_at TIMESTAMPTZ DEFAULT now()
    );
  `);
  await pool.query(`CREATE INDEX IF NOT EXISTS scan_cache_hash_idx ON scan_cache (media_type, hash);`);
  await pool.query(`CREATE INDEX IF NOT EXISTS scan_cache_recent_idx ON scan_cache (media_type, created_at DESC);`);

  // ---------- FEEDBACK LOOP ----------

  // "We're not sure either — what's your gut call?" One row per response.
  // Purely for later analysis (e.g. recalibrating the uncertain threshold)
  // — never displayed anywhere, so no moderation needed.
  await pool.query(`
    CREATE TABLE IF NOT EXISTS uncertain_feedback (
      id SERIAL PRIMARY KEY,
      media_type TEXT NOT NULL,
      confidence DOUBLE PRECISION,
      user_guess TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT now()
    );
  `);

  // Trending posts don't all come from a pasted link anymore (a Photo/Video
  // scan can be submitted to Trending too — see trending_submissions below),
  // so a post doesn't always have an external link to open.
  await pool.query(`ALTER TABLE trending_posts ALTER COLUMN video_url DROP NOT NULL;`);

  // Crowd agree/disagree votes on Trending posts. One row per device per
  // post (re-voting updates the existing row instead of adding a new one).
  await pool.query(`
    CREATE TABLE IF NOT EXISTS trending_votes (
      id SERIAL PRIMARY KEY,
      post_id INTEGER NOT NULL REFERENCES trending_posts(id) ON DELETE CASCADE,
      device_id TEXT NOT NULL,
      vote TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT now(),
      UNIQUE (post_id, device_id)
    );
  `);

  // Anonymous "Add to Trending" submissions, awaiting admin review before
  // (if approved) becoming a real row in trending_posts. video_url is only
  // set for Link-scan submissions; Photo/Video submissions carry the media
  // itself in thumbnail_image instead.
  await pool.query(`
    CREATE TABLE IF NOT EXISTS trending_submissions (
      id SERIAL PRIMARY KEY,
      media_type TEXT NOT NULL,
      video_url TEXT,
      thumbnail_image BYTEA,
      category_tag TEXT,
      verdict TEXT,
      confidence DOUBLE PRECISION,
      reason TEXT,
      status TEXT NOT NULL DEFAULT 'pending',
      created_at TIMESTAMPTZ DEFAULT now()
    );
  `);
  await pool.query(`CREATE INDEX IF NOT EXISTS trending_submissions_status_idx ON trending_submissions (status, created_at DESC);`);

  console.log('Database ready: trending_posts, scan_cache, and feedback-loop tables exist.');
}

module.exports = { pool, initDB };

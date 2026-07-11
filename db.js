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

  console.log('Database ready: trending_posts and scan_cache tables exist.');
}

module.exports = { pool, initDB };

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
  console.log('Database ready: trending_posts table exists.');
}

module.exports = { pool, initDB };

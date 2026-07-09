// Connects to your Neon Postgres database, and makes sure the
// "trending_posts" table exists (creates it automatically the first
// time the server starts, does nothing if it's already there).

const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false },
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
      created_at TIMESTAMPTZ DEFAULT now()
    );
  `);
  // Adds the column automatically if the table already existed
  // from before this change, without losing existing data.
  await pool.query(`
    ALTER TABLE trending_posts ADD COLUMN IF NOT EXISTS source_platform TEXT;
  `);
  console.log('Database ready: trending_posts table exists.');
}

module.exports = { pool, initDB };

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
      created_at TIMESTAMPTZ DEFAULT now()
    );
  `);
  console.log('Database ready: trending_posts table exists.');
}

module.exports = { pool, initDB };

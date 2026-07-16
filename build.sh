#!/usr/bin/env bash
# Render (and any other host) runs this instead of a bare `npm install`.
# It additionally fetches the standalone yt-dlp binary that /analyze-link
# and /admin/*-thumbnail routes shell out to — without this, those routes
# fail with "yt-dlp: command not found" on a fresh deploy.
set -euo pipefail

npm install

# yt-dlp_linux is the self-contained PyInstaller-bundled binary (no system
# python3 dependency) — matches what was actually already proven working
# in production, unlike the generic cross-platform "yt-dlp" build.
curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux -o yt-dlp
chmod +x yt-dlp
./yt-dlp --version

# YouTube's "n challenge" throttling deobfuscation requires a real
# JavaScript runtime as of mid-2026 — without one, yt-dlp fails most
# YouTube downloads with "No video formats found" regardless of IP or
# cookies. yt-dlp looks for Deno specifically, so fetch a standalone
# binary the same way yt-dlp itself is fetched above.
curl -L https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip -o deno.zip
unzip -o deno.zip -d .
rm deno.zip
chmod +x deno
./deno --version

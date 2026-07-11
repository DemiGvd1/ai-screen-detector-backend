#!/usr/bin/env bash
# Render (and any other host) runs this instead of a bare `npm install`.
# It additionally fetches the standalone yt-dlp binary that /analyze-link
# and /admin/*-thumbnail routes shell out to — without this, those routes
# fail with "yt-dlp: command not found" on a fresh deploy.
set -euo pipefail

npm install

curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o yt-dlp
chmod +x yt-dlp
./yt-dlp --version

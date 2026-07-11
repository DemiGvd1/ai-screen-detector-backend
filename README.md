# AI Screen Detector — Backend

A tiny server with one job: take an image, ask Sightengine if it looks AI-generated, and reply with a simple answer.

## Part A — Run it on your own Mac first (to make sure it works)

### 1. Install Node.js (if you don't have it)
Node.js is the program that runs JavaScript outside a browser — it's what makes `server.js` runnable.
- Go to **nodejs.org**, download the **LTS** version for Mac, run the installer.
- Check it worked: open Terminal and run:
  ```
  node --version
  ```
  You should see something like `v22.x.x`.

### 2. Get these files onto your Mac
Download this whole `ai-screen-detector-backend` folder, or ask Claude Code to recreate it in a project folder.

### 3. Install the dependencies
Open Terminal, navigate into the folder, and run:
```
cd ai-screen-detector-backend
npm install
```
This downloads the small libraries the server depends on (listed in `package.json`) into a `node_modules` folder. You'll only need to do this once (or again if you change dependencies).

### 4. Add your real API key
```
cp .env.example .env
```
Then open `.env` in any text editor and paste in the **API User** and **API Secret** you copied from your Sightengine dashboard. This `.env` file stays on your machine only — never upload it anywhere public (it's already excluded via `.gitignore` below).

### 5. Run the server
```
npm start
```
You should see:
```
Server listening on port 3000
```
Leave this Terminal window open — the server is now running on your Mac, reachable at `http://localhost:3000`.

### 6. Test it with curl
Open a **second** Terminal window (leave the server running in the first one), grab any image file, and run:
```
curl -X POST http://localhost:3000/analyze \
  -F "image=@/path/to/some_picture.jpg"
```
Replace `/path/to/some_picture.jpg` with a real file path — e.g. drag a photo from Finder into Terminal after typing `-F "image=@` to auto-fill the path.

If it's working, you'll get back something like:
```json
{ "verdict": "likely_real", "confidence": 0.02, "source": "sightengine" }
```
Try it with a picture you know is AI-generated (e.g. save one from an AI image generator) and confirm the verdict flips to `likely_ai`.

Once this works locally, the logic is proven — the rest is just putting it online so your iPhone can reach it.

---

## Part B — Put it online with Render (free)

Your iPhone can't reach `localhost` on your Mac, so the server needs a public address. Render hosts it for free and gives you a real `https://` URL.

### 1. Put the code on GitHub
Render deploys by connecting to a GitHub repository.
- Create a free account at **github.com** if you don't have one.
- Create a new repository (e.g. `ai-screen-detector-backend`), and push this folder to it:
  ```
  git init
  git add .
  git commit -m "first version of backend"
  git branch -M main
  git remote add origin https://github.com/YOUR_USERNAME/ai-screen-detector-backend.git
  git push -u origin main
  ```
  (Claude Code can run all of this for you if you'd rather not type git commands by hand.)

### 2. Create the service on Render
- Go to **render.com**, sign up (no credit card needed for the free tier).
- Click **New +** → **Web Service**.
- Connect your GitHub account, select the `ai-screen-detector-backend` repo.
- Render will detect it's a Node app. Set:
  - **Build Command:** `bash build.sh` (not just `npm install` — this also downloads the `yt-dlp` binary that Link scan and the admin thumbnail generator need; see `build.sh` and `render.yaml`)
  - **Start Command:** `npm start`
  - **Instance Type:** Free
- Click **Create Web Service**.
- If you already have a service running with `npm install` as its Build Command, update it to `bash build.sh` in the Render dashboard (Settings → Build Command) and trigger a manual redeploy — otherwise yt-dlp only exists on that instance because someone installed it by hand, and it'll vanish on the next redeploy.

### 3. Add your secret key
This is the important part — your `.env` file never got uploaded to GitHub (good, it shouldn't), so Render doesn't have your Sightengine key yet.
- In the Render dashboard for your new service, go to **Environment**.
- Add these environment variables (see `.env.example` for what each one is):
  - `SIGHTENGINE_API_USER`
  - `SIGHTENGINE_API_SECRET`
  - `HUGGINGFACE_API_KEY`
  - `DATABASE_URL`
  - `ADMIN_SECRET`
- Save — Render will redeploy automatically with the keys available.

### 4. Get your public URL
Once deployed, Render shows you a URL like:
```
https://ai-screen-detector-backend.onrender.com
```
Test it exactly like before, just pointed at this URL instead of localhost:
```
curl -X POST https://ai-screen-detector-backend.onrender.com/analyze \
  -F "image=@/path/to/some_picture.jpg"
```

**One quirk of the free tier:** if nobody hits the server for 15 minutes, Render puts it to sleep to save resources. The next request after that takes about 30–60 seconds to "wake up," then it's fast again. Fine for development; if this bugs you later, Render's $7/month tier removes it.

This URL is what your iPhone app will call instead of `localhost`. Save it somewhere — you'll need it in the app code.

---

## What's in this folder
- `server.js` — the actual server logic
- `package.json` — lists the small libraries it needs
- `.env.example` — template for your secret keys (copy to `.env`, never commit `.env` itself)
- `.gitignore` — tells git to skip `node_modules` and `.env` when uploading to GitHub

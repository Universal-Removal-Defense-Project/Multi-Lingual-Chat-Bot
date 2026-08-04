# Deployment Guide

How to run the URDP Multi-Lingual Assistant, from a laptop to production.

## Prerequisites
- A Groq API key: https://console.groq.com/keys

---

## Option A — Run locally (no Docker)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."   # or copy .env.example to .env and fill it in
streamlit run ui.py
```
Open http://localhost:8501.

## Option B — Run with Docker
```bash
export GROQ_API_KEY="gsk_..."   # or put it in a local .env file
docker compose up --build
```
Open http://localhost:8501. Conversation history persists in `./data`.

---

## Option C — Deploy to DigitalOcean App Platform

DigitalOcean App Platform can build and host the app directly from the GitHub repo.

### 1. Push the repo to GitHub
Ensure `main` (or your deploy branch) is up to date. Do **not** commit secrets —
`.env` and `.streamlit/secrets.toml` are gitignored.

### 2. Create the app
1. In the DigitalOcean console: **Create → Apps**.
2. Choose the GitHub repository and the branch to deploy.
3. App Platform detects the `Dockerfile` and builds from it.

### 3. Configure the HTTP port
Set the app's **HTTP Port** to `8501` (the port Streamlit serves on).

### 4. Set environment variables
Under the component's **Settings → Environment Variables**, add:

| Key | Value | Scope | Encrypt |
|-----|-------|-------|---------|
| `GROQ_API_KEY` | your Groq key | Run time | ✅ Yes (secret) |

Optional overrides: `GROQ_MODEL`, `GROQ_DETECT_MODEL`.

### 5. Deploy
Trigger the deployment. App Platform builds the image and gives you a public URL.
Pushes to the deploy branch redeploy automatically.

---

## Deployment checklist
- [ ] `GROQ_API_KEY` set as an **encrypted** env var (never committed)
- [ ] HTTP port set to `8501`
- [ ] Build succeeds from the `Dockerfile`
- [ ] App loads at the public URL and responds to a test message
- [ ] Language selector and dark-mode toggle work
- [ ] (If persistence matters) note that container storage is ephemeral; a
      restart clears local `data/`. Attach persistent storage or a database
      before relying on saved history in production.

# Deployment Guide: Hugging Face Backend + Vercel Frontend

Ship Vibe-Sync end-to-end with zero paid services:

- **FastAPI + LightGBM backend** runs inside a Hugging Face Space using the repo Dockerfile.
- **Spotify-style frontend** is deployed as a static site on Vercel with the project root set to `frontend/`.

> ⚠️ Prerequisites
> * Windows PowerShell (already the default shell in this repo)
> * [Docker Desktop](https://www.docker.com/products/docker-desktop/) for local smoke tests
> * Accounts on Hugging Face and Vercel (both support GitHub sign-in, no billing info required)

---

## 1. Prepare the Repository

1. Clone or open the project root.
2. Ensure the following exist:
   - `backend/`, `src/`, `models/`, `frontend/`
   - `data/raw/spotify_tracks.csv` (full Kaggle dataset) and the trained `models/*.pkl`
3. Copy `.env.example` → `.env` and provide Spotify credentials if you want album art and previews. The app still functions without them (placeholder artwork).

---

## 2. Local Smoke Test (Optional but Recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

uvicorn backend.server:app --reload
```

Browse to http://localhost:8000/ for the UI and http://localhost:8000/docs for the swagger. Stop the server with `Ctrl+C`.

---

## 3. Mirror a Lightweight Space Copy

Hugging Face Spaces reject files larger than 10 MB, so mirror a slim copy of the repo before pushing.

```powershell
# Mirror the repo without heavy folders (idempotent command)
mkdir ..\hf-space -ErrorAction SilentlyContinue
C:\Windows\System32\robocopy.exe . ..\hf-space /MIR /XD .git .venv data\raw data\processed results __pycache__ .pytest_cache

# Remove lingering large files just in case
Remove-Item ..\hf-space\dataset.csv -ErrorAction SilentlyContinue
Remove-Item ..\hf-space\data\raw -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ..\hf-space\data\processed -Recurse -Force -ErrorAction SilentlyContinue

# Initialise and push to the Space (replace the URL with yours)
cd ..\hf-space
git init
git remote add origin https://huggingface.co/spaces/YashHugs/context-aware-music-recommendation
git add .
git commit -m "Initial lightweight Space copy"
git push origin HEAD:main --force
```

The Hugging Face builder automatically executes `docker build` using the repo Dockerfile. Watch the **Build Logs** tab until it prints “Application startup complete”.

---

## 4. Configure Spotify Secrets on the Space

1. Open the Space ➜ **Settings** ➜ **Variables**.
2. Add:
   - `SPOTIFY_CLIENT_ID`
   - `SPOTIFY_CLIENT_SECRET`
   - `SPOTIFY_REDIRECT_URI` (optional, only needed for OAuth callbacks)
3. Restart the Space. Healthy logs look like:

```
✅ Spotify client initialized
✅ Vibe-Sync API ready!
```

Your backend URL follows the pattern `https://<username>-<space>.hf.space` (e.g. `https://yashhugs-context-aware-music-recommendation.hf.space`). Keep it for the frontend.

---

## 5. Deploy the Frontend to Vercel

The `frontend/` folder is a self-contained static bundle. Once linked, Vercel redeploys on every push to `main`.

```powershell
# First-time deploy (links the project automatically)
vercel deploy --prod --yes --cwd frontend

# Optionally point a friendly domain to the new deployment
vercel alias set <deployment-url> context-aware-music-recommendation.vercel.app --scope dev-yash
```

During the initial run the CLI prompts for a few answers:

| Prompt | Answer |
| --- | --- |
| “Set up and deploy `<path>`?” | **Yes** |
| “Which scope?” | Choose your personal/team scope (e.g. *Yash's projects*) |
| “Link to existing project?” | **Yes**, select `context-aware-music-recommendation-system` or create one |
| “Framework / Build / Output directory?” | Accept defaults (no build command, output directory is the project root) |

After linking you can redeploy with a single command:

```powershell
vercel --prod --yes --cwd frontend
```

Make sure `frontend/static/config.js` points the browser at the Space API:

```javascript
window.API_BASE_URL = "https://yashhugs-context-aware-music-recommendation.hf.space";
```

Vercel serves the frontend at `https://context-aware-music-recommendation.vercel.app/` (or whatever alias you set).

---

## 6. Keeping Deployments in Sync

| What changed? | Action |
| --- | --- |
| Backend code / requirements | Re-run the mirror command, amend the commit inside `hf-space`, then `git push origin HEAD:main --force`. |
| Frontend assets | Commit to `main` and run `vercel --prod --yes --cwd frontend` (or let the Git webhook redeploy). |
| Models / dataset | Update them in the primary repo, mirror new copies into `hf-space`, then push again. |

> ✅ Tip: add “Mirror Space” and “Deploy Vercel” to your PR checklist to avoid drift.

---

## 7. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Push blocked by Hugging Face (metadata or >10 MB files) | Trim README metadata to documented limits and remove heavy assets before mirroring. |
| `jinja2 must be installed` in Space logs | Ensure `jinja2>=3.1.0` remains in both `requirements.txt` files before pushing. |
| Frontend still calls the old backend | Update `frontend/static/config.js`, redeploy, and hard refresh (`Ctrl+Shift+R`). |
| Vercel shows 404 | Deploy from `frontend/` so `index.html` lives at the project root and assets resolve via `./static/...`. |
| Space stuck on “Starting…” | Inspect build logs—usually missing secrets or dependency typos. Fix locally, mirror again, and force push. |

---

## 8. Optional Automation

- Wrap the mirror + push commands in a PowerShell helper (e.g. `scripts/publish-space.ps1`).
- Use GitHub Actions with [huggingface/space-push](https://github.com/huggingface/space-push) to update the Space on every `main` push.
- Add a Vercel deploy hook so Git commits trigger redeploys without running the CLI locally.

Happy shipping! 🎧

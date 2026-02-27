# Deploy OCR Agent on Render

## Option A: Deploy with Docker (recommended)

This app needs **Tesseract** and system libs, so Docker is the recommended way to deploy on Render.

### 1. Push your code to GitHub

Ensure your repo is on GitHub (or GitLab / Bitbucket connected to Render).

### 2. Create a new Web Service on Render

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**.
2. Connect your repository and select the **ocr_agent** repo.
3. Configure:
   - **Name:** `ocr-agent` (or any name).
   - **Region:** Choose the one closest to your users.
   - **Branch:** `main` (or your default branch).
   - **Runtime:** **Docker** (important).
   - **Dockerfile Path:** Leave empty if it’s in the repo root, or set to `./Dockerfile`.
   - **Instance type:** Free or paid depending on your needs.

4. **Environment variables** (optional):  
   Render sets `PORT` automatically. You can add:
   - `WORKERS` = `2`
   - `TIMEOUT` = `180`

5. Click **Create Web Service**.

Render will build the image from your `Dockerfile` and start the app. Your app will be available at `https://<your-service-name>.onrender.com`.

### 3. (Optional) One-click deploy with Blueprint

If your repo has a `render.yaml` (Blueprint):

1. **New** → **Blueprint**.
2. Connect the repo; Render will read `render.yaml` and create the Web Service.
3. Adjust name/region if needed and deploy.

---

## Option B: Native Python (no Docker)

If you prefer not to use Docker, you’d need to use a [custom buildpack](https://render.com/docs/buildpacks) or a script that installs Tesseract and system deps. This is more involved; Docker is simpler for this stack.

---

## After deploy

- **Health:** Open `https://<your-service>.onrender.com/` — you should see the OCR Agent UI.
- **Timeout:** Free tier has request timeouts; for long OCR jobs consider a paid instance or increasing `TIMEOUT` (e.g. 300).
- **Cold starts:** Free instances spin down after inactivity; the first request after a while may be slow.

## Windows development

The repo’s `requirements.txt` is set up for **Linux** (Render/Docker). On Windows, install PaddlePaddle from the official Windows URL if needed:

```bash
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

Then install the rest: `pip install -r requirements.txt` (and skip or comment out the `paddlepaddle` line in `requirements.txt` if you already installed it as above).

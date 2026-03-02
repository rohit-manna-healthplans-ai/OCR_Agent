# Deploy OCR Agent on Hugging Face Spaces (free)

Follow these steps to put your project on a **Hugging Face Space** so it runs in the cloud for free.

---

## 1. Create a Hugging Face account

- Go to [huggingface.co](https://huggingface.co) and sign up (free).

---

## 2. Create a new Space

1. Click your profile (top right) → **Spaces** → **Create new Space**  
   or go to [huggingface.co/new-space](https://huggingface.co/new-space).

2. Set:
   - **Space name:** e.g. `ocr-agent` (will be `https://huggingface.co/spaces/YOUR_USERNAME/ocr-agent`).
   - **License:** e.g. MIT or Apache 2.0.
   - **SDK:** choose **Docker** (required for this app).
   - **Hardware:** **CPU basic** (free).

3. Click **Create Space**.

---

## 3. Push your code to the Space

Your Space is a Git repo. You can either push from your existing repo or upload files.

### Option A: Push from your machine (recommended)

1. In the Space page, open **Settings** and note the **Clone repo** URL (e.g. `https://huggingface.co/spaces/YOUR_USERNAME/ocr-agent`).

2. On your PC, in your project folder:

   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/ocr-agent
   git push hf main
   ```

   (Use your Space URL and your actual branch name, e.g. `main`.)

3. If HF asks for login, use your **username** and a **User Access Token** (Settings → Access Tokens on HF) as the password.

### Option B: Upload in the browser

1. In the Space, open the **Files** tab → **Add file** → **Upload files**.
2. Upload your whole project (or drag the folder): at least `app/`, `ocr/`, `web/`, `Dockerfile`, `requirements-docker.txt`, `gunicorn_conf.py`, and `README.md` (the one with the YAML at the top).

---

## 4. Make sure these are in the Space repo

- **README.md** – must have the YAML block at the top with `sdk: docker` and `app_port: 7860` (the one in this project is already set).
- **Dockerfile** – at the repo root (uses `requirements-docker.txt` and listens on `PORT`).
- **requirements-docker.txt** – Linux deps (Paddle from PyPI); the Space builds with this.
- **requirements.txt** – only for your local Windows; the Space does **not** use it for the build.

---

## 5. Build and run

1. After you push or upload, Hugging Face will **build** the Docker image (first time can take several minutes).
2. When the build finishes, the Space will **start** and your app will be at:
   - **https://huggingface.co/spaces/YOUR_USERNAME/ocr-agent**
3. Open that URL: you should see the OCR Agent UI. Upload a file and click **Run OCR**.

---

## 6. If something goes wrong

- **Build fails:** Open the **Logs** tab on the Space and check the error (e.g. missing file or failed `pip install`).
- **502 / app not loading:** Check logs; free CPU can be slow to start or run out of memory on heavy OCR. Try with a small image first.
- **Port:** The app is configured to use port **7860** (via `app_port` in README and `PORT` in Dockerfile). Do not change this unless you know what you’re doing.

---

## Summary

| Step | Action |
|------|--------|
| 1 | Sign up at huggingface.co |
| 2 | Create Space → SDK: **Docker**, Hardware: **CPU basic** |
| 3 | Push your code to the Space repo (or upload files) |
| 4 | Ensure README (with YAML), Dockerfile, requirements-docker.txt are in the repo |
| 5 | Wait for build, then open the Space URL and use the app |

Your single **requirements.txt** stays for **local Windows**. The Space uses **requirements-docker.txt** for the Linux Docker build only.

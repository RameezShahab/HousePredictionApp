
# Ames Housing — Deployable Streamlit App

This folder contains a minimal Streamlit app and your dataset. Follow the steps below to run locally and deploy to the web.

## A) Run locally (Windows, macOS, Linux)

1. **Install Python 3.9+** from https://www.python.org/ if you don't have it.
2. **Open a terminal** (Command Prompt/PowerShell on Windows, Terminal on macOS/Linux).
3. **Go to the project folder** (replace the path with where you extracted this folder):
   ```bash
   cd path/to/deployable_app
   ```
4. **Create and activate a virtual environment**:

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   Your browser will open (or go to http://localhost:8501).

## B) Deploy to the web (Streamlit Community Cloud)

1. Create a free account at https://streamlit.io/cloud and **connect your GitHub**.
2. Push this folder to a **new GitHub repository** (name it anything). Quick commands:
   ```bash
   cd path/to/deployable_app
   git init
   echo "__pycache__/
.venv/
.DS_Store" > .gitignore
   git add .
   git commit -m "Initial deployable Streamlit app"
   git branch -M main
   # Create an empty repo on GitHub via the website (copy its URL), then:
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
3. In Streamlit Cloud, click **Create app** → choose your repo → set **Main file path** to `app.py` → **Deploy**.
4. Wait for the build to finish. You’ll get a public URL like `https://<your-app>.streamlit.app` to share.

### Troubleshooting
- If deploy fails, open **Logs** on Streamlit Cloud:
  - Missing package? Add it to `requirements.txt`, commit, and push.
  - File not found? Make sure `AmesHousing.csv` is included (it’s in this folder) or upload a CSV in the app.
  - Notebook magics in `project_script.py`? Remove lines starting with `%` or `!`.

## Files in this project
- `app.py` — Streamlit app UI (loads `AmesHousing.csv` and shows quick explorer)
- `requirements.txt` — Python dependencies
- `AmesHousing.csv` — dataset included for convenience
- `project_script.py` — code extracted from `Project.ipynb` (edit as needed)
- `README.md` — this guide

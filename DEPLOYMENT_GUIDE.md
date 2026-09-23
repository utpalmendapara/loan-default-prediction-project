# 🚀 Complete Free Cloud Deployment Guide
## Loan Default Prediction System (CrediPulse AI)

This guide provides instructions to deploy the **CrediPulse AI** Loan Default Prediction system to top free hosting platforms.

---

## 📌 Prerequisites: Commit & Push Code to GitHub

All deployment configuration files (`Procfile`, `render.yaml`, `Dockerfile`, `requirements.txt`, `flask_app/`) are already prepared in the project.

Run these commands in your project terminal:
```bash
git add .
git commit -m "Complete ML project with Flask web app and deployment configs"
git push origin main
```
Your code is now updated on GitHub: `https://github.com/utpalmendapara/loan-default-prediction-project`.

---

## 🌐 Option 1: Render.com (Recommended — 100% Free & Automatic)

Render is the standard free hosting platform for Python/Flask web apps. It automatically deploys directly from your GitHub repository whenever you push changes.

### Step-by-Step Instructions:
1. **Sign Up / Log In**:
   - Visit [https://render.com](https://render.com) and click **Sign Up** (or log in with your GitHub account).
2. **Create New Web Service**:
   - From the Render Dashboard, click **New +** &rarr; select **Web Service**.
3. **Connect Your GitHub Repository**:
   - Choose **Build and deploy from a Git repository**.
   - Select your repository: `utpalmendapara/loan-default-prediction-project`.
4. **Configure Service Settings**:
   - **Name**: `loan-default-prediction` (or any name you prefer)
   - **Region**: Closest to your users (e.g., *Singapore*, *Frankfurt*, or *Ohio*)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     gunicorn flask_app.app:app
     ```
   - **Instance Type**: Select **Free** ($0 / month).
5. **Deploy**:
   - Click **Create Web Service**.
   - Render will build the environment, install packages, and launch the server in ~2 minutes.
   - Your live site will be accessible globally at:  
     `https://loan-default-prediction.onrender.com`

---

## 🤗 Option 2: Hugging Face Spaces (100% Free & Unlimited Uptime)

Hugging Face Spaces is designed specifically for Machine Learning projects, offering **16 GB RAM + 2 vCPU** for free.

### Step-by-Step Instructions:
1. Go to [https://huggingface.co](https://huggingface.co) and sign in (or create a free account).
2. Click on your profile icon (top right) &rarr; **New Space**.
3. Fill in the details:
   - **Space Name**: `loan-default-prediction`
   - **License**: `mit` or `apache-2.0`
   - **Select Space SDK**: Choose **Docker** (Blank)
   - **Space Hardware**: Free (2 vCPU, 16 GB RAM)
   - **Visibility**: Public
4. Click **Create Space**.
5. Connect your GitHub repository:
   - Under the Space **Settings**, scroll to **GitHub Synced Space** (or push the repository code directly using the provided git commands).
   - Because our repository already has a production-ready `Dockerfile` and `requirements.txt`, Hugging Face will automatically build and serve the application at:  
     `https://huggingface.co/spaces/<your-username>/loan-default-prediction`

---

## 🐍 Option 3: PythonAnywhere (100% Free Forever)

PythonAnywhere is a dedicated Python hosting environment with a free perpetual tier.

### Step-by-Step Instructions:
1. Create a free account at [https://www.pythonanywhere.com](https://www.pythonanywhere.com).
2. Go to the **Consoles** tab and click **Bash**.
3. Clone your GitHub repository:
   ```bash
   git clone https://github.com/utpalmendapara/loan-default-prediction-project.git
   cd loan-default-prediction-project
   ```
4. Install requirements:
   ```bash
   pip install --user -r requirements.txt
   ```
5. Go to the **Web** tab:
   - Click **Add a new web app** &rarr; Select **Manual configuration** &rarr; Choose **Python 3.10** or **3.11**.
   - Set **Source code directory**: `/home/<username>/loan-default-prediction-project`
   - Set **Working directory**: `/home/<username>/loan-default-prediction-project`
6. Edit the **WSGI configuration file** (click the link under the WSGI section):
   Replace all contents with:
   ```python
   import sys
   path = '/home/<username>/loan-default-prediction-project'
   if path not in sys.path:
       sys.path.append(path)

   from flask_app.app import app as application
   ```
7. Click the green **Reload <username>.pythonanywhere.com** button.
8. Your site is live at:  
   `https://<username>.pythonanywhere.com`

---

## 🐳 Option 4: Local or Cloud Docker Deployment

To test the production container locally or deploy to container platforms (Railway, Fly.io, Google Cloud Run):
```bash
# 1. Build the Docker image
docker build -t credipulse-app .

# 2. Run container on port 5000
docker run -d -p 5000:5000 credipulse-app

# 3. Open in browser
http://localhost:5000
```

---

## 🔍 Verification Checklist After Deployment

Once deployed on any platform, verify these URLs:
- **Landing Page**: `https://<your-app-url>/`
- **Loan Evaluation Form**: `https://<your-app-url>/predict`
- **SOP Visualizations**: `https://<your-app-url>/metrics`
- **REST API Endpoint**: `POST https://<your-app-url>/api/predict`

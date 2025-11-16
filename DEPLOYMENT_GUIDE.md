# 🚀 Deployment Guide: Making Your Crop Disease Detection App Live

## Quick Overview

| Platform | Difficulty | Cost | Best For |
|----------|-----------|------|----------|
| **Streamlit Cloud** | ⭐ Easy | Free | Quick deployment, personal projects |
| **Heroku** | ⭐⭐ Medium | Free tier available | Production apps |
| **AWS/Azure/GCP** | ⭐⭐⭐ Advanced | Pay-as-you-go | Enterprise, high traffic |
| **Docker + VPS** | ⭐⭐⭐ Advanced | VPS cost | Full control, custom setup |

---

## Option 1: Streamlit Cloud (Recommended - Easiest & Free)

### Prerequisites
- GitHub account
- Your code pushed to GitHub repository

### Steps

#### 1. Push Your Code to GitHub

```bash
# Initialize git (if not already done)
git init

# Create .gitignore (already exists)
# Make sure model files are in .gitignore

# Add files
git add .
git commit -m "Initial commit: Crop Disease Detection App"

# Create repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/CropDiseaseDetection.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/CropDiseaseDetection`
5. Main file path: `app/main.py`
6. Click "Deploy"

#### 3. Handle Model File (Important!)

Since model files are large, you have two options:

**Option A: Use Git LFS (Large File Storage)**
```bash
# Install Git LFS
# Windows: Download from https://git-lfs.github.com/
# Or: winget install Git.GitLFS

# Initialize Git LFS
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add model/crop_disease_model.h5
git commit -m "Add model with Git LFS"
git push
```

**Option B: Host Model Separately**
- Upload model to cloud storage (Google Drive, Dropbox, AWS S3)
- Update app to download model on first run
- See "Advanced: External Model Storage" section below

#### 4. Update Requirements

Make sure `requirements.txt` includes all dependencies:
```txt
streamlit>=1.28.0
tensorflow>=2.15.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
```

#### 5. Your App is Live!

Streamlit Cloud will provide a URL like:
```
https://your-app-name.streamlit.app
```

---

## Option 2: Heroku Deployment

### Prerequisites
- Heroku account (free tier available)
- Heroku CLI installed

### Steps

#### 1. Create Required Files

**Create `Procfile`:**
```
web: streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
```

**Create `setup.sh`:**
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
```

**Update `requirements.txt`** (add if missing):
```
streamlit>=1.28.0
tensorflow>=2.15.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
```

#### 2. Deploy to Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

**Note:** Heroku free tier has limitations. Model files may need external storage.

---

## Option 3: Docker Deployment (Any Platform)

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t crop-disease-app .

# Run container
docker run -p 8501:8501 crop-disease-app

# Access at http://localhost:8501
```

### Deploy to Cloud with Docker

**AWS ECS, Google Cloud Run, Azure Container Instances:**
- Build Docker image
- Push to container registry
- Deploy to cloud service

---

## Option 4: VPS Deployment (DigitalOcean, Linode, etc.)

### Steps

1. **Set up VPS** (Ubuntu 20.04+ recommended)
2. **Install dependencies:**
```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx
```

3. **Clone repository:**
```bash
git clone https://github.com/YOUR_USERNAME/CropDiseaseDetection.git
cd CropDiseaseDetection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. **Set up systemd service:**
Create `/etc/systemd/system/streamlit-app.service`:
```ini
[Unit]
Description=Streamlit Crop Disease App
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/CropDiseaseDetection
Environment="PATH=/path/to/CropDiseaseDetection/venv/bin"
ExecStart=/path/to/CropDiseaseDetection/venv/bin/streamlit run app/main.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
```

5. **Start service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit-app
sudo systemctl start streamlit-app
```

6. **Set up Nginx reverse proxy:**
Create `/etc/nginx/sites-available/streamlit-app`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

7. **Enable site:**
```bash
sudo ln -s /etc/nginx/sites-available/streamlit-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Advanced: External Model Storage

If model files are too large for Git, host them externally:

### Update `app/main.py` to download model:

```python
import urllib.request
from pathlib import Path

MODEL_URL = "https://your-storage.com/model/crop_disease_model.h5"

def download_model_if_needed():
    """Download model if not present"""
    if not MODEL_PATH.exists():
        st.info("Downloading model... This may take a few minutes.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded!")
```

### Storage Options:

1. **Google Drive:**
   - Upload model to Google Drive
   - Get shareable link
   - Use `gdown` library to download

2. **AWS S3:**
   - Upload to S3 bucket
   - Use `boto3` to download

3. **GitHub Releases:**
   - Upload model as release asset
   - Download via GitHub API

---

## Security Considerations

### 1. Environment Variables

Create `.streamlit/secrets.toml` for sensitive data:
```toml
[secrets]
MODEL_URL = "https://your-secure-storage.com/model.h5"
API_KEY = "your-api-key"
```

### 2. Rate Limiting

Add to `app/main.py`:
```python
import time

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = 0

if time.time() - st.session_state.last_prediction < 2:
    st.warning("Please wait before making another prediction.")
    st.stop()
```

### 3. File Size Limits

Add validation:
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
if uploaded_file.size > MAX_FILE_SIZE:
    st.error("File too large. Please upload images under 10MB.")
    st.stop()
```

---

## Quick Start: Streamlit Cloud (Recommended)

### Fastest Way to Go Live:

1. **Push to GitHub:**
```bash
git add .
git commit -m "Ready for deployment"
git push
```

2. **Deploy on Streamlit Cloud:**
   - Visit: https://share.streamlit.io
   - Connect GitHub
   - Deploy `app/main.py`

3. **Done!** Your app is live in ~5 minutes.

---

## Troubleshooting

### Issue: Model file too large
**Solution:** Use Git LFS or external storage

### Issue: App crashes on startup
**Solution:** Check `requirements.txt` includes all dependencies

### Issue: Slow predictions
**Solution:** 
- Use GPU-enabled cloud instance
- Optimize model (quantization)
- Use TensorFlow Lite

### Issue: Out of memory
**Solution:**
- Reduce batch size
- Use model quantization
- Upgrade cloud instance

---

## Cost Comparison

| Platform | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Streamlit Cloud | ✅ Free (unlimited) | N/A |
| Heroku | ✅ Free (limited hours) | $7+/month |
| AWS | ❌ No free tier | ~$10-50/month |
| DigitalOcean | ❌ No free tier | $6+/month |
| Google Cloud Run | ✅ Free tier | Pay per use |

---

## Recommendation

**For most users:** Start with **Streamlit Cloud** (free, easy, fast)

**For production:** Use **Heroku** or **Docker on cloud platform**

**For enterprise:** Use **AWS/Azure/GCP** with proper infrastructure

---

## Next Steps

1. Choose deployment platform
2. Push code to GitHub
3. Deploy using platform-specific steps
4. Share your live app URL!

**Your app will be accessible worldwide! 🌍**


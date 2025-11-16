# ⚡ Quick Deploy Guide - Get Live in 5 Minutes!

## 🚀 Streamlit Cloud (Easiest - FREE)

### Step 1: Push to GitHub (2 minutes)

```bash
# If you haven't initialized git yet:
git init
git add .
git commit -m "Crop Disease Detection App - Ready for deployment"

# Create a new repository on GitHub.com, then:
git remote add origin https://github.com/YOUR_USERNAME/CropDiseaseDetection.git
git branch -M main
git push -u origin main
```

### Step 2: Handle Model File

**Option A: Use Git LFS (Recommended for large files)**

```bash
# Install Git LFS (if not installed)
# Windows: Download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Track model files
git lfs track "model/*.h5"
git add .gitattributes
git add model/crop_disease_model.h5
git commit -m "Add model with Git LFS"
git push
```

**Option B: Host Model Externally**

If model is too large, upload to:
- Google Drive (use `gdown` to download)
- AWS S3
- GitHub Releases

Then update `app/main.py` to download on first run.

### Step 3: Deploy on Streamlit Cloud (1 minute)

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"Sign in"** → Use GitHub
3. Click **"New app"**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/CropDiseaseDetection`
   - **Branch:** `main`
   - **Main file path:** `app/main.py`
5. Click **"Deploy"**

### Step 4: Wait & Done! (2 minutes)

Streamlit Cloud will:
- Install dependencies from `requirements.txt`
- Build your app
- Provide a live URL like: `https://your-app-name.streamlit.app`

## ✅ That's It!

Your app is now live and accessible worldwide!

---

## 🔧 Troubleshooting

### Model file too large?
- Use Git LFS (see Step 2)
- Or host externally and download in app

### App won't start?
- Check `requirements.txt` has all dependencies
- Check model file is accessible
- Check `class_names.json` exists

### Need help?
- Check `DEPLOYMENT_GUIDE.md` for detailed options
- Streamlit Cloud docs: https://docs.streamlit.io/streamlit-community-cloud

---

## 📱 Share Your App

Once deployed, share your URL:
```
https://your-app-name.streamlit.app
```

**Congratulations! Your Crop Disease Detection app is live! 🎉**


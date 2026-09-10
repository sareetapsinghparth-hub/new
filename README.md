# SkyGuard AI

Public demo for SIH: **anomaly detection on Automatic Weather Stations** (Delhi–NCR).  
This is a Streamlit web app. GitHub itself cannot run it; anyone can open the **live site** after Streamlit Community Cloud is connected to this repo.

## Live website

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub (`sareetapsinghparth-hub`).
2. **Create app** → repository **`sareetapsinghparth-hub/new`** → branch **`main`** → main file **`app.py`**.
3. Streamlit will give a public URL like  
   `https://<app-name>.streamlit.app`  
   Share that link. Anyone can use it; they do not need GitHub.

Until that deploy exists, clone this public repo and run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## What the app shows

- History data the model learned from (AWS T / P / RH)
- OpenStreetMap of Delhi–NCR with four demo stations
- Optional **Live Open-Meteo** current weather (needs internet, no API key)
- SHAP + LIME explanations, suggested corrected readings, maintenance forecast

## Repo

https://github.com/sareetapsinghparth-hub/new

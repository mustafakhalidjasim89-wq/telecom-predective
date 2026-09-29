# Telecom Predictive Maintenance Platform

A production-ready platform combining Streamlit, Gemini Vision, XGBoost predictive scoring, and FAISS-powered RAG for telecom tower maintenance.

## Folder Structure
- `app.py`: Streamlit frontend dashboard.
- `models/`: XGBoost model training and pickle artifacts.
- `vision/`: Gemini Vision API integration for optical site inspections.
- `reports/`: PDF and Excel report generation.
- `rag/`: Document indexing and SOP retrieval system.
- `data/`: Telemetry dataset (`telecom_sites.csv`).

## Getting Started
1. `pip install -r requirements.txt`
2. `python models/train_model.py`
3. Add your `GEMINI_API_KEY` to `.streamlit/secrets.toml`
4. `streamlit run app.py`

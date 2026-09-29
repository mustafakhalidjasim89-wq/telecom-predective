import os
import joblib
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st

from vision.gemini_analyzer import analyze_site_image
from reports.excel_report import generate_excel_report
from reports.pdf_report import generate_pdf_report
from rag.telecom_rag import TelecomRAG

st.set_page_config(
    page_title="Telecom Predictive Maintenance Platform",
    page_icon="📡",
    layout="wide"
)

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = TelecomRAG()
    st.session_state.rag_engine.build_vector_store()

def calculate_health_score(features: dict, failure_prob: float) -> int:
    score = 100.0
    score -= failure_prob * 50
    if features.get("battery_voltage", 48) < 46.0:
        score -= 10
    if features.get("fuel_level", 100) < 25:
        score -= 10
    if features.get("temperature", 25) > 40:
        score -= 10
    if features.get("rectifier_alarm", 0) == 1:
        score -= 10
    return max(0, int(score))

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "xgb_telecom.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

st.title("📡 Telecom Predictive Maintenance Platform")
st.markdown("AI-driven multi-site asset risk scoring, visual defect inspection, and operational SOP retrieval.")

st.sidebar.header("🕹️ Controls & Navigation")
nav = st.sidebar.radio("Go to", ["Multi-Site Dashboard", "Single Site Inspection", "Gemini Vision", "Telecom RAG KB"])

api_key = st.secrets.get("GEMINI_API_KEY", "")

if nav == "Multi-Site Dashboard":
    st.header("🏢 Multi-Site Operations Dashboard")
    
    csv_path = os.path.join("data", "telecom_sites.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        if model:
            features = ["dg_hours", "battery_voltage", "temperature", "fuel_level", "rectifier_alarm"]
            probs = model.predict_proba(df[features])[:, 1]
            df["failure_probability"] = [round(p, 2) for p in probs]
            
            health_scores = []
            for idx, row in df.iterrows():
                hs = calculate_health_score(row.to_dict(), row["failure_probability"])
                health_scores.append(hs)
            df["health_score"] = health_scores
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Monitored Sites", len(df))
            col2.metric("High Risk Sites (>70%)", len(df[df["failure_probability"] > 0.7]))
            col3.metric("Avg Fleet Health", f"{int(df['health_score'].mean())}%")
            col4.metric("Active Rectifier Alarms", int(df["rectifier_alarm"].sum()))
            
            st.markdown("---")
            
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.bar(df, x="site_id", y="failure_probability", color="failure_probability",
                              title="Site Failure Probabilities", color_continuous_scale="Reds")
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = px.scatter(df, x="temperature", y="battery_voltage", size="dg_hours", color="health_score",
                                  title="Temperature vs Voltage (Size = DG Hours)", color_continuous_scale="Viridis")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("📋 Detailed Site Telemetry Table")
            st.dataframe(df, use_container_width=True)

            if st.button("📥 Export Full Fleet Excel Report"):
                report_path = generate_excel_report(df)
                with open(report_path, "rb") as f:
                    st.download_button("Download Excel File", f, file_name="telecom_fleet_report.xlsx")
        else:
            st.warning("XGBoost model file missing. Please run `python models/train_model.py` first.")

elif nav == "Single Site Inspection":
    st.header("🔍 Single Site Risk & Health Scoring")
    
    col_in, col_res = st.columns([1, 1])
    
    with col_in:
        site_id = st.text_input("Site Identifier", "SITE-NEW")
        dg_hours = st.number_input("DG Run Hours", 0, 5000, 1200)
        battery_voltage = st.number_input("Battery Voltage (V)", 30.0, 60.0, 45.2)
        temperature = st.number_input("Shelter Temp (°C)", 10.0, 65.0, 44.0)
        fuel_level = st.slider("Fuel Level (%)", 0, 100, 18)
        rectifier_alarm = st.selectbox("Rectifier Alarm", [0, 1], format_func=lambda x: "Active (1)" if x == 1 else "Normal (0)")
        
    with col_res:
        st.subheader("Model Decision & Assessment")
        if model:
            input_df = pd.DataFrame([{
                "dg_hours": dg_hours,
                "battery_voltage": battery_voltage,
                "temperature": temperature,
                "fuel_level": fuel_level,
                "rectifier_alarm": rectifier_alarm
            }])
            
            prob = float(model.predict_proba(input_df)[0][1])
            health = calculate_health_score(input_df.iloc[0].to_dict(), prob)
            
            st.metric("Failure Probability", f"{prob*100:.1f}%")
            st.metric("Overall Asset Health", f"{health}%")
            
            if prob > 0.6:
                st.error("⚠️ Maintenance Required Urgently: High Failure Probability")
            elif prob > 0.3:
                st.warning("⚡ Priority Maintenance: Moderate Risk Detected")
            else:
                st.success("✅ Site Operating Within Safe Parameters")

            site_summary = {
                "site_id": site_id, "dg_hours": dg_hours, "battery_voltage": battery_voltage,
                "temperature": temperature, "fuel_level": fuel_level, "rectifier_alarm": rectifier_alarm,
                "failure_prob": prob, "health_score": health
            }
            if st.button("📄 Generate & Download PDF Audit Report"):
                pdf_file = generate_pdf_report(site_summary, vision_data={})
                with open(pdf_file, "rb") as f:
                    st.download_button("Download PDF", f, file_name=f"{site_id}_report.pdf")

elif nav == "Gemini Vision":
    st.header("📸 AI Optical Site Inspection (Gemini Vision)")
    
    uploaded_file = st.file_uploader("Upload Site Photo (Tower, Battery, DG, Shelter)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Equipment Inspection Image", width=400)
        
        if st.button("🔎 Run AI Visual Defect Inspection"):
            if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
                st.error("Please configure a valid GEMINI_API_KEY in `.streamlit/secrets.toml`.")
            else:
                with st.spinner("Analyzing image via Gemini Vision..."):
                    results = analyze_site_image(image, api_key)
                    
                st.subheader("Inspection Results")
                if "error" in results and len(results) == 2:
                    st.error(results["error"])
                else:
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.write(f"**Corrosion Score:** {results.get('corrosion_score')}/100")
                        st.write(f"**Housekeeping Score:** {results.get('housekeeping_score')}/100")
                        st.write(f"**Battery Swelling:** {'🔴 Yes' if results.get('battery_swelling') else '🟢 None'}")
                    with v_col2:
                        st.write(f"**Oil Leakage:** {'🔴 Yes' if results.get('oil_leakage') else '🟢 None'}")
                        st.write(f"**Missing Bolts:** {'🔴 Yes' if results.get('missing_bolts') else '🟢 None'}")
                        st.write(f"**Cable Damage:** {'🔴 Yes' if results.get('cable_damage') else '🟢 None'}")
                    
                    st.info(f"**AI Findings Summary:** {results.get('summary_notes')}")

elif nav == "Telecom RAG KB":
    st.header("📚 Telecom SOP & Manual RAG Assistant")
    
    query = st.text_input("Ask a question about maintenance manual or standard operating procedures:", 
                          "How to fix rectifier alarm?")
    
    if st.button("Search Knowledge Base"):
        with st.spinner("Searching manuals..."):
            response = st.session_state.rag_engine.query(query)
            st.markdown("### Relevant Manual Excerpt & Guidelines:")
            st.info(response)

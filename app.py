import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Employee Promotion Prediction",
    page_icon="👔",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .badge-promoted {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .badge-not-promoted {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(245, 158, 11, 0.3);
    }
    .metric-val {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .card-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">👔 Employee Promotion Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated HR analytics powered by <b>RandomForestClassifier</b> to evaluate employee promotion eligibility and readiness.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "models" / "employee_promotion_model.pkl"
csv_path = BASE_DIR / "data" / "employee_promotion.csv"
chart_path = BASE_DIR / "outputs" / "feature_importance.png"

if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(str(model_path))

model = get_model()

tab1, tab2, tab3 = st.tabs(["🔮 Employee Evaluation & Prediction", "📊 Model Insights & Feature Importance", "📋 HR Records Dataset"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Employee Profile Information")
        
        # Quick Presets
        preset = st.selectbox(
            "⚡ Load Pre-configured Employee Persona",
            ["Custom Input", "🌟 High-Performing Senior (Ready for Promotion)", "🌱 Promising Rising Talent", "⏳ Tenured Average Performer"]
        )
        
        # Default values based on preset
        if preset == "🌟 High-Performing Senior (Ready for Promotion)":
            def_age, def_comp, def_role, def_perf, def_train, def_edu, def_promo, def_sat = 35, 7, 3, 5, 80, 3, 2, 5
        elif preset == "🌱 Promising Rising Talent":
            def_age, def_comp, def_role, def_perf, def_train, def_edu, def_promo, def_sat = 27, 3, 2, 4, 60, 2, 0, 4
        elif preset == "⏳ Tenured Average Performer":
            def_age, def_comp, def_role, def_perf, def_train, def_edu, def_promo, def_sat = 45, 12, 6, 3, 15, 2, 1, 3
        else:
            def_age, def_comp, def_role, def_perf, def_train, def_edu, def_promo, def_sat = 31, 5, 2, 4, 45, 3, 1, 4
        
        with st.form("promotion_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.slider("Employee Age", 20, 65, def_age)
                years_at_company = st.slider("Years at Company", 0, 35, def_comp)
                years_in_current_role = st.slider("Years in Current Role", 0, 20, def_role)
                performance_score = st.select_slider(
                    "Performance Score (Rating)",
                    options=[1, 2, 3, 4, 5],
                    value=def_perf,
                    format_func=lambda x: f"{x} - " + {1: "Poor", 2: "Needs Improvement", 3: "Meets Expectations", 4: "Exceeds Expectations", 5: "Outstanding"}[x]
                )
            with col_b:
                training_hours = st.slider("Training Hours Completed", 0, 150, def_train)
                education_level = st.selectbox(
                    "Education Level",
                    options=[1, 2, 3, 4],
                    index=def_edu - 1,
                    format_func=lambda x: {1: "1 - High School / Diploma", 2: "2 - Bachelor's Degree", 3: "3 - Master's Degree", 4: "4 - Doctorate / Ph.D."}[x]
                )
                previous_promotions = st.slider("Previous Promotions", 0, 5, def_promo)
                job_satisfaction = st.select_slider(
                    "Job Satisfaction",
                    options=[1, 2, 3, 4, 5],
                    value=def_sat,
                    format_func=lambda x: f"{x} / 5 " + ("⭐" * x)
                )
                
            submit_btn = st.form_submit_button("🚀 Evaluate Promotion Eligibility", use_container_width=True)
            
    with col_result:
        st.subheader("Evaluation Results")
        if submit_btn:
            employee = pd.DataFrame([{
                "age": age,
                "years_at_company": years_at_company,
                "years_in_current_role": years_in_current_role,
                "performance_score": performance_score,
                "training_hours": training_hours,
                "education_level": education_level,
                "previous_promotions": previous_promotions,
                "job_satisfaction": job_satisfaction
            }])
            
            prediction = model.predict(employee)[0]
            probability = model.predict_proba(employee)[0][1]
            
            if prediction == 1:
                st.markdown(f"""
                <div class="badge-promoted">
                    <div style="font-size: 1rem; opacity: 0.9;">Recommendation</div>
                    <div class="metric-val">✅ RECOMMENDED FOR PROMOTION</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Promotion Likelihood: {probability:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="badge-not-promoted">
                    <div style="font-size: 1rem; opacity: 0.9;">Recommendation</div>
                    <div class="metric-val">⏳ NOT RECOMMENDED YET</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Promotion Likelihood: {probability:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown("### 📊 Probability Breakdown")
            st.progress(float(probability))
            col_p1, col_p2 = st.columns(2)
            col_p1.metric("Likelihood of Promotion", f"{probability:.2%}")
            col_p2.metric("Retention / Non-Promotion", f"{(1 - probability):.2%}")
            
            st.markdown("### 🎯 HR Readiness Signals")
            checks = [
                ("High Performance Score (>= 4)", performance_score >= 4),
                ("Adequate Company Tenure (>= 2 years)", years_at_company >= 2),
                ("Dedicated Training Commitment (>= 40 hrs)", training_hours >= 40),
                ("Advanced Education (Degree+)", education_level >= 2),
                ("High Job Satisfaction (>= 3)", job_satisfaction >= 3)
            ]
            for label, passed in checks:
                icon = "✅" if passed else "⚠️"
                st.write(f"{icon} **{label}**")
                
            with st.expander("🔍 Raw Model Features Payload"):
                st.json(employee.to_dict(orient="records")[0])
        else:
            st.info("👈 Fill in employee metrics and click **'Evaluate Promotion Eligibility'**.")

with tab2:
    st.subheader("Random Forest Feature Importance")
    if chart_path.exists():
        st.image(str(chart_path), caption="Top Predictive Factors for Promotions", use_container_width=True)
    else:
        st.info("Visual chart not found. Run 'python train_model.py' to generate.")

with tab3:
    st.subheader("Training Dataset (employee_promotion.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Employee Records", f"{len(df):,}")
        col_m2.metric("Overall Promotion Rate", f"{df['promoted'].mean():.1%}")
        col_m3.metric("Evaluated Features", len(df.columns) - 1)
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("Dataset file not found.")

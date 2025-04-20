import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import json
import numpy as np


# Load artifacts
model = joblib.load('models/alzheimer_model.pkl')
evaluation_report = joblib.load('reports/evaluation_report.pkl')  # Add this line

feature_mappings = joblib.load('models/feature_mappings.pkl')
selected_features = joblib.load('models/selected_features.pkl')
explainer = shap.TreeExplainer(model)

# App configuration
st.set_page_config(
    page_title="Alzheimer's Risk Assessment",
    
    page_icon="🧠",
    layout="wide"
)

# Sidebar with user guidance
with st.sidebar:
    st.header("Clinical Guidance")
    st.markdown("""
    **Input Interpretation:**
    - APOE-ε4: 0=No alleles, 1=One allele, 2=Two alleles
    - Activity Levels: 1(Sedentary) → 5(Athlete)
    - Sleep Quality: 1(Poor) → 5(Excellent)
    """)
    
    st.divider()
    st.markdown("**Model Information**")
    st.caption(f"• Algorithm: {model.__class__.__name__}")
    try:
        config = json.loads(model.__dict__.get('_Booster').save_config())
        version = config['learner']['gradient_booster']['name']
    except (AttributeError, KeyError) as e:
        version = "1.0.0 (default)"
        st.caption(f"• Version: {version}")
    st.caption(f"• Trained: {pd.Timestamp('now').strftime('%Y-%m-%d')}")
    st.caption(f"• Features: {len(selected_features)} biomarkers")


# Main interface
st.title('Alzheimer\'s Disease Risk Assessment Tool')

st.info("""
**🧠 Alzheimer's Disease Risk Assessment Tool**  

This clinical decision support system estimates dementia risk using:  
✅ Validated biomarkers (APOE-ε4 status, age)  
✅ Modifiable lifestyle factors (activity, sleep, alcohol)  
✅ Family history  

**How to Use:**  
1. Enter data in all sections  
2. Click "Calculate Risk" for:  
   - Binary risk classification (Low/High)  
   - Probability percentage  
   - SHAP analysis of contributing factors  
3. Review evidence-based mitigation strategies  

*Note: Results should inform—not replace—clinical judgment.*  
""", icon="ℹ️")
# ⭐️ NEW METRICS SECTION ⭐️
with st.expander("🔍 Model Performance Overview", expanded=True):
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        st.metric("Cross-Val Accuracy", 
                f"{evaluation_report['cross_val']['mean_accuracy']*100:.1f}%",
                delta=f"±{evaluation_report['cross_val']['std_deviation']*100:.1f}%")
    
    with col_metrics2:
        st.metric("Test Set Accuracy", 
                f"{evaluation_report['holdout_test']['accuracy']*100:.1f}%")
    
    with col_metrics3:
        gap = evaluation_report['overfitting_check']['accuracy_gap']
        st.metric("Train-Test Gap", 
                f"{gap*100:.1f}%", 
                delta="Good" if gap < 0.05 else "Monitor")

# Input section with enhanced UX
with st.form("patient_inputs"):
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        st.subheader("Biomarkers")
    inputs = {
        'Age': st.slider('Age', 50, 100, 65,
                       help="Patient age in years"),
        'Gender': st.radio(  # Add this input
            'Gender',
            options=feature_mappings['Gender'].keys(),
            horizontal=True,
            help="Biological sex (for genetic risk calculation)"
        ),
        'Genetic Risk Factor (APOE-ε4 allele)': st.selectbox(
            'APOE-ε4 Alleles',
            options=feature_mappings['Genetic Risk Factor (APOE-ε4 allele)'].keys(),
            help="Genetic testing results"
        )
        }

    with col2:
        st.subheader("Lifestyle Factors")
        inputs.update({
            'Physical Activity Level': st.select_slider(
                'Physical Activity',
                options=feature_mappings['Physical Activity Level'].keys(),
                help="Weekly exercise frequency"
            ),
            'Sleep Quality': st.select_slider(
                'Sleep Quality',
                options=feature_mappings['Sleep Quality'].keys(),
                help="Average nightly sleep quality"
            ),
            'Alcohol Consumption': st.select_slider(
            'Alcohol Intake',
            options=feature_mappings['Alcohol Consumption'].keys(),
            help="Weekly alcohol consumption"
        )
        })

    with col3:
        st.subheader("Family History")
        inputs.update({
            'Family History of Alzheimer’s': st.radio(
                'Family History',
                options=feature_mappings['Family History of Alzheimer’s'].keys(),
                horizontal=True,
                help="First-degree relatives with diagnosis"
            )
        })
    
    if st.form_submit_button("Calculate Risk"):
        # Processing logic
        processed_data = {
        # Numerical features (direct assignment)
        'Age': inputs['Age'],
        'Alcohol Consumption': feature_mappings['Alcohol Consumption'][inputs['Alcohol Consumption']],
        
        # Categorical features (mapped values)
        'Gender': feature_mappings['Gender'][inputs['Gender']],
        'Physical Activity Level': feature_mappings['Physical Activity Level'][inputs['Physical Activity Level']],
        'Sleep Quality': feature_mappings['Sleep Quality'][inputs['Sleep Quality']],
        'Family History of Alzheimer’s': feature_mappings['Family History of Alzheimer’s'][inputs['Family History of Alzheimer’s']],
        'Genetic Risk Factor (APOE-ε4 allele)': feature_mappings['Genetic Risk Factor (APOE-ε4 allele)'][inputs['Genetic Risk Factor (APOE-ε4 allele)']]
    }

        # Calculate interaction features EXACTLY as in train.py
        processed_data['Age_Alcohol'] = processed_data['Age'] * processed_data['Alcohol Consumption']
        processed_data['Physical_Sleep'] = processed_data['Physical Activity Level'] * processed_data['Sleep Quality']
        processed_data['Stress_Depression'] = 0  # Placeholder since stress/depression not in UI

        # Create DataFrame with CORRECT feature order matching training
        input_df = pd.DataFrame([processed_data])[selected_features].astype(float)
        
        # Generate predictions
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]  # Probability of high risk
        shap_values = explainer.shap_values(input_df)
        
        # Results visualization
        risk_color = "#ff4b4b" if prediction == 1 else "#2ecc71"
        
        with st.container():
            st.subheader("Clinical Risk Assessment")
            col_res1, col_res2 = st.columns([1,2])
            
            with col_res1:
                st.markdown(f"""
                <div style="border-left: 5px solid {risk_color}; padding: 1rem;">
                    <h3 style="color:{risk_color};">{'High Risk' if prediction else 'Low Risk'}</h3>
                    <p>Probability: {proba*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col_res2:
                with st.spinner('Analyzing feature impacts...'):
                    # Generate SHAP values
                    shap_values = explainer(input_df)
                    
                    # Create waterfall plot
                    fig, ax = plt.subplots()
                    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                    plt.tight_layout()
                    
                    # Display plot
                    st.pyplot(fig)
                    st.caption("How each feature contributes to the risk prediction")

        
        # Lifestyle recommendations
        st.subheader("Risk Mitigation Strategies")
        if prediction == 1:
            st.markdown("""
            - **Increase physical activity** to 150+ minutes/week moderate exercise
            - **Improve sleep hygiene** targeting 7-8 hours/night
            - **Cognitive training** 3x/week (e.g., puzzles, language learning)
            """)
        else:
            st.markdown("""
            - **Maintain current healthy habits**
            - **Annual cognitive screening** recommended
            - **Monitor genetic risk factors**
            """)

# Accessibility features
st.markdown("""
<style>
/* High contrast mode */
@media (prefers-contrast: high) {
    body { font-size: 18px !important; }
    .stSlider>div>div>div>div { background: white !important; }
}
</style>
""", unsafe_allow_html=True)
# Alzheimer's Disease Risk Assessment Tool

A clinical decision support system that estimates Alzheimer's disease risk using machine learning and established biomarkers. Provides explainable AI insights for healthcare professionals.

**Disclaimer**: This research prototype is not intended for clinical use. Always consult qualified medical professionals for diagnostic decisions.

## Key Features

- 🧠 **Risk Prediction Engine**
  - XGBoost model trained on synthetic patient data
  - Incorporates genetic, lifestyle, and demographic factors
  - SHAP waterfall plots for explainable predictions

- 🏥 **Clinical Interface**
  - Streamlit-based web interface
  - Interactive sliders/selectors for patient data
  - Risk stratification (Low/High) with probability scores
  - HIPAA-compliant data handling

- 🔬 **Model Transparency**
  - Detailed performance metrics
  - Feature importance analysis
  - Interaction term calculations (Age×Alcohol, Physical×Sleep)
  - Accessibility features for high-contrast viewing

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone repository:
```bash
git clone https://github.com/your-org/alzheimer-risk-tool.git
cd alzheimer-risk-tool
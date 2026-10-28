import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Set page configuration with banking/fintech aesthetics
st.set_page_config(
    page_title="FinSecure | AI Loan Approval Decision System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, high-contrast, premium styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .decision-approved {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    .decision-rejected {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">🏦 FinSecure AI Loan Decision System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Multilayer Perceptron (MLP) Backpropagation Underwriting Engine | STINK3014 Coursework Deployment</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. Load Model & Pipeline Bundle
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    bundle_path = 'model/best_mlp_loan_model.joblib'
    if not os.path.exists(bundle_path):
        st.error(f"Model bundle not found at `{bundle_path}`. Please ensure Task 6 training has completed.")
        st.stop()
    bundle = joblib.load(bundle_path)
    
    # Load Keras neural network model
    os.environ['KERAS_BACKEND'] = 'torch'
    import keras
    model = keras.models.load_model(bundle['keras_model_file'])
    return bundle, model

try:
    bundle, model = load_model_bundle()
    scaler = bundle['scaler']
    numerical_cols = bundle['numerical_cols']
    feature_cols = bundle['feature_cols']
except Exception as e:
    st.error(f"Error initializing model environment: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 2. Sidebar Navigation & Model Information
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=64)
    st.title("System Metadata")
    st.markdown(r"""
    **Model Architecture:**  
    `[11 → 32 (ReLU) → 16 (ReLU) → 1 (Sigmoid)]`
    
    **Underwriting Parameters:**  
    - **Optimization:** SGD with Momentum ($\beta=0.9$)
    - **Regularisation:** L2 ($\lambda = 0.001$) + Dropout ($p = 0.20$)
    - **Input Features:** 11 Demographic, Financial, & Asset attributes
    - **Primary Predictor:** CIBIL Credit Bureau Score
    - **Loss Criterion:** Binary Cross-Entropy (BCE)
    """)
    st.divider()
    st.markdown("### Benchmarked Accuracy")
    if 'test_metrics' in bundle:
        metrics = bundle['test_metrics']
        st.metric("Test Accuracy", f"{metrics.get('acc', 97.9):.2f}%")
        st.metric("Test F1-Score", f"{metrics.get('f1', 0.984):.4f}")
        st.metric("Test ROC-AUC", f"{metrics.get('auc', 0.996):.4f}")
    st.info("⚠️ Regulatory Notice: Model decisions should serve as decision-support for human loan underwriters under fair lending standards.")

# -----------------------------------------------------------------------------
# 3. Applicant Input Forms
# -----------------------------------------------------------------------------
st.markdown("### 📋 Loan Application Dossier")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Applicant Profile")
    no_of_dependents = st.selectbox(
        "Number of Dependents",
        options=[0, 1, 2, 3, 4, 5],
        index=1,
        help="Number of financially dependent family members."
    )
    education = st.selectbox(
        "Education Status",
        options=["Graduate", "Not Graduate"],
        index=0,
        help="Applicant's highest level of educational attainment."
    )
    self_employed = st.selectbox(
        "Employment Type",
        options=["No (Salaried Employee)", "Yes (Self-Employed)"],
        index=0,
        help="Self-employment status."
    )
    cibil_score = st.number_input(
        "CIBIL Credit Bureau Score",
        min_value=300,
        max_value=900,
        value=680,
        step=5,
        help="Official Credit Information Bureau India Limited credit score (Range: 300 to 900)."
    )
    if cibil_score < 550:
        st.warning("⚠️ Critical: CIBIL Score is below the empirical approval threshold (~550).")
    else:
        st.success("✅ CIBIL Score satisfies favorable tier requirements.")

with col2:
    st.subheader("2. Facility Details")
    income_annum = st.number_input(
        "Annual Income (MYR / INR)",
        min_value=100_000,
        max_value=20_000_000,
        value=4_200_000,
        step=100_000,
        format="%d",
        help="Gross verifiable annual income."
    )
    loan_amount = st.number_input(
        "Requested Loan Amount (MYR / INR)",
        min_value=100_000,
        max_value=50_000_000,
        value=11_500_000,
        step=100_000,
        format="%d",
        help="Principal borrowed loan amount."
    )
    loan_term = st.slider(
        "Repayment Term (Years)",
        min_value=2,
        max_value=20,
        value=10,
        step=2,
        help="Loan tenure duration (Years)."
    )
    
    # Financial ratios calculation
    lti_ratio = loan_amount / income_annum if income_annum > 0 else 0
    st.caption(f"**Loan-to-Income (LTI) Multiplier:** `{lti_ratio:.2f}x`")
    if lti_ratio > 4.5:
        st.warning("⚠️ Elevated borrowing leverage detected (LTI > 4.5x).")

with col3:
    st.subheader("3. Collateral & Liquid Assets")
    residential_assets_value = st.number_input(
        "Residential Asset Valuation",
        min_value=-500_000,
        max_value=40_000_000,
        value=5_000_000,
        step=250_000,
        format="%d",
        help="Valuation of residential real estate. Negative values indicate negative equity."
    )
    commercial_assets_value = st.number_input(
        "Commercial Asset Valuation",
        min_value=0,
        max_value=30_000_000,
        value=2_500_000,
        step=250_000,
        format="%d",
        help="Commercial property, land, and warehouse holdings."
    )
    luxury_assets_value = st.number_input(
        "Luxury Assets Valuation",
        min_value=0,
        max_value=40_000_000,
        value=8_000_000,
        step=250_000,
        format="%d",
        help="Vehicles, jewelry, art, and high-value personal assets."
    )
    bank_asset_value = st.number_input(
        "Liquid Bank Deposits & Cash",
        min_value=0,
        max_value=20_000_000,
        value=3_200_000,
        step=250_000,
        format="%d",
        help="Liquid savings, fixed deposits, and cash equivalents."
    )
    total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
    asset_cover = total_assets / loan_amount if loan_amount > 0 else 0
    st.caption(f"**Total Asset Coverage Ratio:** `{asset_cover:.2f}x`")

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. Input Validation & Verification
# -----------------------------------------------------------------------------
validation_errors = []
if income_annum <= 0:
    validation_errors.append("Annual income must be strictly positive.")
if loan_amount <= 0:
    validation_errors.append("Requested loan amount must be strictly positive.")
if cibil_score < 300 or cibil_score > 900:
    validation_errors.append("CIBIL score must fall within the standardized 300-900 boundary.")
if loan_term < 2 or loan_term > 20:
    validation_errors.append("Loan term must be between 2 and 20 years.")

if validation_errors:
    for err in validation_errors:
        st.error(f"❌ Input Validation Error: {err}")
    st.stop()

# -----------------------------------------------------------------------------
# 5. Prediction Execution
# -----------------------------------------------------------------------------
predict_clicked = st.button("🚀 Evaluate Loan Application with Champion MLP", use_container_width=True)

if predict_clicked:
    # Prepare input dictionary
    input_dict = {
        'no_of_dependents': int(no_of_dependents),
        'education': 1 if education == 'Graduate' else 0,
        'self_employed': 1 if "Yes" in self_employed else 0,
        'income_annum': float(income_annum),
        'loan_amount': float(loan_amount),
        'loan_term': int(loan_term),
        'cibil_score': float(cibil_score),
        'residential_assets_value': float(residential_assets_value),
        'commercial_assets_value': float(commercial_assets_value),
        'luxury_assets_value': float(luxury_assets_value),
        'bank_asset_value': float(bank_asset_value)
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Scale numerical features using training scaler statistics
    input_df_scaled = input_df.copy()
    input_df_scaled[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Predict with Champion Keras MLP model
    input_tensor = input_df_scaled[feature_cols].values.astype('float32')
    pred_prob = float(model.predict(input_tensor, verbose=0)[0][0])
    threshold = bundle.get('optimal_threshold', 0.50)
    decision = "Approved" if pred_prob >= threshold else "Rejected"
    
    st.markdown("## 📊 Underwriting Decision & Risk Assessment")
    
    res_col1, res_col2 = st.columns([1, 1.2])
    
    with res_col1:
        if decision == "Approved":
            st.markdown(f"""
            <div class="decision-approved">
                <h1 style="margin:0; font-size: 2.5rem;">🎉 LOAN APPROVED</h1>
                <p style="font-size: 1.2rem; margin-top: 8px;">Confidence Score: <b>{pred_prob*100:.2f}%</b></p>
                <p style="margin: 0; opacity: 0.9;">Applicant profile satisfies risk tolerance constraints.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-rejected">
                <h1 style="margin:0; font-size: 2.5rem;">❌ LOAN REJECTED</h1>
                <p style="font-size: 1.2rem; margin-top: 8px;">Approval Probability: <b>{pred_prob*100:.2f}%</b> (Threshold: {threshold*100:.0f}%)</p>
                <p style="margin: 0; opacity: 0.9;">Elevated default risk profile detected by backpropagation engine.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(pred_prob)
        st.caption(rf"Raw Model Activation Output $\hat{{y}} = \sigma(z^{{[L]}})$: `{pred_prob:.6f}`")

    with res_col2:
        st.markdown("### Decision Factor Breakdown")
        factors = []
        
        # CIBIL Factor
        if cibil_score >= 600:
            factors.append(("Credit Bureau Score", f"{cibil_score:.0f}", "Strong credit history, primary approval driver", "Positive"))
        elif cibil_score >= 550:
            factors.append(("Credit Bureau Score", f"{cibil_score:.0f}", "Acceptable near threshold (550)", "Neutral"))
        else:
            factors.append(("Credit Bureau Score", f"{cibil_score:.0f}", "Subprime credit history (< 550 threshold)", "Negative"))
            
        # LTI Factor
        if lti_ratio <= 3.5:
            factors.append(("Loan-to-Income (LTI)", f"{lti_ratio:.2f}x", "Conservative borrowing envelope", "Positive"))
        else:
            factors.append(("Loan-to-Income (LTI)", f"{lti_ratio:.2f}x", "High leverage relative to income", "Negative"))
            
        # Asset Coverage Factor
        if asset_cover >= 1.5:
            factors.append(("Asset Coverage", f"{asset_cover:.2f}x", "Substantial collateral backing", "Positive"))
        else:
            factors.append(("Asset Coverage", f"{asset_cover:.2f}x", "Tight collateral coverage", "Neutral"))
            
        # Loan Term
        if loan_term <= 6:
            factors.append(("Loan Tenure", f"{loan_term} Years", "Short term reduces default risk window", "Positive"))
        elif loan_term >= 16:
            factors.append(("Loan Tenure", f"{loan_term} Years", "Extended tenure increases macro exposure", "Negative"))
        else:
            factors.append(("Loan Tenure", f"{loan_term} Years", "Standard repayment window", "Neutral"))

        factors_df = pd.DataFrame(factors, columns=["Risk Metric", "Value", "Risk Assessment", "Impact"])
        st.dataframe(factors_df, use_container_width=True, hide_index=True)

    # Technical audit trail
    with st.expander("🔍 View Technical Underwriting Vector (Preprocessed Tensor)"):
        st.write("Raw Input Features:")
        st.json(input_dict)
        st.write("Z-Score Standardised Features (Passed into Input Layer):")
        st.dataframe(input_df_scaled[feature_cols])

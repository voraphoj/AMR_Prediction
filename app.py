import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="Staph Oxacillin Predictor", layout="wide")

@st.cache_resource
def load_assets():
    model = xgb.Booster()
    model.load_model("staph_oxa_demo_xgb_model.json")
    return model, model.feature_names

model, feature_names = load_assets()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- 2. HEADER & RESET ---
h1, h2 = st.columns([8, 2])
with h1:
    st.title("🧫 Oxacillin Resistance Demo")
with h2:
    st.button("🔄 Reset All", on_click=reset_app)

# --- 3. INPUT SECTIONS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Info")
    sex = st.radio("Sex", ["Female (0)", "Male (1)"], horizontal=True, key="sex")
    sex_val = 1 if "Male" in sex else 0
    
    st.write("**Age**")
    y = st.number_input("Years", min_value=0, step=1, key="age_y")
    m = st.number_input("Months", min_value=0, max_value=11, key="age_m")
    d = st.number_input("Days", min_value=0, max_value=30, key="age_d")
    age_days = (y * 365) + (m * 30) + d

with col2:
    st.subheader("Clinical Data")
    sample = st.selectbox("Sample Type", ["Blood", "Pus", "Sputum", "Endotracheal_aspirate", "Urine", "Other"], key="sample")
    
    pos_min = 0
    if sample == "Blood":
        p_d = st.number_input("Positivity (Days)", min_value=0, key="pos_d")
        p_h = st.number_input("Hours", min_value=0, max_value=23, key="pos_h")
        p_m = st.number_input("Minutes", min_value=0, max_value=59, key="pos_m")
        pos_min = (p_d * 1440) + (p_h * 60) + p_m

st.divider()

# --- 4. ANTIBIOTICS ---
st.subheader("Antibiotics (Past 3 Months)")
if 'abx_list' not in st.session_state:
    st.session_state.abx_list = []

def add_abx():
    st.session_state.abx_list.append({"class": "BLBI", "dur": 0, "last": 0})

st.button("➕ Add Antibiotic Row", on_click=add_abx)

abx_classes = ["BLBI", "BI", "Glycopeptide", "Ceph12", "Sulfa", "Aminoglycoside", "Quinolone", 
               "Macrolide", "Metronidazole", "Penicillin", "Carbapenem", "Ceph34", 
               "Tetracycline", "Lincosamide", "Fusidic", "Polymyxin", "Fosfomycin", 
               "Oxazolidinone", "Monobactam"]

abx_features = {cls: 0.0 for cls in abx_classes}

for i, entry in enumerate(st.session_state.abx_list):
    c1, c2, c3 = st.columns(3)
    entry['class'] = c1.selectbox(f"Class #{i+1}", abx_classes, key=f"c_{i}")
    entry['dur'] = c2.number_input(f"Duration (days) #{i+1}", min_value=0, key=f"d_{i}")
    entry['last'] = c3.number_input(f"Days since last dose #{i+1}", min_value=0, key=f"l_{i}")
    abx_features[entry['class']] = entry['dur'] * np.exp(-0.05 * entry['last'])

st.divider()

# --- 5. ADMISSION ---
c_adm1, c_adm2 = st.columns(2)
adm_yes = c_adm1.toggle("Current Admission?", key="adm_t")
adm_val = c_adm1.number_input("Admission days", min_value=0, key="adm_v") if adm_yes else 0

prev_yes = c_adm2.toggle("Previous Admission?", key="prev_t")
prev_val = c_adm2.number_input("Previous discharge (days)", min_value=0, value=9999, key="prev_v") if prev_yes else 9999

# --- 6. CALCULATION & OUTPUT ---
st.divider()

# Important: This button is outside of any conditional logic 
# to ensure it always appears.
if st.button("🚀 RUN PREDICTION", type="primary", use_container_width=True):
    # Prepare Data
    input_dict = {
        "Sex": sex_val, "Age": age_days, "Positive_MIN": pos_min,
        "Admission_to_Receive": adm_val, "Discharge_to_Admission": prev_val,
        "Blood": 1 if sample == "Blood" else 0,
        "Pus": 1 if sample == "Pus" else 0,
        "Sputum": 1 if sample == "Sputum" else 0,
        "Endotracheal_aspirate": 1 if sample == "Endotracheal_aspirate" else 0,
        "Urine": 1 if sample == "Urine" else 0,
        "Other": 1 if sample == "Other" else 0
    }
    input_dict.update(abx_features)
    
    df_final = pd.DataFrame([input_dict])[feature_names]
    
    # Run Model
    dmat = xgb.DMatrix(df_final)
    prob = model.predict(dmat)[0]
    
    # Results Display
    st.balloons()
    st.markdown(f"### Predicted Probability of Resistance: **{prob*100:.2f}%**")
    
    # SHAP Plot
    st.subheader("Feature Influence (Probability Scale)")
    explainer = shap.TreeExplainer(model)
    shap_obj = explainer(df_final)
    
    v = shap_obj.values[0]
    base_lo = shap_obj.base_values[0]
    
    # Manual Probability Transformation
    p_contribs = np.zeros_like(v)
    curr_lo = base_lo
    for i in range(len(v)):
        p_before = sigmoid(curr_lo)
        curr_lo += v[i]
        p_contribs[i] = sigmoid(curr_lo) - p_before
        
    # Clean Labels for Plot
    label_map = {"Positive_MIN": "Time to Positivity", "Admission_to_Receive": "Admission Days"}
    clean_labels = [label_map.get(f, f.replace("_", " ")) for f in feature_names]

    prob_exp = shap.Explanation(
        values=p_contribs, 
        base_values=sigmoid(base_lo), 
        data=df_final.iloc[0], 
        feature_names=clean_labels
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(prob_exp, show=False)
    st.pyplot(fig)

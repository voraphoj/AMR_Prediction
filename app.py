import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="Staph Oxacillin Predictor", layout="wide")

@st.cache_resource
def load_model_assets():
    model = xgb.Booster()
    model.load_model("staph_oxa_demo_xgb_model.json")
    return model, model.feature_names

model, feature_names = load_model_assets()

# --- NEW: RESET FUNCTION ---
def reset_app():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# --- 2. USER INTERFACE ---
header_col1, header_col2 = st.columns([8, 2])
with header_col1:
    st.title("🧫 Oxacillin Resistance Demo")
with header_col2:
    st.button("🔄 Reset All Fields", on_click=reset_app, help="Clear all inputs and start over")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Info")
    # Note: We use 'key' parameters to ensure the values are cleared on reset
    sex = st.radio("Sex", ["Female (0)", "Male (1)"], horizontal=True, key="sex_radio")
    sex_val = 1 if "Male" in sex else 0
    
    st.write("**Age**")
    y = st.number_input("Years", min_value=0, step=1, value=0, key="age_y")
    m = st.number_input("Months", min_value=0, max_value=11, step=1, key="age_m")
    d = st.number_input("Days", min_value=0, max_value=30, step=1, key="age_d")
    age_days = (y * 365) + (m * 30) + d

with col2:
    st.subheader("Clinical Data")
    sample = st.selectbox("Sample Type", ["Blood", "Pus", "Sputum", "Endotracheal_aspirate", "Urine", "Other"], key="sample_sel")
    
    pos_min = 0
    if sample == "Blood":
        p_d = st.number_input("Time to positivity (Days)", min_value=0, key="pos_d_in")
        p_h = st.number_input("Hours", min_value=0, max_value=23, key="pos_h_in")
        p_m = st.number_input("Minutes", min_value=0, max_value=59, key="pos_m_in")
        pos_min = (p_d * 1440) + (p_h * 60) + p_m
    else:
        st.info("Time to positivity disabled (Non-Blood sample).")

# --- 3. ANTIBIOTICS (Dynamic Rows) ---
st.divider()
st.subheader("Antibiotics (Past 3 Months)")

if 'abx_list' not in st.session_state:
    st.session_state.abx_list = []

def add_row():
    st.session_state.abx_list.append({"class": "BLBI", "dur": 0, "last": 0})

st.button("➕ Add Antibiotic Row", on_click=add_row)

abx_classes = ["BLBI", "BI", "Glycopeptide", "Ceph12", "Sulfa", "Aminoglycoside", 
               "Quinolone", "Macrolide", "Metronidazole", "Penicillin", 
               "Carbapenem", "Ceph34", "Tetracycline", "Lincosamide", 
               "Fusidic", "Polymyxin", "Fosfomycin", "Oxazolidinone", "Monobactam"]

abx_features = {cls: 0.0 for cls in abx_classes}

# Display rows and calculate decay
for i, entry in enumerate(st.session_state.abx_list):
    c1, c2, c3 = st.columns(3)
    with c1:
        entry['class'] = st.selectbox(f"Class #{i+1}", abx_classes, key=f"cls_{i}")
    with c2:
        entry['dur'] = st.number_input(f"Duration (days) #{i+1}", min_value=0, key=f"dur_{i}")
    with c3:
        entry['last'] = st.number_input(f"Days since last dose #{i+1}", min_value=0, key=f"last_{i}")
    
    abx_features[entry['class']] = entry['dur'] * np.exp(-0.05 * entry['last'])

# --- 4. ADMISSION HISTORY ---
st.divider()
c_a, c_b = st.columns(2)
with c_a:
    adm_yes = st.toggle("Current Admission?", key="adm_tog")
    adm_val = st.number_input("Admission days", min_value=0, key="adm_val_in") if adm_yes else 0
with c_b:
    prev_yes = st.toggle("Previous Admission?", key="prev_tog")
    prev_val = st.number_input("Previous discharge (days)", min_value=0, value=9999, key="prev_val_in") if prev_yes else 9999

# --- 5. PREDICTION & SHAP ---
# [Include the previous prediction and get_probability_waterfall logic here]
# ...
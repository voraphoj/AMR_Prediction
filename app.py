import os

import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Updated version: clear row delete buttons + one-time balloons after Run Prediction.

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="Staph Oxacillin Predictor", layout="wide")

MODEL_PATHS = [
    "staph_oxa_demo_xgb_model.json",
    "staph_oxa_demo_xgb_model(1).json",  # fallback for local testing after upload
]

SAMPLE_TYPES = ["Blood", "Pus", "Sputum", "Endotracheal_aspirate", "Urine", "Other"]
ABX_CLASSES = [
    "BLBI", "BI", "Glycopeptide", "Ceph12", "Sulfa", "Aminoglycoside", "Quinolone",
    "Macrolide", "Metronidazole", "Penicillin", "Carbapenem", "Ceph34",
    "Tetracycline", "Lincosamide", "Fusidic", "Polymyxin", "Fosfomycin",
    "Oxazolidinone", "Monobactam"
]
LAMBDA_DECAY = 0.05
TREND_DAYS = 90


def find_model_path() -> str:
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path
    return MODEL_PATHS[0]


@st.cache_resource
def load_assets():
    model = xgb.Booster()
    model.load_model(find_model_path())
    return model, model.feature_names


model, feature_names = load_assets()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def next_row_id(counter_key):
    if counter_key not in st.session_state:
        st.session_state[counter_key] = 0
    st.session_state[counter_key] += 1
    return st.session_state[counter_key]


def ensure_row_ids(list_key, counter_key):
    if list_key not in st.session_state:
        st.session_state[list_key] = []
    for row in st.session_state[list_key]:
        if "_id" not in row:
            row["_id"] = next_row_id(counter_key)


def safe_class_index(value):
    return ABX_CLASSES.index(value) if value in ABX_CLASSES else 0


# --- 2. FEATURE BUILDING HELPERS ---
def calculate_current_abx_features(abx_list):
    """Current prediction: one decayed exposure value per antibiotic class."""
    abx_features = {cls: 0.0 for cls in ABX_CLASSES}
    for entry in abx_list:
        cls = entry.get("class", "BLBI")
        dur = float(entry.get("dur", 0) or 0)
        last = float(entry.get("last", 0) or 0)
        if cls in abx_features:
            abx_features[cls] = dur * np.exp(-LAMBDA_DECAY * last)
    return abx_features


def overlap_duration(start, end, window_start, window_end):
    """Overlap length between [start, end] and (window_start, window_end]."""
    left = max(start, window_start)
    right = min(end, window_end)
    return max(0.0, right - left)


def calculate_trend_abx_features(past_abx_list, plan_abx_list, future_day):
    """
    Calculate antibiotic exposure for a future day.

    For each class, Duration is the sum of all treatment days that remain inside
    the previous 90-day window. If a past antibiotic is restarted in the plan,
    past + planned days are summed, but days since last dose follows the planned
    antibiotic.
    """
    exposures = {cls: 0.0 for cls in ABX_CLASSES}
    window_start = future_day - 90
    window_end = future_day

    for cls in ABX_CLASSES:
        total_duration = 0.0
        past_last_values = []
        plan_durations = []

        # Past antibiotics: interval ends before or at day 0.
        # Example: duration=20, last=0 gives interval [-20, 0].
        for entry in past_abx_list:
            if entry.get("class") != cls:
                continue
            dur = float(entry.get("dur", 0) or 0)
            last = float(entry.get("last", 0) or 0)
            if dur <= 0:
                continue
            start = -last - dur
            end = -last
            total_duration += overlap_duration(start, end, window_start, window_end)
            past_last_values.append(last)

        # Planned antibiotics: all plans start at day 0 and last for their duration.
        for entry in plan_abx_list:
            if entry.get("class") != cls:
                continue
            dur = float(entry.get("dur", 0) or 0)
            if dur <= 0:
                continue
            start = 0.0
            end = dur
            total_duration += overlap_duration(start, end, window_start, window_end)
            plan_durations.append(dur)

        if total_duration <= 0:
            exposures[cls] = 0.0
            continue

        if plan_durations:
            latest_plan_end = max(plan_durations)
            days_since_last_dose = max(0.0, future_day - latest_plan_end)
        elif past_last_values:
            days_since_last_dose = future_day + min(past_last_values)
        else:
            days_since_last_dose = 0.0

        exposures[cls] = total_duration * np.exp(-LAMBDA_DECAY * days_since_last_dose)

    return exposures


def build_input_row(
    sex_val,
    age_days,
    sample_type,
    positive_min,
    admission_days,
    previous_discharge_days,
    abx_features,
):
    input_dict = {
        "Sex": int(sex_val),
        "Age": int(age_days),
        "Positive_MIN": float(positive_min) if sample_type == "Blood" else 0.0,
        "Admission_to_Receive": int(admission_days),
        "Discharge_to_Admission": int(previous_discharge_days),
        "Blood": 1 if sample_type == "Blood" else 0,
        "Pus": 1 if sample_type == "Pus" else 0,
        "Sputum": 1 if sample_type == "Sputum" else 0,
        "Endotracheal_aspirate": 1 if sample_type == "Endotracheal_aspirate" else 0,
        "Urine": 1 if sample_type == "Urine" else 0,
        "Other": 1 if sample_type == "Other" else 0,
    }
    input_dict.update(abx_features)
    return pd.DataFrame([input_dict])[feature_names]


def predict_probability(df):
    dmat = xgb.DMatrix(df)
    return float(model.predict(dmat)[0])


def build_trend_dataframe(baseline, plan_abx_list, discharge_plan_days):
    rows = []
    for day in range(TREND_DAYS + 1):
        age_future = baseline["age_days"] + day

        if discharge_plan_days < 9999 and day >= discharge_plan_days:
            admission_future = 0
            previous_discharge_future = day - discharge_plan_days
        else:
            admission_future = baseline["admission_days"] + day
            previous_discharge_future = baseline["previous_discharge_days"]

        abx_features = calculate_trend_abx_features(
            baseline["past_abx_list"],
            plan_abx_list,
            future_day=day,
        )

        for sample_type in SAMPLE_TYPES:
            df = build_input_row(
                sex_val=baseline["sex_val"],
                age_days=age_future,
                sample_type=sample_type,
                positive_min=baseline["positive_min"],
                admission_days=admission_future,
                previous_discharge_days=previous_discharge_future,
                abx_features=abx_features,
            )
            rows.append({
                "Day": day,
                "Sample Type": sample_type,
                "Probability": predict_probability(df) * 100,
            })

    return pd.DataFrame(rows)


# --- 3. HEADER & RESET ---
h1, h2 = st.columns([8, 2])
with h1:
    st.title("🧫 Oxacillin Resistance Demo")
with h2:
    st.button("🔄 Reset All", on_click=reset_app)

# --- 4. INPUT SECTIONS ---
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
    sample = st.selectbox("Sample Type", SAMPLE_TYPES, key="sample")

    pos_min = 0
    if sample == "Blood":
        p_d = st.number_input("Positivity (Days)", min_value=0, key="pos_d")
        p_h = st.number_input("Hours", min_value=0, max_value=23, key="pos_h")
        p_m = st.number_input("Minutes", min_value=0, max_value=59, key="pos_m")
        pos_min = (p_d * 1440) + (p_h * 60) + p_m

st.divider()

# --- 5. ANTIBIOTICS ---
st.subheader("Antibiotics (Past 3 Months)")
ensure_row_ids("abx_list", "abx_row_counter")


def add_abx():
    st.session_state.abx_list.append({
        "_id": next_row_id("abx_row_counter"),
        "class": "BLBI",
        "dur": 0,
        "last": 0,
    })


def delete_abx(row_id):
    st.session_state.abx_list = [
        row for row in st.session_state.abx_list
        if row.get("_id") != row_id
    ]
    for key in [f"abx_class_{row_id}", f"abx_dur_{row_id}", f"abx_last_{row_id}"]:
        st.session_state.pop(key, None)
    # Old prediction/trend can become stale after row deletion.
    st.session_state.pop("trend_df", None)


st.button("➕ Add Antibiotic Row", on_click=add_abx)

for i, entry in enumerate(st.session_state.abx_list):
    row_id = entry["_id"]
    c1, c2, c3, c4 = st.columns([3.0, 3.0, 3.0, 1.4])

    entry["class"] = c1.selectbox(
        f"Class #{i + 1}",
        ABX_CLASSES,
        index=safe_class_index(entry.get("class", "BLBI")),
        key=f"abx_class_{row_id}",
    )
    entry["dur"] = c2.number_input(
        f"Duration (days) #{i + 1}",
        min_value=0,
        value=int(entry.get("dur", 0) or 0),
        key=f"abx_dur_{row_id}",
    )
    entry["last"] = c3.number_input(
        f"Days since last dose #{i + 1}",
        min_value=0,
        value=int(entry.get("last", 0) or 0),
        key=f"abx_last_{row_id}",
    )
    c4.write("")
    c4.write("")
    if c4.button("🗑️ Remove", key=f"delete_abx_{row_id}", use_container_width=True):
        delete_abx(row_id)
        st.rerun()

current_abx_features = calculate_current_abx_features(st.session_state.abx_list)

st.divider()

# --- 6. ADMISSION ---
c_adm1, c_adm2 = st.columns(2)
adm_yes = c_adm1.toggle("Current Admission?", key="adm_t")
adm_val = c_adm1.number_input("Admission days", min_value=0, key="adm_v") if adm_yes else 0

prev_yes = c_adm2.toggle("Previous Admission?", key="prev_t")
prev_val = c_adm2.number_input("Previous discharge (days)", min_value=0, value=9999, key="prev_v") if prev_yes else 9999

# --- 7. CALCULATION & OUTPUT ---
st.divider()

if st.button("🚀 RUN PREDICTION", type="primary", use_container_width=True):
    df_final = build_input_row(
        sex_val=sex_val,
        age_days=age_days,
        sample_type=sample,
        positive_min=pos_min,
        admission_days=adm_val,
        previous_discharge_days=prev_val,
        abx_features=current_abx_features,
    )

    prob = predict_probability(df_final)

    st.session_state.prediction_done = True
    st.session_state.latest_probability = prob
    st.session_state.latest_df = df_final
    st.session_state.baseline_inputs = {
        "sex_val": sex_val,
        "age_days": age_days,
        "sample": sample,
        "positive_min": pos_min,
        "admission_days": adm_val,
        "previous_discharge_days": prev_val,
        "past_abx_list": [dict(x) for x in st.session_state.abx_list],
    }
    st.session_state.pop("trend_df", None)

    # Trigger balloons once for this prediction click only.
    st.session_state.show_prediction_balloons_once = True

if st.session_state.pop("show_prediction_balloons_once", False):
    st.balloons()

if st.session_state.get("prediction_done", False):
    prob = st.session_state.latest_probability
    df_final = st.session_state.latest_df

    st.markdown(f"### Predicted Probability of Resistance: **{prob * 100:.2f}%**")

    st.subheader("Feature Influence (Probability Scale)")
    explainer = shap.TreeExplainer(model)
    shap_obj = explainer(df_final)

    v = shap_obj.values[0]
    base_lo = shap_obj.base_values[0]

    p_contribs = np.zeros_like(v)
    curr_lo = base_lo
    for i in range(len(v)):
        p_before = sigmoid(curr_lo)
        curr_lo += v[i]
        p_contribs[i] = sigmoid(curr_lo) - p_before

    label_map = {"Positive_MIN": "Time to Positivity", "Admission_to_Receive": "Admission Days"}
    clean_labels = [label_map.get(f, f.replace("_", " ")) for f in feature_names]

    prob_exp = shap.Explanation(
        values=p_contribs,
        base_values=sigmoid(base_lo),
        data=df_final.iloc[0],
        feature_names=clean_labels,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(prob_exp, show=False)
    st.pyplot(fig)
    plt.close(fig)

    # --- 8. ANTIBIOTIC PLAN & TREND SIMULATION ---
    st.divider()
    st.subheader("Antibiotic Plan")

    ensure_row_ids("plan_abx_list", "plan_abx_row_counter")

    def add_plan_abx():
        st.session_state.plan_abx_list.append({
            "_id": next_row_id("plan_abx_row_counter"),
            "class": "BLBI",
            "dur": 0,
        })

    def delete_plan_abx(row_id):
        st.session_state.plan_abx_list = [
            row for row in st.session_state.plan_abx_list
            if row.get("_id") != row_id
        ]
        for key in [f"plan_class_{row_id}", f"plan_dur_{row_id}"]:
            st.session_state.pop(key, None)
        st.session_state.pop("trend_df", None)

    st.button("➕ Add Antibiotic Plan Row", on_click=add_plan_abx)

    for i, entry in enumerate(st.session_state.plan_abx_list):
        row_id = entry["_id"]
        c1, c2, c3 = st.columns([4.0, 4.0, 1.4])
        entry["class"] = c1.selectbox(
            f"Plan Class #{i + 1}",
            ABX_CLASSES,
            index=safe_class_index(entry.get("class", "BLBI")),
            key=f"plan_class_{row_id}",
        )
        entry["dur"] = c2.number_input(
            f"Plan Duration (days) #{i + 1}",
            min_value=0,
            value=int(entry.get("dur", 0) or 0),
            key=f"plan_dur_{row_id}",
        )
        c3.write("")
        c3.write("")
        if c3.button("🗑️ Remove", key=f"delete_plan_abx_{row_id}", use_container_width=True):
            delete_plan_abx(row_id)
            st.rerun()

    discharge_plan = st.number_input(
        "Discharge Plan (days)",
        min_value=0,
        value=9999,
        key="discharge_plan_days",
        help="Use 9999 if no discharge is planned within the 90-day simulation.",
    )

    if st.button("📈 RUN TREND", type="primary", use_container_width=True):
        trend_df = build_trend_dataframe(
            baseline=st.session_state.baseline_inputs,
            plan_abx_list=[dict(x) for x in st.session_state.plan_abx_list],
            discharge_plan_days=int(discharge_plan),
        )
        st.session_state.trend_df = trend_df

    if "trend_df" in st.session_state:
        trend_df = st.session_state.trend_df
        st.subheader("Predicted Oxacillin Resistance Trend")

        color_map = {
            "Blood": "red",
            "Pus": "green",
            "Sputum": "blue",
            "Endotracheal_aspirate": "brown",
            "Urine": "gold",
            "Other": "black",
        }

        fig, ax = plt.subplots(figsize=(11, 6))
        for sample_type in SAMPLE_TYPES:
            plot_df = trend_df[trend_df["Sample Type"] == sample_type]
            ax.plot(
                plot_df["Day"],
                plot_df["Probability"],
                label=sample_type.replace("_", " "),
                color=color_map[sample_type],
                linewidth=2,
            )

        ax.set_xlabel("Days from present")
        ax.set_ylabel("Predicted probability of oxacillin resistance (%)")
        ax.set_xlim(0, TREND_DAYS)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sample Type", loc="best")
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Show trend table"):
            st.dataframe(trend_df, use_container_width=True)

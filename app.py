#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# STAGE 10 — CCA WOOD TREATMENT PREDICTOR
# =============================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CCA Treatment Predictor",
    page_icon="🌲",
    layout="centered"
)


# =============================================================================
# MODEL PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

LP_MODEL_PATH = BASE_DIR / "final_gp_Lp.joblib"
RCCA_MODEL_PATH = BASE_DIR / "final_gp_Rcca.joblib"


# =============================================================================
# LOAD MODELS
# =============================================================================

@st.cache_resource
def load_models():

    lp_model = joblib.load(LP_MODEL_PATH)
    rcca_model = joblib.load(RCCA_MODEL_PATH)

    return lp_model, rcca_model


try:

    lp_model, rcca_model = load_models()

except Exception as error:

    st.error("Unable to load the prediction models.")
    st.exception(error)
    st.stop()


# =============================================================================
# EXPECTED MODEL FEATURES
# =============================================================================

FEATURES = [
    "Mc",
    "ρod",
    "C(cca)",
    "P(kPA)",
    "V(kPA)",
    "V(t)"
]


# =============================================================================
# SIDEBAR INPUT PANEL
# =============================================================================

with st.sidebar:

    st.title("🌲 CCA Treatment")

    st.caption(
        "Enter wood condition and treatment parameters."
    )

    st.divider()

    # -------------------------------------------------------------------------
    # MATERIAL
    # -------------------------------------------------------------------------

    st.subheader("Wood")

    mc = st.slider(
        "Moisture content, Mc",
        min_value=0.14,
        max_value=0.45,
        value=0.28,
        step=0.01
    )

    rho_od = st.slider(
        "Oven-dry density, ρod",
        min_value=445.2,
        max_value=1005.4,
        value=725.0,
        step=1.0
    )

    # -------------------------------------------------------------------------
    # TREATMENT
    # -------------------------------------------------------------------------

    st.subheader("Treatment")

    cca = st.slider(
        "CCA concentration, C(cca)",
        min_value=1.0,
        max_value=10.0,
        value=5.5,
        step=0.1
    )

    pressure = st.slider(
        "Pressure, P(kPA)",
        min_value=1.0,
        max_value=14.0,
        value=7.5,
        step=0.1
    )

    vacuum = st.slider(
        "Vacuum, V(kPA)",
        min_value=1.0,
        max_value=8.0,
        value=4.5,
        step=0.1
    )

    treatment_time = st.slider(
        "Treatment time, V(t)",
        min_value=10.0,
        max_value=120.0,
        value=60.0,
        step=1.0
    )

    st.divider()

    predict_button = st.button(
        "Predict",
        type="primary",
        use_container_width=True
    )


# =============================================================================
# MAIN PAGE
# =============================================================================

st.title("CCA Wood Treatment Predictor")

st.write(
    "Predict lateral preservative penetration and CCA retention "
    "for the selected wood and treatment conditions."
)


# =============================================================================
# PREDICTION
# =============================================================================

if predict_button:

    # -------------------------------------------------------------------------
    # EXACT FEATURE ORDER
    # -------------------------------------------------------------------------

    X = pd.DataFrame(
        [[
            mc,
            rho_od,
            cca,
            pressure,
            vacuum,
            treatment_time
        ]],
        columns=FEATURES
    )

    try:

        # ---------------------------------------------------------------------
        # GP PREDICTIONS
        # ---------------------------------------------------------------------

        lp_mean, lp_std = lp_model.predict(
            X,
            return_std=True
        )

        rcca_mean, rcca_std = rcca_model.predict(
            X,
            return_std=True
        )

    except Exception as error:

        st.error("Prediction failed.")
        st.exception(error)
        st.stop()


    # -------------------------------------------------------------------------
    # CONVERT TO SCALARS
    # -------------------------------------------------------------------------

    lp_mean = float(np.asarray(lp_mean).ravel()[0])
    lp_std = float(np.asarray(lp_std).ravel()[0])

    rcca_mean = float(np.asarray(rcca_mean).ravel()[0])
    rcca_std = float(np.asarray(rcca_std).ravel()[0])


    # -------------------------------------------------------------------------
    # 95% PREDICTIVE INTERVALS
    # -------------------------------------------------------------------------

    lp_lower = lp_mean - 1.96 * lp_std
    lp_upper = lp_mean + 1.96 * lp_std

    rcca_lower = rcca_mean - 1.96 * rcca_std
    rcca_upper = rcca_mean + 1.96 * rcca_std


    # =========================================================================
    # RESULTS
    # =========================================================================

    st.divider()

    st.subheader("Predicted Treatment Performance")


    col1, col2 = st.columns(2)


    # -------------------------------------------------------------------------
    # LATERAL PENETRATION
    # -------------------------------------------------------------------------

    with col1:

        st.metric(
            "Lateral penetration",
            f"{lp_mean:.2f} mm"
        )

        st.caption(
            f"Predictive SD: {lp_std:.2f} mm"
        )

        st.caption(
            f"Approx. 95% interval: "
            f"{lp_lower:.2f} – {lp_upper:.2f} mm"
        )


    # -------------------------------------------------------------------------
    # CCA RETENTION
    # -------------------------------------------------------------------------

    with col2:

        st.metric(
            "CCA retention",
            f"{rcca_mean:.2f} kg/m³"
        )

        st.caption(
            f"Predictive SD: {rcca_std:.2f} kg/m³"
        )

        st.caption(
            f"Approx. 95% interval: "
            f"{rcca_lower:.2f} – {rcca_upper:.2f} kg/m³"
        )


    # =========================================================================
    # SIMPLE INTERPRETATION
    # =========================================================================

    st.divider()

    st.subheader("Interpretation")


    # -------------------------------------------------------------------------
    # RETENTION INTERPRETATION
    # -------------------------------------------------------------------------

    if rcca_mean < 20:

        retention_label = "Low predicted retention"

    elif rcca_mean < 30:

        retention_label = "Moderate predicted retention"

    else:

        retention_label = "High predicted retention"


    # -------------------------------------------------------------------------
    # PENETRATION INTERPRETATION
    # -------------------------------------------------------------------------

    if lp_mean < 10:

        penetration_label = "Low predicted penetration"

    elif lp_mean < 20:

        penetration_label = "Moderate predicted penetration"

    else:

        penetration_label = "High predicted penetration"


    st.write(
        f"**Penetration:** {penetration_label}"
    )

    st.write(
        f"**Retention:** {retention_label}"
    )


# =============================================================================
# INITIAL STATE
# =============================================================================

else:

    st.info(
        "Adjust the treatment and wood parameters in the sidebar, "
        "then select **Predict**."
    )


# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    "Probabilistic CCA wood treatment prediction using frozen "
    "Gaussian Process surrogate models."
)

#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# CCA WOOD TREATMENT PREDICTOR — USER INTERFACE
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
    page_icon="D.O.O",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
<style>

/* -------------------------------------------------------------------------- */
/* APP WIDTH                                                                  */
/* -------------------------------------------------------------------------- */

.block-container {
    max-width: 980px;
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}


/* -------------------------------------------------------------------------- */
/* FONT                                                                       */
/* -------------------------------------------------------------------------- */

html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}


/* -------------------------------------------------------------------------- */
/* HERO                                                                       */
/* -------------------------------------------------------------------------- */

.hero {
    background: linear-gradient(135deg, #12372A 0%, #1E5A43 100%);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 18px;
    color: white;
}

.hero-title {
    font-size: 30px;
    font-weight: 750;
    margin: 0;
    line-height: 1.15;
}

.hero-text {
    font-size: 15px;
    margin-top: 8px;
    margin-bottom: 0;
    opacity: 0.88;
}


/* -------------------------------------------------------------------------- */
/* SECTION LABEL                                                              */
/* -------------------------------------------------------------------------- */

.section-title {
    font-size: 17px;
    font-weight: 700;
    color: #17382D;
    margin-top: 4px;
    margin-bottom: 10px;
}


/* -------------------------------------------------------------------------- */
/* INPUT CARD                                                                 */
/* -------------------------------------------------------------------------- */

[data-testid="stForm"] {
    background: #F7F9F7;
    border: 1px solid #E1E8E3;
    border-radius: 16px;
    padding: 18px 20px 12px 20px;
}


/* -------------------------------------------------------------------------- */
/* BUTTON                                                                     */
/* -------------------------------------------------------------------------- */

.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 46px;
}


/* -------------------------------------------------------------------------- */
/* RESULT CARDS                                                               */
/* -------------------------------------------------------------------------- */

.result-card {
    background: #FFFFFF;
    border: 1px solid #DDE7E1;
    border-top: 5px solid #2F7659;
    border-radius: 15px;
    padding: 18px 20px;
    min-height: 155px;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.045);
}

.result-label {
    color: #52635B;
    font-size: 14px;
    font-weight: 650;
    margin-bottom: 5px;
}

.result-value {
    color: #12372A;
    font-size: 32px;
    font-weight: 760;
    line-height: 1.15;
}

.result-range {
    margin-top: 10px;
    color: #66756E;
    font-size: 13px;
    line-height: 1.45;
}


/* -------------------------------------------------------------------------- */
/* SUMMARY CARD                                                               */
/* -------------------------------------------------------------------------- */

.summary-card {
    background: #EEF6F1;
    border-left: 5px solid #2F7659;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 16px;
    color: #213C31;
    font-size: 14px;
}


/* -------------------------------------------------------------------------- */
/* FOOTER                                                                     */
/* -------------------------------------------------------------------------- */

.footer {
    text-align: center;
    color: #7A8781;
    font-size: 12px;
    margin-top: 20px;
}


/* Reduce excess Streamlit spacing */
div[data-testid="stVerticalBlock"] {
    gap: 0.7rem;
}

</style>
""", unsafe_allow_html=True)


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

except Exception:
    st.error(
        "The prediction service is currently unavailable. "
        "Please check the model files and try again."
    )
    st.stop()


# =============================================================================
# MODEL FEATURES
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
# HEADER
# =============================================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🌲 CCA Treatment Predictor</div>
    <div class="hero-text">
        Estimate preservative penetration and retention before treatment.
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# INPUT FORM
# =============================================================================

with st.form("prediction_form"):

    st.markdown(
        '<div class="section-title">Wood properties</div>',
        unsafe_allow_html=True
    )

    wood_col1, wood_col2 = st.columns(2)

    with wood_col1:

        mc_percent = st.number_input(
            "Moisture content (%)",
            min_value=14.0,
            max_value=45.0,
            value=28.0,
            step=1.0,
            help="Current moisture content of the wood."
        )

    with wood_col2:

        rho_od = st.number_input(
            "Wood density (kg/m³)",
            min_value=445.2,
            max_value=1005.4,
            value=725.0,
            step=5.0,
            help="Oven-dry density of the wood."
        )


    st.markdown(
        '<div class="section-title">Treatment settings</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        cca = st.number_input(
            "CCA (%)",
            min_value=1.0,
            max_value=10.0,
            value=5.5,
            step=0.1,
            help="Concentration of the CCA treatment solution."
        )

    with col2:

        pressure = st.number_input(
            "Pressure (kPa)",
            min_value=1.0,
            max_value=14.0,
            value=7.5,
            step=0.1
        )

    with col3:

        vacuum = st.number_input(
            "Vacuum (kPa)",
            min_value=1.0,
            max_value=8.0,
            value=4.5,
            step=0.1
        )

    with col4:

        treatment_time = st.number_input(
            "Vacuum time (min)",
            min_value=10.0,
            max_value=120.0,
            value=60.0,
            step=5.0
        )


    predict_button = st.form_submit_button(
        "Calculate treatment performance",
        type="primary",
        use_container_width=True
    )


# =============================================================================
# PREDICTION
# =============================================================================

if predict_button:

    # Convert user-friendly percentage to model scale
    mc = mc_percent / 100.0

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

        lp_mean, lp_std = lp_model.predict(
            X,
            return_std=True
        )

        rcca_mean, rcca_std = rcca_model.predict(
            X,
            return_std=True
        )

    except Exception:

        st.error(
            "A prediction could not be produced for these settings."
        )
        st.stop()


    # =========================================================================
    # CONVERT RESULTS
    # =========================================================================

    lp_mean = float(np.asarray(lp_mean).ravel()[0])
    lp_std = float(np.asarray(lp_std).ravel()[0])

    rcca_mean = float(np.asarray(rcca_mean).ravel()[0])
    rcca_std = float(np.asarray(rcca_std).ravel()[0])


    # =========================================================================
    # EXPECTED RANGES
    # =========================================================================

    lp_lower = max(0.0, lp_mean - 1.96 * lp_std)
    lp_upper = max(0.0, lp_mean + 1.96 * lp_std)

    rcca_lower = max(0.0, rcca_mean - 1.96 * rcca_std)
    rcca_upper = max(0.0, rcca_mean + 1.96 * rcca_std)


    # =========================================================================
    # RESULTS
    # =========================================================================

    st.markdown(
        '<div class="section-title" style="margin-top:18px;">'
        'Predicted performance'
        '</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)


    # -------------------------------------------------------------------------
    # PENETRATION
    # -------------------------------------------------------------------------

    with result_col1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    ↔ Preservative penetration
                </div>

                <div class="result-value">
                    {lp_mean:.2f} mm
                </div>

                <div class="result-range">
                    Expected range<br>
                    <strong>{lp_lower:.2f} – {lp_upper:.2f} mm</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # -------------------------------------------------------------------------
    # RETENTION
    # -------------------------------------------------------------------------

    with result_col2:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    ◈ CCA retention
                </div>

                <div class="result-value">
                    {rcca_mean:.2f} kg/m³
                </div>

                <div class="result-range">
                    Expected range<br>
                    <strong>
                        {rcca_lower:.2f} – {rcca_upper:.2f} kg/m³
                    </strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =========================================================================
    # SIMPLE SUMMARY
    # =========================================================================

    st.markdown(
        f"""
        <div class="summary-card">
            <strong>✓ Treatment estimate</strong><br>
            With the selected settings, the wood is expected to achieve
            approximately <strong>{lp_mean:.2f} mm penetration</strong>
            and <strong>{rcca_mean:.2f} kg/m³ CCA retention</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# INITIAL STATE
# =============================================================================

else:

    st.caption(
        "Enter the wood properties and treatment settings above, "
        "then calculate the expected performance."
    )


# =============================================================================
# OPTIONAL INFORMATION
# =============================================================================

with st.expander("About this prediction"):

    st.write(
        "The predictor estimates lateral preservative penetration "
        "and CCA retention from wood properties and selected treatment "
        "conditions. Predictions should be used to support treatment "
        "planning and should be verified where required."
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
    <div class="footer">
        CCA Wood Treatment Decision Support
    </div>
    """,
    unsafe_allow_html=True
)

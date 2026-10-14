#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# CCA WOOD TREATMENT DECISION SUPPORT
# Forward prediction + uncertainty + reverse treatment search
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
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# STYLING
# =============================================================================

st.markdown("""
<style>

/* Main application width */
.block-container {
    max-width: 1180px;
    padding-top: 1.25rem;
    padding-bottom: 1.5rem;
}

/* Font */
html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

/* Headings */
h1, h2, h3 {
    letter-spacing: -0.02em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F4F8F5;
    border-right: 1px solid #DDE7E1;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #12372A 0%, #216348 100%);
    border-radius: 16px;
    padding: 20px 26px;
    margin-bottom: 16px;
    color: white;
}

.hero-title {
    font-size: 28px;
    font-weight: 750;
    line-height: 1.15;
}

.hero-text {
    font-size: 14px;
    margin-top: 6px;
    opacity: 0.90;
}

/* Section headings */
.section-title {
    font-size: 17px;
    font-weight: 700;
    color: #17382D;
    margin-top: 6px;
    margin-bottom: 8px;
}

/* Input form */
[data-testid="stForm"] {
    background: #F8FAF8;
    border: 1px solid #DFE8E2;
    border-radius: 15px;
    padding: 16px 18px 10px 18px;
}

/* Buttons */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 44px;
}

/* Result cards */
.result-card {
    background: #FFFFFF;
    border: 1px solid #DDE7E1;
    border-top: 5px solid #2F7659;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 160px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.04);
}

.result-label {
    color: #52635B;
    font-size: 14px;
    font-weight: 700;
}

.result-value {
    color: #12372A;
    font-size: 32px;
    font-weight: 760;
    margin-top: 5px;
    line-height: 1.15;
}

.result-range {
    color: #64736C;
    font-size: 13px;
    margin-top: 10px;
    line-height: 1.5;
}

.uncertainty-card {
    background: #F5F8F6;
    border-radius: 10px;
    padding: 10px 13px;
    margin-top: 11px;
    font-size: 12px;
    color: #56665E;
}

/* Summary */
.summary-card {
    background: #EAF5EE;
    border-left: 5px solid #2F7659;
    border-radius: 11px;
    padding: 13px 17px;
    margin-top: 14px;
    font-size: 14px;
    color: #213C31;
}

/* Reverse recommendation */
.recommend-card {
    background: #F1F6F3;
    border: 1px solid #D8E5DD;
    border-radius: 14px;
    padding: 17px 20px;
    margin-top: 12px;
}

/* Small muted text */
.small-note {
    color: #718078;
    font-size: 12px;
}

/* Footer */
.footer {
    text-align: center;
    color: #7A8781;
    font-size: 11px;
    margin-top: 18px;
}

/* Reduce excessive Streamlit vertical spacing */
div[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
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
# CALIBRATED UNCERTAINTY FACTORS
# =============================================================================

LP_CALIBRATION = 1.1459
RCCA_CALIBRATION = 1.2378

Z_95 = 1.96


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
# LOAD MODELS
# =============================================================================

@st.cache_resource
def load_models():

    lp = joblib.load(LP_MODEL_PATH)
    rcca = joblib.load(RCCA_MODEL_PATH)

    return lp, rcca


try:
    lp_model, rcca_model = load_models()

except Exception:
    st.error(
        "The prediction models could not be loaded. "
        "Please check the model files."
    )
    st.stop()


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_treatment(X):

    lp_mean, lp_std = lp_model.predict(
        X,
        return_std=True
    )

    rcca_mean, rcca_std = rcca_model.predict(
        X,
        return_std=True
    )

    lp_mean = np.asarray(lp_mean).ravel()
    lp_std = np.asarray(lp_std).ravel()

    rcca_mean = np.asarray(rcca_mean).ravel()
    rcca_std = np.asarray(rcca_std).ravel()

    # Calibrated predictive uncertainty
    lp_std_cal = lp_std * LP_CALIBRATION
    rcca_std_cal = rcca_std * RCCA_CALIBRATION

    return (
        lp_mean,
        lp_std_cal,
        rcca_mean,
        rcca_std_cal
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown("## 🌲 CCA Treatment")

    st.caption(
        "Prediction and treatment-setting support"
    )

    st.divider()

    mode = st.radio(
        "Select tool",
        [
            "Forward prediction",
            "Reverse prediction"
        ]
    )

    st.divider()

    st.markdown("##### What each tool does")

    if mode == "Forward prediction":

        st.caption(
            "Estimate penetration and CCA retention "
            "from known treatment settings."
        )

    else:

        st.caption(
            "Find treatment settings likely to achieve "
            "your required performance."
        )

    st.divider()

    with st.expander("Prediction uncertainty"):

        st.caption(
            "Results include a calibrated 95% prediction "
            "range to show uncertainty around each estimate."
        )


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🌲 CCA Wood Treatment Decision Support
        </div>
        <div class="hero-text">
            Predict treatment performance or identify suitable
            operating settings.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# FORWARD PREDICTION
# =============================================================================

if mode == "Forward prediction":

    st.markdown(
        '<div class="section-title">Enter treatment conditions</div>',
        unsafe_allow_html=True
    )

    with st.form("forward_form"):

        # ---------------------------------------------------------------------
        # WOOD
        # ---------------------------------------------------------------------

        st.markdown("**Wood properties**")

        wood1, wood2 = st.columns(2)

        with wood1:

            mc_percent = st.number_input(
                "Moisture content (%)",
                min_value=14.0,
                max_value=45.0,
                value=28.0,
                step=1.0,
                help="Moisture content of the wood before treatment."
            )

        with wood2:

            rho_od = st.number_input(
                "Wood density (kg/m³)",
                min_value=445.2,
                max_value=1005.4,
                value=725.0,
                step=5.0,
                help="Oven-dry density of the wood."
            )


        # ---------------------------------------------------------------------
        # TREATMENT
        # ---------------------------------------------------------------------

        st.markdown("**Treatment settings**")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            cca = st.number_input(
                "CCA concentration (%)",
                min_value=1.0,
                max_value=10.0,
                value=5.5,
                step=0.1
            )

        with c2:

            pressure = st.number_input(
                "Pressure (kPa)",
                min_value=1.0,
                max_value=14.0,
                value=7.5,
                step=0.1
            )

        with c3:

            vacuum = st.number_input(
                "Vacuum (kPa)",
                min_value=1.0,
                max_value=8.0,
                value=4.5,
                step=0.1
            )

        with c4:

            treatment_time = st.number_input(
                "Vacuum time (min)",
                min_value=10.0,
                max_value=120.0,
                value=60.0,
                step=5.0
            )

        predict_button = st.form_submit_button(
            "Predict treatment performance",
            type="primary",
            use_container_width=True
        )


    # =========================================================================
    # RUN FORWARD PREDICTION
    # =========================================================================

    if predict_button:

        # IMPORTANT:
        # This assumes the GP was trained using Mc = 0.14–0.45.
        mc_model = mc_percent / 100.0

        X = pd.DataFrame(
            [[
                mc_model,
                rho_od,
                cca,
                pressure,
                vacuum,
                treatment_time
            ]],
            columns=FEATURES
        )

        try:

            (
                lp_mean,
                lp_std,
                rcca_mean,
                rcca_std
            ) = predict_treatment(X)

        except Exception:
            st.error("Prediction could not be completed.")
            st.stop()


        lp_mean = float(lp_mean[0])
        lp_std = float(lp_std[0])

        rcca_mean = float(rcca_mean[0])
        rcca_std = float(rcca_std[0])


        # ---------------------------------------------------------------------
        # 95% CALIBRATED INTERVAL
        # ---------------------------------------------------------------------

        lp_margin = Z_95 * lp_std
        rcca_margin = Z_95 * rcca_std

        lp_lower = max(0.0, lp_mean - lp_margin)
        lp_upper = lp_mean + lp_margin

        rcca_lower = max(0.0, rcca_mean - rcca_margin)
        rcca_upper = rcca_mean + rcca_margin


        # ---------------------------------------------------------------------
        # RESULTS
        # ---------------------------------------------------------------------

        st.markdown(
            '<div class="section-title" '
            'style="margin-top:16px;">'
            'Predicted performance'
            '</div>',
            unsafe_allow_html=True
        )

        result1, result2 = st.columns(2)


        with result1:

            html = f"""
<div class="result-card">
    <div class="result-label">
        ↔ Preservative penetration
    </div>

    <div class="result-value">
        {lp_mean:.2f} mm
    </div>

    <div class="result-range">
        95% prediction range:
        <strong>{lp_lower:.2f} – {lp_upper:.2f} mm</strong>
    </div>

    <div class="uncertainty-card">
        Prediction uncertainty:
        ± {lp_margin:.2f} mm
    </div>
</div>
"""

            st.markdown(
                html,
                unsafe_allow_html=True
            )


        with result2:

            html = f"""
<div class="result-card">
    <div class="result-label">
        ◈ CCA retention
    </div>

    <div class="result-value">
        {rcca_mean:.2f} kg/m³
    </div>

    <div class="result-range">
        95% prediction range:
        <strong>
            {rcca_lower:.2f} – {rcca_upper:.2f} kg/m³
        </strong>
    </div>

    <div class="uncertainty-card">
        Prediction uncertainty:
        ± {rcca_margin:.2f} kg/m³
    </div>
</div>
"""

            st.markdown(
                html,
                unsafe_allow_html=True
            )


        # ---------------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------------

        summary_html = f"""
<div class="summary-card">
    <strong>✓ Treatment estimate</strong><br>
    Expected performance is approximately
    <strong>{lp_mean:.2f} mm penetration</strong>
    and
    <strong>{rcca_mean:.2f} kg/m³ retention</strong>.
    The ranges above show uncertainty around these predictions.
</div>
"""

        st.markdown(
            summary_html,
            unsafe_allow_html=True
        )


# =============================================================================
# REVERSE PREDICTION
# =============================================================================

else:

    st.markdown(
        '<div class="section-title">'
        'Find suitable treatment settings'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Specify the wood condition and required treatment performance. "
        "The system will search for operating settings that best match "
        "your targets."
    )


    with st.form("reverse_form"):

        # ---------------------------------------------------------------------
        # WOOD CONDITION
        # ---------------------------------------------------------------------

        st.markdown("**Wood properties**")

        w1, w2 = st.columns(2)

        with w1:

            rev_mc_percent = st.number_input(
                "Moisture content (%)",
                min_value=14.0,
                max_value=45.0,
                value=28.0,
                step=1.0,
                key="rev_mc"
            )

        with w2:

            rev_density = st.number_input(
                "Wood density (kg/m³)",
                min_value=445.2,
                max_value=1005.4,
                value=725.0,
                step=5.0,
                key="rev_density"
            )


        # ---------------------------------------------------------------------
        # TARGETS
        # ---------------------------------------------------------------------

        st.markdown("**Required performance**")

        t1, t2 = st.columns(2)

        with t1:

            target_lp = st.number_input(
                "Target penetration (mm)",
                min_value=0.0,
                value=15.0,
                step=0.5
            )

        with t2:

            target_rcca = st.number_input(
                "Target CCA retention (kg/m³)",
                min_value=0.0,
                value=20.0,
                step=0.5
            )


        search_button = st.form_submit_button(
            "Find treatment settings",
            type="primary",
            use_container_width=True
        )


    # =========================================================================
    # REVERSE SEARCH
    # =========================================================================

    if search_button:

        with st.spinner("Searching suitable treatment settings..."):

            rng = np.random.default_rng(42)

            N_SEARCH = 8000

            # -------------------------------------------------------------
            # Generate candidate operating conditions
            # -------------------------------------------------------------

            candidates = pd.DataFrame({

                "Mc": np.full(
                    N_SEARCH,
                    rev_mc_percent / 100.0
                ),

                "ρod": np.full(
                    N_SEARCH,
                    rev_density
                ),

                "C(cca)": rng.uniform(
                    1.0,
                    10.0,
                    N_SEARCH
                ),

                "P(kPA)": rng.uniform(
                    1.0,
                    14.0,
                    N_SEARCH
                ),

                "V(kPA)": rng.uniform(
                    1.0,
                    8.0,
                    N_SEARCH
                ),

                "V(t)": rng.uniform(
                    10.0,
                    120.0,
                    N_SEARCH
                )
            })


            # -------------------------------------------------------------
            # Predict candidate performance
            # -------------------------------------------------------------

            try:

                (
                    lp_pred,
                    lp_sd,
                    rcca_pred,
                    rcca_sd
                ) = predict_treatment(candidates)

            except Exception:

                st.error(
                    "Unable to search for suitable treatment settings."
                )

                st.stop()


            # -------------------------------------------------------------
            # Normalized distance from target
            # -------------------------------------------------------------

            lp_scale = max(target_lp, 1.0)
            rcca_scale = max(target_rcca, 1.0)

            performance_error = np.sqrt(

                ((lp_pred - target_lp) / lp_scale) ** 2

                +

                ((rcca_pred - target_rcca) / rcca_scale) ** 2
            )


            # -------------------------------------------------------------
            # Add uncertainty penalty
            # -------------------------------------------------------------

            uncertainty_penalty = (

                (lp_sd / lp_scale)

                +

                (rcca_sd / rcca_scale)
            )


            score = (
                performance_error
                +
                0.20 * uncertainty_penalty
            )


            best_index = np.argmin(score)

            best = candidates.iloc[best_index]

            best_lp = float(lp_pred[best_index])
            best_lp_sd = float(lp_sd[best_index])

            best_rcca = float(rcca_pred[best_index])
            best_rcca_sd = float(rcca_sd[best_index])


            # -------------------------------------------------------------
            # UNCERTAINTY
            # -------------------------------------------------------------

            best_lp_margin = Z_95 * best_lp_sd
            best_rcca_margin = Z_95 * best_rcca_sd


        # =====================================================================
        # RECOMMENDED SETTINGS
        # =====================================================================

        st.markdown(
            '<div class="section-title" '
            'style="margin-top:16px;">'
            'Recommended treatment settings'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "CCA concentration",
            f"{best['C(cca)']:.2f}%"
        )

        c2.metric(
            "Pressure",
            f"{best['P(kPA)']:.2f} kPa"
        )

        c3.metric(
            "Vacuum",
            f"{best['V(kPA)']:.2f} kPa"
        )

        c4.metric(
            "Vacuum time",
            f"{best['V(t)']:.0f} min"
        )


        # =====================================================================
        # EXPECTED PERFORMANCE
        # =====================================================================

        st.markdown(
            '<div class="section-title">'
            'Expected performance'
            '</div>',
            unsafe_allow_html=True
        )

        p1, p2 = st.columns(2)


        with p1:

            st.metric(
                "Predicted penetration",
                f"{best_lp:.2f} mm"
            )

            st.caption(
                f"95% prediction range: "
                f"{max(0, best_lp - best_lp_margin):.2f} – "
                f"{best_lp + best_lp_margin:.2f} mm"
            )


        with p2:

            st.metric(
                "Predicted retention",
                f"{best_rcca:.2f} kg/m³"
            )

            st.caption(
                f"95% prediction range: "
                f"{max(0, best_rcca - best_rcca_margin):.2f} – "
                f"{best_rcca + best_rcca_margin:.2f} kg/m³"
            )


        # =====================================================================
        # TARGET COMPARISON
        # =====================================================================

        lp_difference = best_lp - target_lp
        rcca_difference = best_rcca - target_rcca

        comparison_html = f"""
<div class="summary-card">
    <strong>✓ Closest treatment solution found</strong><br>
    Penetration target:
    <strong>{target_lp:.2f} mm</strong>
    → predicted
    <strong>{best_lp:.2f} mm</strong>
    ({lp_difference:+.2f} mm)
    <br>
    Retention target:
    <strong>{target_rcca:.2f} kg/m³</strong>
    → predicted
    <strong>{best_rcca:.2f} kg/m³</strong>
    ({rcca_difference:+.2f} kg/m³)
</div>
"""

        st.markdown(
            comparison_html,
            unsafe_allow_html=True
        )


# =============================================================================
# INFORMATION
# =============================================================================

with st.expander("About the predictor"):

    st.write(
        "This decision-support tool estimates lateral preservative "
        "penetration and CCA retention from wood properties and "
        "treatment operating conditions."
    )

    st.write(
        "Prediction ranges communicate model uncertainty. "
        "Reverse prediction searches feasible operating conditions "
        "and identifies settings whose predicted performance most "
        "closely matches the selected targets."
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

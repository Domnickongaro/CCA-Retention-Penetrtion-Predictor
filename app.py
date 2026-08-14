#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# STAGE 10 — STREAMLIT DEPLOYMENT
# =============================================================================
#
# PROJECT
# -------
# Probabilistic Optimization of CCA Wood Treatment for Eucalyptus
#
# PURPOSE
# -------
# Deployment interface for the frozen Stage 09 Gaussian Process models.
#
# DEPLOYED MODELS
# ----------------
#   final_gp_Lp.joblib
#   final_gp_Rcca.joblib
#
# MODEL INPUTS
# ------------
#   Mc
#   ρod
#   C(cca)
#   P(kPA)
#   V(kPA)
#   V(t)
#
# IMPORTANT
# ---------
# The models are frozen.
# This application:
#
#   - does NOT retrain the models
#   - does NOT modify the models
#   - does NOT optimize treatment conditions
#   - does NOT generate synthetic observations
#
# It performs prediction and uncertainty visualization only.
#
# =============================================================================


from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CCA Wood Treatment Predictor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# 2. APPLICATION TITLE
# =============================================================================

st.title("🌲 CCA Wood Treatment Predictor")

st.markdown(
    """
    ### Probabilistic prediction of CCA treatment performance

    This application uses the frozen Gaussian Process surrogate models
    developed in the thesis modelling pipeline to predict:

    - **Lateral penetration (Lp)**
    - **CCA retention (R(cca))**
    - **Predictive uncertainty**
    """
)

st.info(
    """
    **Deployment mode:** The models used by this application are frozen.
    No model training, optimization, parameter estimation, or modification
    is performed inside the application.
    """
)


# =============================================================================
# 3. MODEL LOCATIONS
# =============================================================================

# The GitHub repository currently stores the models in the repository root.
#
# The additional models/ path is supported so that the application can also
# work if the repository is later reorganized.

ROOT_DIR = Path(__file__).resolve().parent

LP_MODEL_CANDIDATES = [
    ROOT_DIR / "final_gp_Lp.joblib",
    ROOT_DIR / "models" / "final_gp_Lp.joblib",
]

RCCA_MODEL_CANDIDATES = [
    ROOT_DIR / "final_gp_Rcca.joblib",
    ROOT_DIR / "models" / "final_gp_Rcca.joblib",
]


def find_model(candidates):
    """
    Return the first existing model path.
    """
    for path in candidates:
        if path.exists():
            return path

    return None


LP_MODEL_PATH = find_model(LP_MODEL_CANDIDATES)
RCCA_MODEL_PATH = find_model(RCCA_MODEL_CANDIDATES)


# =============================================================================
# 4. MODEL LOADING
# =============================================================================

@st.cache_resource
def load_models(lp_path, rcca_path):
    """
    Load the frozen Gaussian Process models.
    """

    lp_model = joblib.load(lp_path)
    rcca_model = joblib.load(rcca_path)

    return lp_model, rcca_model


if LP_MODEL_PATH is None or RCCA_MODEL_PATH is None:

    st.error("❌ Frozen model files could not be found.")

    st.markdown("### Required files")

    st.code(
        """
final_gp_Lp.joblib
final_gp_Rcca.joblib
        """.strip()
    )

    st.markdown(
        """
        Place both files in the same directory as `app.py`, as shown in
        the current GitHub repository structure.
        """
    )

    st.stop()


try:

    lp_model, rcca_model = load_models(
        LP_MODEL_PATH,
        RCCA_MODEL_PATH
    )

except Exception as error:

    st.error("❌ Unable to load the frozen Gaussian Process models.")

    st.exception(error)

    st.stop()


# =============================================================================
# 5. MODEL FEATURE STRUCTURE
# =============================================================================

EXPECTED_FEATURES = [
    "Mc",
    "ρod",
    "C(cca)",
    "P(kPA)",
    "V(kPA)",
    "V(t)"
]


def get_model_features(model):
    """
    Extract feature names from a sklearn Pipeline/model where available.
    """

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):

        for step in model.named_steps.values():

            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


lp_features = get_model_features(lp_model)
rcca_features = get_model_features(rcca_model)


# =============================================================================
# 6. MODEL VALIDATION
# =============================================================================

model_structure_valid = True

if lp_features is not None:

    if lp_features != EXPECTED_FEATURES:

        st.error(
            "The lateral penetration model does not match the expected "
            "Stage 09 feature structure."
        )

        st.write("Expected features:")
        st.code(str(EXPECTED_FEATURES))

        st.write("Loaded Lp model features:")
        st.code(str(lp_features))

        model_structure_valid = False


if rcca_features is not None:

    if rcca_features != EXPECTED_FEATURES:

        st.error(
            "The CCA retention model does not match the expected "
            "Stage 09 feature structure."
        )

        st.write("Expected features:")
        st.code(str(EXPECTED_FEATURES))

        st.write("Loaded R(cca) model features:")
        st.code(str(rcca_features))

        model_structure_valid = False


if not model_structure_valid:
    st.stop()


# =============================================================================
# 7. SIDEBAR — MODEL STATUS
# =============================================================================

with st.sidebar:

    st.header("Model Status")

    st.success("✓ Lp model loaded")

    st.success("✓ R(cca) model loaded")

    st.caption(
        f"Lp model: {LP_MODEL_PATH.name}"
    )

    st.caption(
        f"R(cca) model: {RCCA_MODEL_PATH.name}"
    )

    st.divider()

    st.header("Prediction Targets")

    st.write(
        """
        **Lp**

        Lateral preservative penetration.

        **R(cca)**

        True/net CCA preservative retention.
        """
    )

    st.divider()

    st.caption(
        "Stage 10 — Frozen-model deployment"
    )


# =============================================================================
# 8. INPUT SECTION
# =============================================================================

st.header("1. Treatment and Material Inputs")

st.write(
    """
    Enter the biological/material condition and treatment parameters.
    The values are passed directly to the frozen GP surrogate models.
    """
)


col1, col2, col3 = st.columns(3)


# =============================================================================
# 9. BIOLOGICAL / MATERIAL INPUTS
# =============================================================================

with col1:

    st.subheader("Material")

    mc = st.number_input(
        "Moisture content, Mc",
        min_value=0.0,
        value=28.0,
        step=0.5,
        format="%.2f",
        help="Moisture content represented by Mc."
    )

    rho_od = st.number_input(
        "Oven-dry density, ρod",
        min_value=0.0,
        value=725.0,
        step=10.0,
        format="%.2f",
        help="Oven-dry wood density."
    )


# =============================================================================
# 10. TREATMENT VARIABLES
# =============================================================================

with col2:

    st.subheader("Treatment")

    cca = st.number_input(
        "CCA concentration, C(cca)",
        min_value=0.0,
        value=5.5,
        step=0.1,
        format="%.4f",
        help="CCA solution concentration."
    )

    pressure = st.number_input(
        "Pressure, P(kPA)",
        min_value=0.0,
        value=7.5,
        step=0.1,
        format="%.4f",
        help="Applied treatment pressure."
    )


# =============================================================================
# 11. VACUUM / TREATMENT TIME
# =============================================================================

with col3:

    st.subheader("Process")

    vacuum = st.number_input(
        "Vacuum, V(kPA)",
        min_value=0.0,
        value=4.5,
        step=0.1,
        format="%.4f",
        help="Applied vacuum level."
    )

    treatment_time = st.number_input(
        "Treatment time, V(t)",
        min_value=0.0,
        value=10.0,
        step=0.5,
        format="%.4f",
        help="Treatment time represented by V(t)."
    )


# =============================================================================
# 12. INPUT DATASET
# =============================================================================

X = pd.DataFrame(
    [[
        mc,
        rho_od,
        cca,
        pressure,
        vacuum,
        treatment_time
    ]],
    columns=EXPECTED_FEATURES
)


# =============================================================================
# 13. INPUT SUMMARY
# =============================================================================

with st.expander("View model input vector"):

    st.dataframe(
        X,
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# 14. PREDICTION BUTTON
# =============================================================================

predict_button = st.button(
    "🔮 Predict Treatment Performance",
    type="primary",
    use_container_width=True
)


# =============================================================================
# 15. PREDICTION
# =============================================================================

if predict_button:

    try:

        # ---------------------------------------------------------------------
        # Gaussian Process predictions
        # ---------------------------------------------------------------------

        lp_prediction = lp_model.predict(
            X,
            return_std=True
        )

        rcca_prediction = rcca_model.predict(
            X,
            return_std=True
        )

        lp_mean, lp_std = lp_prediction
        rcca_mean, rcca_std = rcca_prediction

    except Exception as error:

        st.error("❌ Prediction failed.")

        st.exception(error)

        st.stop()


    # =========================================================================
    # 16. CONVERT TO SCALARS
    # =========================================================================

    lp_mean = float(
        np.asarray(lp_mean).ravel()[0]
    )

    lp_std = float(
        np.asarray(lp_std).ravel()[0]
    )

    rcca_mean = float(
        np.asarray(rcca_mean).ravel()[0]
    )

    rcca_std = float(
        np.asarray(rcca_std).ravel()[0]
    )


    # =========================================================================
    # 17. APPROXIMATE 95% PREDICTIVE INTERVALS
    # =========================================================================

    lp_lower = lp_mean - 1.96 * lp_std
    lp_upper = lp_mean + 1.96 * lp_std

    rcca_lower = rcca_mean - 1.96 * rcca_std
    rcca_upper = rcca_mean + 1.96 * rcca_std


    # =========================================================================
    # 18. RESULTS
    # =========================================================================

    st.divider()

    st.header("2. Predicted Treatment Performance")


    result_col1, result_col2 = st.columns(2)


    # -------------------------------------------------------------------------
    # LATERAL PENETRATION
    # -------------------------------------------------------------------------

    with result_col1:

        st.subheader("Lateral Penetration")

        st.metric(
            "Predicted Lp",
            f"{lp_mean:.2f} mm"
        )

        st.write(
            f"Predictive SD: **{lp_std:.2f} mm**"
        )

        st.write(
            f"Approx. 95% predictive interval: "
            f"**{lp_lower:.2f} – {lp_upper:.2f} mm**"
        )


    # -------------------------------------------------------------------------
    # CCA RETENTION
    # -------------------------------------------------------------------------

    with result_col2:

        st.subheader("CCA Retention")

        st.metric(
            "Predicted R(cca)",
            f"{rcca_mean:.2f}"
        )

        st.write(
            f"Predictive SD: **{rcca_std:.2f}**"
        )

        st.write(
            f"Approx. 95% predictive interval: "
            f"**{rcca_lower:.2f} – {rcca_upper:.2f}**"
        )


    # =========================================================================
    # 19. UNCERTAINTY INTERPRETATION
    # =========================================================================

    st.divider()

    st.header("3. Predictive Uncertainty")

    uncertainty_col1, uncertainty_col2 = st.columns(2)


    with uncertainty_col1:

        st.write("### Lateral penetration")

        lp_uncertainty = (
            lp_upper - lp_lower
        )

        st.write(
            f"Prediction interval width: "
            f"**{lp_uncertainty:.2f} mm**"
        )


    with uncertainty_col2:

        st.write("### CCA retention")

        rcca_uncertainty = (
            rcca_upper - rcca_lower
        )

        st.write(
            f"Prediction interval width: "
            f"**{rcca_uncertainty:.2f}**"
        )


    # =========================================================================
    # 20. PREDICTION SUMMARY
    # =========================================================================

    st.divider()

    st.header("4. Prediction Summary")


    prediction_summary = pd.DataFrame({

        "Quantity": [
            "Lateral penetration — mean",
            "Lateral penetration — SD",
            "Lateral penetration — lower 95%",
            "Lateral penetration — upper 95%",
            "CCA retention — mean",
            "CCA retention — SD",
            "CCA retention — lower 95%",
            "CCA retention — upper 95%"
        ],

        "Value": [
            lp_mean,
            lp_std,
            lp_lower,
            lp_upper,
            rcca_mean,
            rcca_std,
            rcca_lower,
            rcca_upper
        ]
    })


    st.dataframe(
        prediction_summary,
        use_container_width=True,
        hide_index=True
    )


    # =========================================================================
    # 21. INPUTS USED FOR THIS PREDICTION
    # =========================================================================

    st.subheader("Model Inputs Used")

    input_summary = pd.DataFrame({

        "Variable": EXPECTED_FEATURES,

        "Value": [
            mc,
            rho_od,
            cca,
            pressure,
            vacuum,
            treatment_time
        ]
    })


    st.dataframe(
        input_summary,
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# 22. MODEL INFORMATION
# =============================================================================

st.divider()

st.header("5. Deployed Model Information")

model_info_col1, model_info_col2 = st.columns(2)


with model_info_col1:

    st.subheader("Lateral Penetration")

    st.write(
        "Frozen Gaussian Process surrogate model"
    )

    st.code(
        LP_MODEL_PATH.name
    )


with model_info_col2:

    st.subheader("CCA Retention")

    st.write(
        "Frozen Gaussian Process surrogate model"
    )

    st.code(
        RCCA_MODEL_PATH.name
    )


# =============================================================================
# 23. METHODOLOGICAL NOTE
# =============================================================================

with st.expander("About this deployment"):

    st.markdown(
        """
        This application is the deployment layer of the probabilistic
        modelling framework.

        **The application does not:**

        - retrain the Gaussian Process models;
        - change model hyperparameters;
        - perform feature engineering;
        - generate new experimental observations;
        - perform NSGA-II optimization;
        - replace experimental validation.

        **The application does:**

        - accept biological/material inputs;
        - accept treatment-process inputs;
        - construct the model input vector;
        - generate predictions from the frozen GP models;
        - report predictive standard deviations;
        - calculate approximate 95% predictive intervals.
        """
    )


# =============================================================================
# 24. FOOTER
# =============================================================================

st.caption(
    """
    Stage 10 — Deployment of the uncertainty-aware Gaussian Process
    surrogate models for CCA wood treatment.
    """
)

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
# Simple deployment interface for the frozen Stage 09 Gaussian Process models.
#
# FROZEN MODELS
# -------------
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
# The scaler is already embedded inside each frozen sklearn Pipeline.
#
# This application:
#
#   - does NOT retrain the models
#   - does NOT optimize treatment conditions
#   - does NOT generate synthetic observations
#   - does NOT modify the models
#
# It only performs prediction using the frozen Stage 09 models.
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
    layout="centered"
)


# =============================================================================
# 2. MODEL PATHS
# =============================================================================

MODEL_DIR = Path("models")

LP_MODEL_PATH = (
    MODEL_DIR /
    "final_gp_Lp.joblib"
)

RCCA_MODEL_PATH = (
    MODEL_DIR /
    "final_gp_Rcca.joblib"
)


# =============================================================================
# 3. LOAD FROZEN MODELS
# =============================================================================

@st.cache_resource
def load_models():

    lp_model = joblib.load(
        LP_MODEL_PATH
    )

    rcca_model = joblib.load(
        RCCA_MODEL_PATH
    )

    return (
        lp_model,
        rcca_model
    )


# =============================================================================
# 4. APPLICATION HEADER
# =============================================================================

st.title(
    "CCA Wood Treatment Predictor"
)

st.write(
    """
    A probabilistic surrogate-model application for predicting lateral
    penetration and CCA retention in Eucalyptus wood.
    """
)

st.info(
    """
    The deployed Gaussian Process models are frozen Stage 09 models.
    This application performs prediction only and does not retrain or
    optimize the models.
    """
)


# =============================================================================
# 5. LOAD MODELS
# =============================================================================

try:

    (
        lp_model,
        rcca_model
    ) = load_models()

except Exception as error:

    st.error(
        "Unable to load the frozen GP models."
    )

    st.code(
        str(error)
    )

    st.stop()


# =============================================================================
# 6. VERIFY MODEL INPUT STRUCTURE
# =============================================================================

EXPECTED_FEATURES = [

    "Mc",
    "ρod",
    "C(cca)",
    "P(kPA)",
    "V(kPA)",
    "V(t)"
]


model_features = list(
    lp_model.feature_names_in_
)


if model_features != EXPECTED_FEATURES:

    st.error(
        "The loaded model feature structure does not match "
        "the expected Stage 09 feature structure."
    )

    st.write(
        "Expected:"
    )

    st.code(
        str(EXPECTED_FEATURES)
    )

    st.write(
        "Model:"
    )

    st.code(
        str(model_features)
    )

    st.stop()


# =============================================================================
# 7. INPUT SECTION
# =============================================================================

st.header(
    "Treatment and Material Inputs"
)

st.write(
    """
    Enter the wood material condition and treatment parameters.
    """
)


col1, col2 = st.columns(2)


# =============================================================================
# 8. BIOLOGICAL / MATERIAL INPUTS
# =============================================================================

with col1:

    mc = st.number_input(
        "Moisture content, Mc (%)",
        min_value=0.0,
        value=12.0,
        step=0.5
    )


    rho_od = st.number_input(
        "Oven-dry density, ρod",
        min_value=0.0,
        value=500.0,
        step=10.0
    )


# =============================================================================
# 9. TREATMENT INPUTS
# =============================================================================

with col2:

    cca = st.number_input(
        "CCA concentration, C(cca)",
        min_value=0.0,
        value=0.0976,
        step=0.005,
        format="%.5f"
    )


    pressure = st.number_input(
        "Pressure, P (kPa)",
        min_value=0.0,
        value=1.0007,
        step=0.1,
        format="%.4f"
    )


    vacuum = st.number_input(
        "Vacuum, V (kPa)",
        min_value=0.0,
        value=6.7860,
        step=0.1,
        format="%.4f"
    )


    treatment_time = st.number_input(
        "Treatment time, V(t)",
        min_value=0.0,
        value=10.0008,
        step=0.5,
        format="%.4f"
    )


# =============================================================================
# 10. PREDICTION BUTTON
# =============================================================================

predict_button = st.button(
    "Predict Treatment Performance",
    type="primary",
    use_container_width=True
)


# =============================================================================
# 11. PREDICTION
# =============================================================================

if predict_button:

    # -------------------------------------------------------------------------
    # EXACT STAGE 09 FEATURE ORDER
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

        columns=EXPECTED_FEATURES
    )


    try:

        # ---------------------------------------------------------------------
        # Gaussian Process predictions
        #
        # The StandardScaler is already contained inside the Pipeline.
        # ---------------------------------------------------------------------

        lp_mean, lp_std = (
            lp_model.predict(
                X,
                return_std=True
            )
        )


        rcca_mean, rcca_std = (
            rcca_model.predict(
                X,
                return_std=True
            )
        )


    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.code(
            str(error)
        )

        st.stop()


    # -------------------------------------------------------------------------
    # Extract scalar values
    # -------------------------------------------------------------------------

    lp_mean = float(
        np.asarray(
            lp_mean
        ).ravel()[0]
    )


    lp_std = float(
        np.asarray(
            lp_std
        ).ravel()[0]
    )


    rcca_mean = float(
        np.asarray(
            rcca_mean
        ).ravel()[0]
    )


    rcca_std = float(
        np.asarray(
            rcca_std
        ).ravel()[0]
    )


    # =============================================================================
    # 12. APPROXIMATE 95% PREDICTIVE INTERVALS
    # =============================================================================

    lp_lower = (
        lp_mean
        -
        1.96 * lp_std
    )


    lp_upper = (
        lp_mean
        +
        1.96 * lp_std
    )


    rcca_lower = (
        rcca_mean
        -
        1.96 * rcca_std
    )


    rcca_upper = (
        rcca_mean
        +
        1.96 * rcca_std
    )


    # =============================================================================
    # 13. DISPLAY PREDICTIONS
    # =============================================================================

    st.divider()

    st.header(
        "Predicted Treatment Performance"
    )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Lateral penetration",
            f"{lp_mean:.2f} mm"
        )

        st.write(
            f"Predictive SD: **{lp_std:.2f} mm**"
        )

        st.write(
            f"Approx. 95% interval: "
            f"**{lp_lower:.2f} – {lp_upper:.2f} mm**"
        )


    with result_col2:

        st.metric(
            "CCA retention",
            f"{rcca_mean:.2f}"
        )

        st.write(
            f"Predictive SD: **{rcca_std:.2f}**"
        )

        st.write(
            f"Approx. 95% interval: "
            f"**{rcca_lower:.2f} – {rcca_upper:.2f}**"
        )


    # =============================================================================
    # 14. INPUT SUMMARY
    # =============================================================================

    st.subheader(
        "Prediction Inputs"
    )


    input_summary = pd.DataFrame({

        "Variable": [

            "Mc",

            "ρod",

            "C(cca)",

            "P(kPA)",

            "V(kPA)",

            "V(t)"
        ],

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
# 15. STAGE 07 SELECTED ROBUST OPERATING POINT
# =============================================================================

st.divider()

st.header(
    "Stage 07 Selected Robust Operating Point"
)

st.write(
    """
    This operating point was selected during the uncertainty-aware
    decision/risk analysis. It is presented as the recommended robust
    operating reference and is not recalculated by the application.
    """
)


selected_solution = pd.DataFrame({

    "Parameter": [

        "C(cca)",

        "P(kPA)",

        "V(kPA)",

        "V(t)",

        "Expected Lp",

        "Expected Rcca",

        "Robust Rcca",

        "Failure probability",

        "Reliability"
    ],

    "Value": [

        0.097607877,

        1.0007337053215035,

        6.785953211399391,

        10.00082770306525,

        17.14914442628544,

        33.694366108766594,

        25.15888240744537,

        0.00105,

        0.99895
    ]
})


st.dataframe(
    selected_solution,
    use_container_width=True,
    hide_index=True
)


# =============================================================================
# 16. MODEL INFORMATION
# =============================================================================

st.divider()

st.subheader(
    "Deployed Model"
)


st.write(
    """
    Two frozen Gaussian Process surrogate models are deployed:
    """
)

st.write(
    "• Lateral penetration model: `final_gp_Lp.joblib`"
)

st.write(
    "• CCA retention model: `final_gp_Rcca.joblib`"
)

st.success(
    "Stage 09 frozen models loaded successfully."
)


# =============================================================================
# 17. FOOTER
# =============================================================================

st.caption(
    """
    Stage 10 — Deployment of the uncertainty-aware CCA wood treatment
    surrogate model.
    """
)


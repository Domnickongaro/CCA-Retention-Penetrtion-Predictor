#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# GP–NSGA-II UNCERTAINTY-AWARE CCA TREATMENT FRAMEWORK
# Eucalyptus Wood Treatment Decision Support
# =============================================================================

from pathlib import Path
from math import erf, sqrt

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CCA Treatment Framework",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 2. USER INTERFACE STYLING
# =============================================================================

st.markdown(
    """
    <style>

    /* ============================================================
       GLOBAL
       ============================================================ */

    .stApp {
        background: #F5F8F6;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif;
        color: #26352F;
    }

    .block-container {
    max-width: 1180px;
    padding-top: 0.35rem;
    padding-bottom: 1.25rem;
    }

    div[data-testid="stVerticalBlock"] {
    gap: 0.50rem;
    }
   header[data-testid="stHeader"] {
    background: transparent;
    }

    /* ============================================================
       SIDEBAR — JUNGLE GREEN
       ============================================================ */

    section[data-testid="stSidebar"] {
        background: #14532D !important;
        border-right: none !important;
    }

    section[data-testid="stSidebar"] > div {
        background: #14532D !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #F7FFF9 !important;
    }

    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-size: 0.82rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.11em !important;
        margin-bottom: 0.8rem !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.55rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 9px;
        padding: 0.58rem 0.65rem;
        transition: all 0.15s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.30);
    }
/* ============================================================
   SIDEBAR WIDTH
   ============================================================ */

section[data-testid="stSidebar"] {
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
}

section[data-testid="stSidebar"] > div:first-child {
    width: 300px !important;
}

/* ============================================================
   FRAMEWORK BANNER
   ============================================================ */

.framework-banner {
    background: linear-gradient(
        110deg,
        #D9F1FF 0%,
        #EAF8FF 100%
    );

    border: 1px solid #B8DFF1;
    border-left: 6px solid #14532D;

    border-radius: 11px;

    padding: 14px 20px;

    margin: 0 0 0.85rem 0;

    box-shadow:
        0 2px 8px
        rgba(26, 75, 95, 0.06);
}

.framework-name {
    color: #704522;

    font-size: 1.25rem;

    line-height: 1.25;

    font-weight: 800;

    font-style: italic;

    letter-spacing: 0.005em;

    margin: 0;
}

.framework-subtitle {
    color: #14532D;

    font-size: 0.90rem;

    font-weight: 650;

    margin-top: 3px;
}
    /* ============================================================
       HEADINGS
       ============================================================ */

    h1 {
        color: #173F2C !important;

        font-size: 1.85rem !important;

        font-weight: 760 !important;

        letter-spacing: -0.025em;

        margin-top: 0 !important;

        margin-bottom: 0.15rem !important;
    }

    h2 {
        color: #173F2C !important;

        font-size: 1.24rem !important;

        font-weight: 750 !important;

        margin-top: 0.35rem !important;
    }

    h3 {
        color: #214D39 !important;

        font-size: 1.05rem !important;

        font-weight: 730 !important;
    }


    /* ============================================================
       MAIN CARDS
       ============================================================ */

    div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;

    border: 1px solid #DCE6E0 !important;

    border-radius: 11px !important;

    box-shadow:
        0 2px 7px
        rgba(22, 65, 46, 0.04);
}


    /* ============================================================
       SECTION LABELS
       ============================================================ */

    .section-label {
        color: #6B4423;

        font-size: 0.74rem;

        font-weight: 800;

        letter-spacing: 0.10em;

        margin-top: 0.20rem;

        margin-bottom: -0.15rem;
    }


    /* ============================================================
       INPUTS
       ============================================================ */

    div[data-baseweb="input"] {
        border-radius: 8px !important;
        background: #FAFCFB;
    }

    div[data-testid="stNumberInput"] input {
        font-size: 0.95rem;
        color: #26352F;
    }

    div[data-testid="stNumberInput"] label p {
        font-weight: 650;
        color: #344B40;
    }


    /* ============================================================
       BUTTONS
       ============================================================ */

    div[data-testid="stButton"] button,
    div[data-testid="stFormSubmitButton"] button {

        min-height: 44px;

        background: #176B45 !important;

        color: #FFFFFF !important;

        border: none !important;

        border-radius: 8px !important;

        font-size: 0.95rem;

        font-weight: 750;

        box-shadow:
            0 2px 5px
            rgba(20, 83, 45, 0.14);

        transition:
            background 0.15s ease,
            transform 0.15s ease;
    }

    div[data-testid="stButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {

        background: #0F5132 !important;

        color: #FFFFFF !important;

        transform: translateY(-1px);
    }


    /* ============================================================
       METRIC CARDS
       ============================================================ */

    div[data-testid="stMetric"] {

        background: #F5FBF7;

        border: 1px solid #D5E8DC;

        border-top: 4px solid #258657;

        border-radius: 10px;

        padding: 13px 15px;

        box-shadow:
            0 2px 7px
            rgba(20, 83, 45, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #53675D;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        color: #143D2A;
        font-weight: 760;
    }


    /* ============================================================
       ALERTS
       ============================================================ */

    div[data-testid="stAlert"] {
        border-radius: 9px;
        font-size: 0.93rem;
    }


    /* ============================================================
       SMALL RESULT RANGE
       ============================================================ */

    .result-range {
        background: #F8FAF9;

        border-left: 3px solid #8BBEA2;

        border-radius: 6px;

        padding: 8px 11px;

        margin-top: 5px;

        font-size: 0.90rem;

        color: #455B50;
    }


    /* ============================================================
       FOOTER
       ============================================================ */

    .app-footer {
        text-align: center;

        color: #7A8B82;

        font-size: 0.76rem;

        margin-top: 1.2rem;

        padding-top: 0.6rem;
    }


    /* ============================================================
       HIDE STREAMLIT DECORATION
       ============================================================ */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 3. PROJECT PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent


def resolve_model_path(filename):
    """
    Allow models to be stored either beside app.py
    or inside a models/ directory.
    """

    root_file = ROOT / filename
    model_file = ROOT / "models" / filename

    if model_file.exists():
        return model_file

    return root_file


GP_LP_FILE = resolve_model_path(
    "final_gp_Lp.joblib"
)

GP_RCCA_FILE = resolve_model_path(
    "final_gp_Rcca.joblib"
)


CALIBRATION_FILE = (
    ROOT
    / "pipeline_data"
    / "stage_05_calibration_factors.csv"
)


# =============================================================================
# 4. MODEL FEATURE ORDER
# =============================================================================

PREDICTORS = [
    "Mc",
    "ρod",
    "C(cca)",
    "P(kPA)",
    "V(kPA)",
    "V(t)",
]


# =============================================================================
# 5. VALIDATED INPUT RANGES
# =============================================================================

BOUNDS = {

    "Mc_percent": (
        14.0,
        45.0,
    ),

    "rho_od": (
        445.2003,
        1005.3631,
    ),

    "CCA_percent": (
        1.0,
        10.0,
    ),

    "pressure": (
        1.0,
        14.0,
    ),

    "vacuum": (
        1.0,
        8.0,
    ),

    "duration": (
        10.0,
        120.0,
    ),
}


# =============================================================================
# 6. CONSTANTS
# =============================================================================

RETENTION_THRESHOLD = 20.0

Z_95 = 1.96

MIN_STD = 1e-12


# =============================================================================
# 7. LOAD MODELS
# =============================================================================

@st.cache_resource
def load_models():

    if not GP_LP_FILE.exists():
        raise FileNotFoundError(
            f"Missing model: {GP_LP_FILE}"
        )

    if not GP_RCCA_FILE.exists():
        raise FileNotFoundError(
            f"Missing model: {GP_RCCA_FILE}"
        )

    return {

        "Lp": joblib.load(
            GP_LP_FILE
        ),

        "R(cca)": joblib.load(
            GP_RCCA_FILE
        ),
    }


# =============================================================================
# 8. LOAD UNCERTAINTY FACTORS
# =============================================================================

@st.cache_resource
def load_uncertainty_factors():

    factors = {
        "Lp": 1.0,
        "R(cca)": 1.0,
    }

    if not CALIBRATION_FILE.exists():
        return factors

    try:

        table = pd.read_csv(
            CALIBRATION_FILE
        )

        required = {
            "Response",
            "Calibration_Factor",
        }

        if not required.issubset(
            table.columns
        ):
            return factors

        for response in factors:

            match = table.loc[
                table["Response"] == response
            ]

            if not match.empty:

                factors[response] = float(
                    match[
                        "Calibration_Factor"
                    ].iloc[0]
                )

    except Exception:
        return factors

    return factors


# =============================================================================
# 9. INITIALISE MODELS
# =============================================================================

try:

    MODELS = load_models()

    UNCERTAINTY_FACTORS = (
        load_uncertainty_factors()
    )

except Exception:

    st.error(
        "Prediction models could not be loaded."
    )

    st.stop()


# =============================================================================
# 10. NORMAL CDF
# =============================================================================

def normal_cdf(value):

    return (
        0.5
        *
        (
            1.0
            +
            erf(
                value
                /
                sqrt(2.0)
            )
        )
    )


# =============================================================================
# 11. CREATE MODEL INPUT
# =============================================================================

def create_input(
    mc_percent,
    rho_od,
    cca_percent,
    pressure,
    vacuum,
    duration,
):

    """
    User enters moisture content and CCA concentration
    as percentages.

    Frozen models expect them as fractions.
    """

    return pd.DataFrame(
        [[

            mc_percent / 100.0,

            rho_od,

            cca_percent / 100.0,

            pressure,

            vacuum,

            duration,

        ]],
        columns=PREDICTORS,
    )


# =============================================================================
# 12. PREDICT ONE RESPONSE
# =============================================================================

def predict_response(
    model,
    X_input,
    uncertainty_factor,
):

    mean, raw_std = model.predict(
        X_input,
        return_std=True,
    )

    mean = float(
        np.asarray(mean).ravel()[0]
    )

    raw_std = float(
        np.asarray(raw_std).ravel()[0]
    )

    calibrated_std = (
        max(
            raw_std,
            MIN_STD,
        )
        *
        uncertainty_factor
    )

    margin = (
        Z_95
        *
        calibrated_std
    )

    lower = (
        mean
        -
        margin
    )

    upper = (
        mean
        +
        margin
    )

    return {

        "mean": mean,

        "std": calibrated_std,

        "margin": margin,

        "lower": lower,

        "upper": upper,
    }


# =============================================================================
# 13. COMPLETE TREATMENT PREDICTION
# =============================================================================

def predict_treatment(
    mc_percent,
    rho_od,
    cca_percent,
    pressure,
    vacuum,
    duration,
):

    X = create_input(
        mc_percent,
        rho_od,
        cca_percent,
        pressure,
        vacuum,
        duration,
    )

    lp = predict_response(
        MODELS["Lp"],
        X,
        UNCERTAINTY_FACTORS["Lp"],
    )

    rcca = predict_response(
        MODELS["R(cca)"],
        X,
        UNCERTAINTY_FACTORS["R(cca)"],
    )

    z = (
        RETENTION_THRESHOLD
        -
        rcca["mean"]
    ) / max(
        rcca["std"],
        MIN_STD,
    )

    p_below = normal_cdf(z)

    return {

        "Lp": lp,

        "Rcca": rcca,

        "P_below": float(
            p_below
        ),
    }


# =============================================================================
# 14. SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        "### CHOOSE A TOOL"
    )

    mode = st.radio(
        "Choose a tool",
        [
            "Predict Treatment Results",
            "Find Treatment Settings",
        ],
        label_visibility="collapsed",
    )


# =============================================================================
# 15. FRAMEWORK BANNER
# =============================================================================

st.markdown(
    """
    <div class="framework-banner">

        <div class="framework-name">
            GP–NSGA-II Uncertainty-Aware CCA Treatment Framework
        </div>

        <div class="framework-subtitle">
            Eucalyptus Wood Treatment Decision Support
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 16. FORWARD PREDICTION
# =============================================================================

if mode == "Predict Treatment Results":

    st.title(
        "Predict Treatment Results"
    )

    # -------------------------------------------------------------------------
    # INPUT CARD
    # -------------------------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "Enter Treatment Details"
        )

        # ---------------------------------------------------------------------
        # WOOD DETAILS
        # ---------------------------------------------------------------------

        st.markdown(
            """
            <div class="section-label">
                WOOD DETAILS
            </div>
            """,
            unsafe_allow_html=True,
        )

        wood1, wood2 = st.columns(
            2
        )

        with wood1:

            mc_percent = st.number_input(
                "Moisture content (%)",
                min_value=BOUNDS[
                    "Mc_percent"
                ][0],
                max_value=BOUNDS[
                    "Mc_percent"
                ][1],
                value=28.0,
                step=1.0,
            )

        with wood2:

            rho_od = st.number_input(
                "Wood density (kg/m³)",
                min_value=BOUNDS[
                    "rho_od"
                ][0],
                max_value=BOUNDS[
                    "rho_od"
                ][1],
                value=725.0,
                step=5.0,
            )

        # ---------------------------------------------------------------------
        # TREATMENT SETTINGS
        # ---------------------------------------------------------------------

        st.markdown(
            """
            <div class="section-label">
                TREATMENT SETTINGS
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(
            4
        )

        with c1:

            cca_percent = st.number_input(
                "CCA concentration (%)",
                min_value=BOUNDS[
                    "CCA_percent"
                ][0],
                max_value=BOUNDS[
                    "CCA_percent"
                ][1],
                value=5.5,
                step=0.1,
            )

        with c2:

            pressure = st.number_input(
                "Pressure (kPa)",
                min_value=BOUNDS[
                    "pressure"
                ][0],
                max_value=BOUNDS[
                    "pressure"
                ][1],
                value=7.5,
                step=0.1,
            )

        with c3:

            vacuum = st.number_input(
                "Vacuum (kPa)",
                min_value=BOUNDS[
                    "vacuum"
                ][0],
                max_value=BOUNDS[
                    "vacuum"
                ][1],
                value=4.5,
                step=0.1,
            )

        with c4:

            duration = st.number_input(
                "Vacuum time (min)",
                min_value=BOUNDS[
                    "duration"
                ][0],
                max_value=BOUNDS[
                    "duration"
                ][1],
                value=60.0,
                step=1.0,
            )

        run_prediction = st.button(
            "Predict Results",
            type="primary",
            use_container_width=True,
        )


    # =========================================================================
    # FORWARD RESULTS
    # =========================================================================

    if run_prediction:

        try:

            result = predict_treatment(
                mc_percent,
                rho_od,
                cca_percent,
                pressure,
                vacuum,
                duration,
            )

        except Exception:

            st.error(
                "The prediction could not be completed."
            )

            st.stop()

        lp = result["Lp"]

        rcca = result["Rcca"]

        p_below = result["P_below"]


        # ---------------------------------------------------------------------
        # EXPECTED RESULTS
        # ---------------------------------------------------------------------

        st.subheader(
            "Expected Results"
        )

        result1, result2 = st.columns(
            2
        )


        # ---------------------------------------------------------------------
        # PENETRATION
        # ---------------------------------------------------------------------

        with result1:

            with st.container(
                border=True
            ):

                st.markdown(
                    """
                    <div class="section-label">
                        PENETRATION
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Expected penetration",
                    f"{lp['mean']:.2f} mm",
                )

                st.markdown(
                    f"""
                    <div class="result-range">
                        <strong>Expected range</strong><br>
                        {lp['lower']:.2f} – {lp['upper']:.2f} mm
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        # ---------------------------------------------------------------------
        # RETENTION
        # ---------------------------------------------------------------------

        with result2:

            with st.container(
                border=True
            ):

                st.markdown(
                    """
                    <div class="section-label">
                        CCA RETENTION
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Expected retention",
                    f"{rcca['mean']:.2f} kg/m³",
                )

                st.markdown(
                    f"""
                    <div class="result-range">
                        <strong>Expected range</strong><br>
                        {rcca['lower']:.2f} – {rcca['upper']:.2f} kg/m³
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        # ---------------------------------------------------------------------
        # TREATMENT CHECK
        # ---------------------------------------------------------------------

        st.subheader(
            "Treatment Check"
        )

        check1, check2 = st.columns(
            2
        )

        with check1:

            st.metric(
                "Required retention",
                f"{RETENTION_THRESHOLD:.0f} kg/m³",
            )

        with check2:

            st.metric(
                "Risk of low retention",
                f"{p_below * 100:.1f}%",
            )


        # ---------------------------------------------------------------------
        # SIMPLE INTERPRETATION
        # ---------------------------------------------------------------------

        if (
            rcca["lower"]
            >= RETENTION_THRESHOLD
        ):

            st.success(
                "✓ Retention is expected to meet the required level."
            )

        elif (
            rcca["mean"]
            >= RETENTION_THRESHOLD
        ):

            st.warning(
                "⚠ Retention may fall below the required level."
            )

        else:

            st.error(
                "✕ Retention is expected to be below the required level."
            )


# =============================================================================
# 17. TREATMENT FINDER
# =============================================================================

else:

    st.title(
        "Find Treatment Settings"
    )

    # -------------------------------------------------------------------------
    # TARGET INPUT CARD
    # -------------------------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "Set Your Target"
        )

        # ---------------------------------------------------------------------
        # WOOD DETAILS
        # ---------------------------------------------------------------------

        st.markdown(
            """
            <div class="section-label">
                WOOD DETAILS
            </div>
            """,
            unsafe_allow_html=True,
        )

        w1, w2 = st.columns(
            2
        )

        with w1:

            finder_mc = st.number_input(
                "Moisture content (%)",
                min_value=BOUNDS[
                    "Mc_percent"
                ][0],
                max_value=BOUNDS[
                    "Mc_percent"
                ][1],
                value=28.0,
                step=1.0,
                key="finder_mc",
            )

        with w2:

            finder_density = st.number_input(
                "Wood density (kg/m³)",
                min_value=BOUNDS[
                    "rho_od"
                ][0],
                max_value=BOUNDS[
                    "rho_od"
                ][1],
                value=725.0,
                step=5.0,
                key="finder_density",
            )


        # ---------------------------------------------------------------------
        # TARGET RESULTS
        # ---------------------------------------------------------------------

        st.markdown(
            """
            <div class="section-label">
                TARGET RESULTS
            </div>
            """,
            unsafe_allow_html=True,
        )

        t1, t2 = st.columns(
            2
        )

        with t1:

            target_lp = st.number_input(
                "Target penetration (mm)",
                min_value=0.0,
                value=10.0,
                step=0.5,
            )

        with t2:

            target_rcca = st.number_input(
                "Target retention (kg/m³)",
                min_value=RETENTION_THRESHOLD,
                value=RETENTION_THRESHOLD,
                step=0.5,
            )

        find_button = st.button(
            "Find Treatment Settings",
            type="primary",
            use_container_width=True,
        )


    # =========================================================================
    # TREATMENT SEARCH
    # =========================================================================

    if find_button:

        with st.spinner(
            "Finding suitable settings..."
        ):

            # -----------------------------------------------------------------
            # REPRODUCIBLE CANDIDATE SET
            # -----------------------------------------------------------------

            rng = np.random.default_rng(
                42
            )

            N_CANDIDATES = 12000


            # -----------------------------------------------------------------
            # CANDIDATE TREATMENT SETTINGS
            # -----------------------------------------------------------------

            cca_candidates = rng.uniform(
                BOUNDS[
                    "CCA_percent"
                ][0],
                BOUNDS[
                    "CCA_percent"
                ][1],
                N_CANDIDATES,
            )

            pressure_candidates = rng.uniform(
                BOUNDS[
                    "pressure"
                ][0],
                BOUNDS[
                    "pressure"
                ][1],
                N_CANDIDATES,
            )

            vacuum_candidates = rng.uniform(
                BOUNDS[
                    "vacuum"
                ][0],
                BOUNDS[
                    "vacuum"
                ][1],
                N_CANDIDATES,
            )

            duration_candidates = rng.uniform(
                BOUNDS[
                    "duration"
                ][0],
                BOUNDS[
                    "duration"
                ][1],
                N_CANDIDATES,
            )


            # -----------------------------------------------------------------
            # CREATE MODEL MATRIX
            # -----------------------------------------------------------------

            candidates = pd.DataFrame({

                "Mc":
                    np.full(
                        N_CANDIDATES,
                        finder_mc / 100.0,
                    ),

                "ρod":
                    np.full(
                        N_CANDIDATES,
                        finder_density,
                    ),

                "C(cca)":
                    cca_candidates / 100.0,

                "P(kPA)":
                    pressure_candidates,

                "V(kPA)":
                    vacuum_candidates,

                "V(t)":
                    duration_candidates,
            })


            # -----------------------------------------------------------------
            # PREDICT CANDIDATES
            # -----------------------------------------------------------------

            lp_mean, lp_std = (
                MODELS["Lp"].predict(
                    candidates,
                    return_std=True,
                )
            )

            rcca_mean, rcca_std = (
                MODELS["R(cca)"].predict(
                    candidates,
                    return_std=True,
                )
            )


            # -----------------------------------------------------------------
            # APPLY UNCERTAINTY CALIBRATION
            # -----------------------------------------------------------------

            lp_std = (
                np.maximum(
                    lp_std,
                    MIN_STD,
                )
                *
                UNCERTAINTY_FACTORS[
                    "Lp"
                ]
            )

            rcca_std = (
                np.maximum(
                    rcca_std,
                    MIN_STD,
                )
                *
                UNCERTAINTY_FACTORS[
                    "R(cca)"
                ]
            )


            # -----------------------------------------------------------------
            # 95% LOWER BOUNDS
            # -----------------------------------------------------------------

            lp_lower = (
                lp_mean
                -
                Z_95
                *
                lp_std
            )

            rcca_lower = (
                rcca_mean
                -
                Z_95
                *
                rcca_std
            )


            # -----------------------------------------------------------------
            # IDENTIFY CANDIDATES THAT MEET BOTH TARGETS
            # -----------------------------------------------------------------

            feasible = (

                (
                    lp_lower
                    >= target_lp
                )

                &

                (
                    rcca_lower
                    >= target_rcca
                )
            )

            feasible_indices = (
                np.where(
                    feasible
                )[0]
            )


            # =================================================================
            # NO SUITABLE SETTING
            # =================================================================

            if (
                len(
                    feasible_indices
                )
                == 0
            ):

                st.warning(
                    "No suitable treatment setting was found "
                    "for these targets. Try adjusting the target values."
                )


            # =================================================================
            # SUITABLE SETTINGS FOUND
            # =================================================================

            else:

                # -------------------------------------------------------------
                # NORMALISE TREATMENT SETTINGS
                # -------------------------------------------------------------

                cca_norm = (

                    cca_candidates
                    -
                    BOUNDS[
                        "CCA_percent"
                    ][0]

                ) / (

                    BOUNDS[
                        "CCA_percent"
                    ][1]
                    -
                    BOUNDS[
                        "CCA_percent"
                    ][0]
                )


                pressure_norm = (

                    pressure_candidates
                    -
                    BOUNDS[
                        "pressure"
                    ][0]

                ) / (

                    BOUNDS[
                        "pressure"
                    ][1]
                    -
                    BOUNDS[
                        "pressure"
                    ][0]
                )


                vacuum_norm = (

                    vacuum_candidates
                    -
                    BOUNDS[
                        "vacuum"
                    ][0]

                ) / (

                    BOUNDS[
                        "vacuum"
                    ][1]
                    -
                    BOUNDS[
                        "vacuum"
                    ][0]
                )


                duration_norm = (

                    duration_candidates
                    -
                    BOUNDS[
                        "duration"
                    ][0]

                ) / (

                    BOUNDS[
                        "duration"
                    ][1]
                    -
                    BOUNDS[
                        "duration"
                    ][0]
                )


                # -------------------------------------------------------------
                # RELATIVE TREATMENT INTENSITY
                # -------------------------------------------------------------

                treatment_intensity = (

                    cca_norm
                    +
                    pressure_norm
                    +
                    vacuum_norm
                    +
                    duration_norm

                ) / 4.0


                feasible_intensity = (
                    treatment_intensity[
                        feasible_indices
                    ]
                )


                best_index = (
                    feasible_indices[
                        np.argmin(
                            feasible_intensity
                        )
                    ]
                )


                # -------------------------------------------------------------
                # BEST SETTINGS
                # -------------------------------------------------------------

                best_cca = float(
                    cca_candidates[
                        best_index
                    ]
                )

                best_pressure = float(
                    pressure_candidates[
                        best_index
                    ]
                )

                best_vacuum = float(
                    vacuum_candidates[
                        best_index
                    ]
                )

                best_duration = float(
                    duration_candidates[
                        best_index
                    ]
                )


                # -------------------------------------------------------------
                # BEST EXPECTED RESULTS
                # -------------------------------------------------------------

                best_lp = float(
                    lp_mean[
                        best_index
                    ]
                )

                best_lp_std = float(
                    lp_std[
                        best_index
                    ]
                )

                best_rcca = float(
                    rcca_mean[
                        best_index
                    ]
                )

                best_rcca_std = float(
                    rcca_std[
                        best_index
                    ]
                )


                # -------------------------------------------------------------
                # RESULT RANGES
                # -------------------------------------------------------------

                best_lp_lower = (
                    best_lp
                    -
                    Z_95
                    *
                    best_lp_std
                )

                best_lp_upper = (
                    best_lp
                    +
                    Z_95
                    *
                    best_lp_std
                )

                best_rcca_lower = (
                    best_rcca
                    -
                    Z_95
                    *
                    best_rcca_std
                )

                best_rcca_upper = (
                    best_rcca
                    +
                    Z_95
                    *
                    best_rcca_std
                )


                # =============================================================
                # DISPLAY RECOMMENDATION
                # =============================================================

                st.success(
                    "✓ Suitable treatment settings found."
                )


                # -------------------------------------------------------------
                # RECOMMENDED SETTINGS
                # -------------------------------------------------------------

                st.subheader(
                    "Recommended Settings"
                )

                s1, s2, s3, s4 = st.columns(
                    4
                )

                s1.metric(
                    "CCA concentration",
                    f"{best_cca:.2f}%",
                )

                s2.metric(
                    "Pressure",
                    f"{best_pressure:.2f} kPa",
                )

                s3.metric(
                    "Vacuum",
                    f"{best_vacuum:.2f} kPa",
                )

                s4.metric(
                    "Vacuum time",
                    f"{best_duration:.0f} min",
                )


                # -------------------------------------------------------------
                # EXPECTED RESULTS
                # -------------------------------------------------------------

                st.subheader(
                    "Expected Results"
                )

                p1, p2 = st.columns(
                    2
                )


                # -------------------------------------------------------------
                # PENETRATION
                # -------------------------------------------------------------

                with p1:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            """
                            <div class="section-label">
                                PENETRATION
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.metric(
                            "Expected penetration",
                            f"{best_lp:.2f} mm",
                        )

                        st.markdown(
                            f"""
                            <div class="result-range">
                                <strong>Expected range</strong><br>
                                {best_lp_lower:.2f} – {best_lp_upper:.2f} mm
                                <br>
                                <span style="
                                    color:#6C7D74;
                                    font-size:0.82rem;
                                ">
                                    Target: {target_lp:.2f} mm
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


                # -------------------------------------------------------------
                # RETENTION
                # -------------------------------------------------------------

                with p2:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            """
                            <div class="section-label">
                                CCA RETENTION
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.metric(
                            "Expected retention",
                            f"{best_rcca:.2f} kg/m³",
                        )

                        st.markdown(
                            f"""
                            <div class="result-range">
                                <strong>Expected range</strong><br>
                                {best_rcca_lower:.2f} – {best_rcca_upper:.2f} kg/m³
                                <br>
                                <span style="
                                    color:#6C7D74;
                                    font-size:0.82rem;
                                ">
                                    Target: {target_rcca:.2f} kg/m³
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# =============================================================================
# 18. SMALL FOOTER
# =============================================================================

st.markdown(
    """
    <div class="app-footer">
        CCA Treatment Decision Support
    </div>
    """,
    unsafe_allow_html=True,
)

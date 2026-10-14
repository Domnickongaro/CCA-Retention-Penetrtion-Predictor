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
# 2. APPLICATION STYLE
# =============================================================================

st.markdown(
    """
<style>

/* ==========================================================================
   GLOBAL
   ========================================================================== */

:root {
    --jungle: #14532D;
    --jungle-dark: #0D3D22;
    --green: #197149;
    --green-soft: #EFF8F2;

    --brown: #6B4423;

    --sky: #D9F1FF;
    --sky-light: #EDF9FF;
    --sky-border: #9DD4ED;

    --page: #F5F8F6;
    --card: #FFFFFF;
    --input: #F3F6F4;

    --text: #263A30;
    --muted: #6C7B73;
    --border: #D8E3DC;
}


html,
body,
[class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}


.stApp {
    background: var(--page);
    color: var(--text);
}


/* ==========================================================================
   STREAMLIT TOP BAR
   ========================================================================== */

header[data-testid="stHeader"] {
    height: 3.5rem;
    background: rgba(255, 255, 255, 0.98);
}


/* ==========================================================================
   MAIN PAGE
   ========================================================================== */

.block-container {
    max-width: 1180px;

    padding-top: 3.9rem !important;
    padding-bottom: 1.1rem !important;

    margin-left: auto;
    margin-right: auto;
}


div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}


/* ==========================================================================
   SIDEBAR
   ========================================================================== */

section[data-testid="stSidebar"] {
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;

    background: var(--jungle) !important;

    border-right: none !important;
}


section[data-testid="stSidebar"] > div:first-child {
    width: 300px !important;
    background: var(--jungle) !important;
}


section[data-testid="stSidebar"] .block-container {
    padding-top: 6.0rem !important;
    padding-left: 1.35rem !important;
    padding-right: 1.35rem !important;
}


section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;

    font-size: 0.82rem !important;
    font-weight: 800 !important;

    letter-spacing: 0.13em;

    margin-bottom: 0.9rem !important;
}


section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}


section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.60rem;
}


section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.075);

    border:
        1px solid
        rgba(255, 255, 255, 0.17);

    border-radius: 9px;

    padding:
        0.65rem
        0.70rem;

    transition:
        all
        0.15s
        ease;
}


section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background:
        rgba(255, 255, 255, 0.14);

    border-color:
        rgba(255, 255, 255, 0.30);
}


/* ==========================================================================
   MASTER FRAMEWORK HEADER
   ========================================================================== */

.framework-banner {
    width: 100%;

    box-sizing: border-box;

    background:
        linear-gradient(
            105deg,
            var(--sky) 0%,
            var(--sky-light) 100%
        );

    border:
        1px solid
        var(--sky-border);

    border-left:
        7px solid
        var(--jungle);

    border-radius: 12px;

    padding:
        18px
        25px
        17px
        25px;

    margin:
        0
        0
        1.15rem
        0;

    box-shadow:
        0 3px 10px
        rgba(20, 83, 45, 0.06);
}


.framework-name {
    display: block;

    color: var(--brown) !important;

    font-size: 1.55rem !important;
    line-height: 1.22 !important;

    font-weight: 850 !important;
    font-style: italic;

    letter-spacing: -0.015em;

    margin: 0 !important;
    padding: 0 !important;

    opacity: 1 !important;
    visibility: visible !important;
}


.framework-subtitle {
    display: block;

    color: var(--jungle) !important;

    font-size: 0.96rem !important;
    line-height: 1.30 !important;

    font-weight: 750 !important;

    margin:
        6px
        0
        0
        0 !important;

    padding: 0 !important;
}


/* ==========================================================================
   PAGE HEADINGS
   ========================================================================== */

h1 {
    color: #153E2A !important;

    font-size: 1.72rem !important;

    font-weight: 800 !important;

    letter-spacing: -0.025em;

    margin-top: 0.15rem !important;
    margin-bottom: 0.35rem !important;
}


h2 {
    color: #173F2C !important;

    font-size: 1.18rem !important;

    font-weight: 760 !important;

    margin-top: 0.30rem !important;
    margin-bottom: 0.20rem !important;
}


h3 {
    color: #214D39 !important;

    font-size: 1.02rem !important;

    font-weight: 730 !important;
}


/* ==========================================================================
   CARDS
   ========================================================================== */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);

    border:
        1px solid
        var(--border) !important;

    border-radius:
        12px !important;

    box-shadow:
        0 2px 8px
        rgba(20, 83, 45, 0.04);
}


/* ==========================================================================
   SMALL SECTION LABEL
   ========================================================================== */

.section-label {
    color: var(--brown);

    font-size: 0.72rem;

    font-weight: 850;

    letter-spacing: 0.12em;

    margin-top: 0.15rem;
    margin-bottom: -0.10rem;
}


/* ==========================================================================
   INPUTS
   ========================================================================== */

div[data-baseweb="input"] {
    background:
        var(--input) !important;

    border-radius:
        8px !important;
}


div[data-testid="stNumberInput"] input {
    color: #25392F;

    font-size: 0.94rem;
}


div[data-testid="stNumberInput"] label p {
    color: #344B40;

    font-weight: 650;

    font-size: 0.90rem;
}


/* ==========================================================================
   BUTTON
   ========================================================================== */

div[data-testid="stButton"] button {
    min-height: 44px;

    background:
        var(--green) !important;

    color:
        #FFFFFF !important;

    border:
        none !important;

    border-radius:
        8px !important;

    font-size:
        0.94rem;

    font-weight:
        750;

    box-shadow:
        0 2px 5px
        rgba(20, 83, 45, 0.14);

    transition:
        all
        0.15s
        ease;
}


div[data-testid="stButton"] button:hover {
    background:
        var(--jungle) !important;

    color:
        #FFFFFF !important;

    transform:
        translateY(-1px);
}


/* ==========================================================================
   METRICS
   ========================================================================== */

div[data-testid="stMetric"] {
    background:
        #F4FAF6;

    border:
        1px solid
        #D5E7DB;

    border-top:
        4px solid
        #258657;

    border-radius:
        10px;

    padding:
        12px
        14px;
}


div[data-testid="stMetricLabel"] {
    color:
        #52665B;

    font-weight:
        650;
}


div[data-testid="stMetricValue"] {
    color:
        #143D2A;

    font-weight:
        780;
}


/* ==========================================================================
   RESULT RANGE
   ========================================================================== */

.result-range {
    background:
        #F1F9FD;

    border-left:
        4px solid
        #5BA7C9;

    border-radius:
        7px;

    padding:
        9px
        11px;

    margin-top:
        6px;

    color:
        #455B50;

    font-size:
        0.88rem;

    line-height:
        1.45;
}


/* ==========================================================================
   STATUS CARDS
   ========================================================================== */

.status-good {
    background:
        #EAF7EE;

    border-left:
        5px solid
        #218653;

    border-radius:
        8px;

    padding:
        11px
        14px;

    color:
        #17492F;

    font-weight:
        650;
}


.status-watch {
    background:
        #FFF7E5;

    border-left:
        5px solid
        #C58A19;

    border-radius:
        8px;

    padding:
        11px
        14px;

    color:
        #6A4B10;

    font-weight:
        650;
}


.status-low {
    background:
        #FFF0EE;

    border-left:
        5px solid
        #B94A3D;

    border-radius:
        8px;

    padding:
        11px
        14px;

    color:
        #713229;

    font-weight:
        650;
}


/* ==========================================================================
   FOOTER
   ========================================================================== */

.app-footer {
    text-align:
        center;

    color:
        #829087;

    font-size:
        0.74rem;

    margin-top:
        1.0rem;

    padding-top:
        0.25rem;
}


/* ==========================================================================
   RESPONSIVE
   ========================================================================== */

@media (max-width: 900px) {

    section[data-testid="stSidebar"] {
        width: 270px !important;
        min-width: 270px !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        width: 270px !important;
    }

    .framework-name {
        font-size:
            1.20rem !important;
    }

}


/* ==========================================================================
   REMOVE UNNECESSARY STREAMLIT DECORATION
   ========================================================================== */

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
# 3. FILE PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent


def resolve_model_path(filename):

    model_path = (
        ROOT
        / "models"
        / filename
    )

    root_path = (
        ROOT
        / filename
    )

    if model_path.exists():

        return model_path

    return root_path


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
# 4. MODEL FEATURES
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
# 5. VALIDATED OPERATING RANGES
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
# 6. DECISION SETTINGS
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
# 8. LOAD UNCERTAINTY CALIBRATION
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

        required_columns = {
            "Response",
            "Calibration_Factor",
        }

        if not required_columns.issubset(
            table.columns
        ):

            return factors

        for response in factors:

            match = table.loc[
                table["Response"]
                ==
                response
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
# 9. INITIALISE
# =============================================================================

try:

    MODELS = load_models()

    UNCERTAINTY_FACTORS = (
        load_uncertainty_factors()
    )

except Exception:

    st.error(
        "The prediction system could not be loaded."
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
# 11. PREPARE MODEL INPUT
# =============================================================================

def create_input(
    mc_percent,
    rho_od,
    cca_percent,
    pressure,
    vacuum,
    duration,
):

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
        np.asarray(
            mean
        ).ravel()[0]
    )

    raw_std = float(
        np.asarray(
            raw_std
        ).ravel()[0]
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

    return {

        "mean":
            mean,

        "std":
            calibrated_std,

        "margin":
            margin,

        "lower":
            mean - margin,

        "upper":
            mean + margin,
    }


# =============================================================================
# 13. COMPLETE PREDICTION
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

    penetration = predict_response(
        MODELS["Lp"],
        X,
        UNCERTAINTY_FACTORS["Lp"],
    )

    retention = predict_response(
        MODELS["R(cca)"],
        X,
        UNCERTAINTY_FACTORS["R(cca)"],
    )

    z = (
        RETENTION_THRESHOLD
        -
        retention["mean"]
    ) / max(
        retention["std"],
        MIN_STD,
    )

    p_below = normal_cdf(
        z
    )

    return {

        "Lp":
            penetration,

        "Rcca":
            retention,

        "P_below":
            float(
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
# 15. MASTER HEADER
# =============================================================================

banner_html = (
    '<div class="framework-banner">'
    '<div class="framework-name">'
    'GP–NSGA-II Uncertainty-Aware CCA Treatment Framework'
    '</div>'
    '<div class="framework-subtitle">'
    'Eucalyptus Wood Treatment Decision Support'
    '</div>'
    '</div>'
)

st.markdown(
    banner_html,
    unsafe_allow_html=True,
)


# =============================================================================
# 16. FORWARD PREDICTION
# =============================================================================

if mode == "Predict Treatment Results":

    st.title(
        "Treatment Performance"
    )

    # -------------------------------------------------------------------------
    # INPUT CARD
    # -------------------------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "Treatment Inputs"
        )

        # ---------------------------------------------------------------------
        # WOOD
        # ---------------------------------------------------------------------

        st.markdown(
            '<div class="section-label">'
            'WOOD'
            '</div>',
            unsafe_allow_html=True,
        )

        wood_col1, wood_col2 = st.columns(
            2,
            gap="medium",
        )

        with wood_col1:

            mc_percent = st.number_input(
                "Moisture content (%)",
                min_value=
                    BOUNDS["Mc_percent"][0],
                max_value=
                    BOUNDS["Mc_percent"][1],
                value=28.0,
                step=1.0,
            )

        with wood_col2:

            rho_od = st.number_input(
                "Wood density (kg/m³)",
                min_value=
                    BOUNDS["rho_od"][0],
                max_value=
                    BOUNDS["rho_od"][1],
                value=725.0,
                step=5.0,
            )


        # ---------------------------------------------------------------------
        # TREATMENT
        # ---------------------------------------------------------------------

        st.markdown(
            '<div class="section-label">'
            'TREATMENT'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(
            4,
            gap="medium",
        )

        with col1:

            cca_percent = st.number_input(
                "CCA concentration (%)",
                min_value=
                    BOUNDS["CCA_percent"][0],
                max_value=
                    BOUNDS["CCA_percent"][1],
                value=5.5,
                step=0.1,
            )

        with col2:

            pressure = st.number_input(
                "Pressure (kPa)",
                min_value=
                    BOUNDS["pressure"][0],
                max_value=
                    BOUNDS["pressure"][1],
                value=7.5,
                step=0.1,
            )

        with col3:

            vacuum = st.number_input(
                "Vacuum (kPa)",
                min_value=
                    BOUNDS["vacuum"][0],
                max_value=
                    BOUNDS["vacuum"][1],
                value=4.5,
                step=0.1,
            )

        with col4:

            duration = st.number_input(
                "Vacuum time (min)",
                min_value=
                    BOUNDS["duration"][0],
                max_value=
                    BOUNDS["duration"][1],
                value=60.0,
                step=1.0,
            )


        predict_button = st.button(
            "Predict Treatment Performance",
            type="primary",
            use_container_width=True,
        )


    # =========================================================================
    # FORWARD RESULTS
    # =========================================================================

    if predict_button:

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


        penetration = result["Lp"]

        retention = result["Rcca"]

        p_below = result["P_below"]


        # ---------------------------------------------------------------------
        # PREDICTED PERFORMANCE
        # ---------------------------------------------------------------------

        st.subheader(
            "Predicted Performance"
        )

        result_col1, result_col2 = st.columns(
            2,
            gap="medium",
        )


        # ---------------------------------------------------------------------
        # PENETRATION
        # ---------------------------------------------------------------------

        with result_col1:

            with st.container(
                border=True
            ):

                st.markdown(
                    '<div class="section-label">'
                    'PENETRATION'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Expected penetration",
                    f"{penetration['mean']:.2f} mm",
                )

                st.markdown(
                    (
                        '<div class="result-range">'
                        '<strong>Likely range</strong><br>'
                        f"{penetration['lower']:.2f}"
                        " – "
                        f"{penetration['upper']:.2f}"
                        " mm"
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )


        # ---------------------------------------------------------------------
        # RETENTION
        # ---------------------------------------------------------------------

        with result_col2:

            with st.container(
                border=True
            ):

                st.markdown(
                    '<div class="section-label">'
                    'CCA RETENTION'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Expected retention",
                    f"{retention['mean']:.2f} kg/m³",
                )

                st.markdown(
                    (
                        '<div class="result-range">'
                        '<strong>Likely range</strong><br>'
                        f"{retention['lower']:.2f}"
                        " – "
                        f"{retention['upper']:.2f}"
                        " kg/m³"
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )


        # ---------------------------------------------------------------------
        # RETENTION CHECK
        # ---------------------------------------------------------------------

        st.subheader(
            "Retention Check"
        )

        check_col1, check_col2 = st.columns(
            2,
            gap="medium",
        )

        with check_col1:

            st.metric(
                "Required retention",
                f"{RETENTION_THRESHOLD:.0f} kg/m³",
            )

        with check_col2:

            st.metric(
                "Chance of falling below requirement",
                f"{p_below * 100:.1f}%",
            )


        # ---------------------------------------------------------------------
        # INTERPRETATION
        # ---------------------------------------------------------------------

        if (
            retention["lower"]
            >=
            RETENTION_THRESHOLD
        ):

            st.markdown(
                (
                    '<div class="status-good">'
                    '✓ This treatment is expected to meet '
                    'the retention requirement.'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

        elif (
            retention["mean"]
            >=
            RETENTION_THRESHOLD
        ):

            st.markdown(
                (
                    '<div class="status-watch">'
                    '⚠ The expected retention meets the '
                    'requirement, but there is some uncertainty.'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                (
                    '<div class="status-low">'
                    '✕ This treatment is expected to fall '
                    'below the retention requirement.'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )


# =============================================================================
# 17. REVERSE PREDICTION / TREATMENT FINDER
# =============================================================================

else:

    st.title(
        "Find Treatment Settings"
    )

    # -------------------------------------------------------------------------
    # TARGET CARD
    # -------------------------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "Treatment Target"
        )


        # ---------------------------------------------------------------------
        # WOOD
        # ---------------------------------------------------------------------

        st.markdown(
            '<div class="section-label">'
            'WOOD'
            '</div>',
            unsafe_allow_html=True,
        )

        wood_col1, wood_col2 = st.columns(
            2,
            gap="medium",
        )

        with wood_col1:

            finder_mc = st.number_input(
                "Moisture content (%)",
                min_value=
                    BOUNDS["Mc_percent"][0],
                max_value=
                    BOUNDS["Mc_percent"][1],
                value=28.0,
                step=1.0,
                key="finder_mc",
            )

        with wood_col2:

            finder_density = st.number_input(
                "Wood density (kg/m³)",
                min_value=
                    BOUNDS["rho_od"][0],
                max_value=
                    BOUNDS["rho_od"][1],
                value=725.0,
                step=5.0,
                key="finder_density",
            )


        # ---------------------------------------------------------------------
        # TARGET PERFORMANCE
        # ---------------------------------------------------------------------

        st.markdown(
            '<div class="section-label">'
            'TARGET PERFORMANCE'
            '</div>',
            unsafe_allow_html=True,
        )

        target_col1, target_col2 = st.columns(
            2,
            gap="medium",
        )

        with target_col1:

            target_penetration = st.number_input(
                "Required penetration (mm)",
                min_value=0.0,
                value=10.0,
                step=0.5,
            )

        with target_col2:

            target_retention = st.number_input(
                "Required retention (kg/m³)",
                min_value=RETENTION_THRESHOLD,
                value=RETENTION_THRESHOLD,
                step=0.5,
            )


        find_button = st.button(
            "Find Suitable Treatment",
            type="primary",
            use_container_width=True,
        )


    # =========================================================================
    # FIND TREATMENT
    # =========================================================================

    if find_button:

        with st.spinner(
            "Finding suitable treatment settings..."
        ):

            # -----------------------------------------------------------------
            # REPRODUCIBLE SEARCH
            # -----------------------------------------------------------------

            rng = np.random.default_rng(
                42
            )

            N_CANDIDATES = 12000


            # -----------------------------------------------------------------
            # CREATE CANDIDATE SETTINGS
            # -----------------------------------------------------------------

            cca_candidates = rng.uniform(
                BOUNDS["CCA_percent"][0],
                BOUNDS["CCA_percent"][1],
                N_CANDIDATES,
            )

            pressure_candidates = rng.uniform(
                BOUNDS["pressure"][0],
                BOUNDS["pressure"][1],
                N_CANDIDATES,
            )

            vacuum_candidates = rng.uniform(
                BOUNDS["vacuum"][0],
                BOUNDS["vacuum"][1],
                N_CANDIDATES,
            )

            duration_candidates = rng.uniform(
                BOUNDS["duration"][0],
                BOUNDS["duration"][1],
                N_CANDIDATES,
            )


            # -----------------------------------------------------------------
            # BUILD MODEL MATRIX
            # -----------------------------------------------------------------

            candidates = pd.DataFrame(
                {

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
                }
            )


            # -----------------------------------------------------------------
            # PREDICT PENETRATION
            # -----------------------------------------------------------------

            penetration_mean, penetration_std = (
                MODELS["Lp"].predict(
                    candidates,
                    return_std=True,
                )
            )


            # -----------------------------------------------------------------
            # PREDICT RETENTION
            # -----------------------------------------------------------------

            retention_mean, retention_std = (
                MODELS["R(cca)"].predict(
                    candidates,
                    return_std=True,
                )
            )


            # -----------------------------------------------------------------
            # CALIBRATE UNCERTAINTY
            # -----------------------------------------------------------------

            penetration_std = (
                np.maximum(
                    penetration_std,
                    MIN_STD,
                )
                *
                UNCERTAINTY_FACTORS["Lp"]
            )

            retention_std = (
                np.maximum(
                    retention_std,
                    MIN_STD,
                )
                *
                UNCERTAINTY_FACTORS["R(cca)"]
            )


            # -----------------------------------------------------------------
            # LOWER PREDICTION LIMITS
            # -----------------------------------------------------------------

            penetration_lower = (
                penetration_mean
                -
                Z_95
                *
                penetration_std
            )

            retention_lower = (
                retention_mean
                -
                Z_95
                *
                retention_std
            )


            # -----------------------------------------------------------------
            # FIND SETTINGS MEETING BOTH TARGETS
            # -----------------------------------------------------------------

            feasible = (
                (
                    penetration_lower
                    >=
                    target_penetration
                )
                &
                (
                    retention_lower
                    >=
                    target_retention
                )
            )

            feasible_indices = np.where(
                feasible
            )[0]


            # -----------------------------------------------------------------
            # NO SOLUTION
            # -----------------------------------------------------------------

            if (
                len(
                    feasible_indices
                )
                ==
                0
            ):

                st.warning(
                    "No suitable treatment setting was found "
                    "for these targets. Try adjusting one of "
                    "the required values."
                )


            # -----------------------------------------------------------------
            # SOLUTION FOUND
            # -----------------------------------------------------------------

            else:

                # -------------------------------------------------------------
                # NORMALISE TREATMENT VARIABLES
                # -------------------------------------------------------------

                cca_norm = (
                    (
                        cca_candidates
                        -
                        BOUNDS["CCA_percent"][0]
                    )
                    /
                    (
                        BOUNDS["CCA_percent"][1]
                        -
                        BOUNDS["CCA_percent"][0]
                    )
                )

                pressure_norm = (
                    (
                        pressure_candidates
                        -
                        BOUNDS["pressure"][0]
                    )
                    /
                    (
                        BOUNDS["pressure"][1]
                        -
                        BOUNDS["pressure"][0]
                    )
                )

                vacuum_norm = (
                    (
                        vacuum_candidates
                        -
                        BOUNDS["vacuum"][0]
                    )
                    /
                    (
                        BOUNDS["vacuum"][1]
                        -
                        BOUNDS["vacuum"][0]
                    )
                )

                duration_norm = (
                    (
                        duration_candidates
                        -
                        BOUNDS["duration"][0]
                    )
                    /
                    (
                        BOUNDS["duration"][1]
                        -
                        BOUNDS["duration"][0]
                    )
                )


                # -------------------------------------------------------------
                # TREATMENT INTENSITY
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


                # -------------------------------------------------------------
                # SELECT LOWEST-INTENSITY FEASIBLE SETTING
                # -------------------------------------------------------------

                best_index = feasible_indices[
                    np.argmin(
                        treatment_intensity[
                            feasible_indices
                        ]
                    )
                ]


                # -------------------------------------------------------------
                # RECOMMENDED SETTINGS
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
                # EXPECTED PERFORMANCE
                # -------------------------------------------------------------

                best_penetration = float(
                    penetration_mean[
                        best_index
                    ]
                )

                best_penetration_std = float(
                    penetration_std[
                        best_index
                    ]
                )

                best_retention = float(
                    retention_mean[
                        best_index
                    ]
                )

                best_retention_std = float(
                    retention_std[
                        best_index
                    ]
                )


                # -------------------------------------------------------------
                # UNCERTAINTY RANGES
                # -------------------------------------------------------------

                best_penetration_lower = (
                    best_penetration
                    -
                    Z_95
                    *
                    best_penetration_std
                )

                best_penetration_upper = (
                    best_penetration
                    +
                    Z_95
                    *
                    best_penetration_std
                )

                best_retention_lower = (
                    best_retention
                    -
                    Z_95
                    *
                    best_retention_std
                )

                best_retention_upper = (
                    best_retention
                    +
                    Z_95
                    *
                    best_retention_std
                )


                # =============================================================
                # DISPLAY RECOMMENDATION
                # =============================================================

                st.markdown(
                    (
                        '<div class="status-good">'
                        '✓ Suitable treatment found.'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )


                # -------------------------------------------------------------
                # SETTINGS
                # -------------------------------------------------------------

                st.subheader(
                    "Recommended Treatment"
                )

                set1, set2, set3, set4 = st.columns(
                    4,
                    gap="medium",
                )

                set1.metric(
                    "CCA concentration",
                    f"{best_cca:.2f}%",
                )

                set2.metric(
                    "Pressure",
                    f"{best_pressure:.2f} kPa",
                )

                set3.metric(
                    "Vacuum",
                    f"{best_vacuum:.2f} kPa",
                )

                set4.metric(
                    "Vacuum time",
                    f"{best_duration:.0f} min",
                )


                # -------------------------------------------------------------
                # EXPECTED PERFORMANCE
                # -------------------------------------------------------------

                st.subheader(
                    "Expected Performance"
                )

                perf1, perf2 = st.columns(
                    2,
                    gap="medium",
                )


                # -------------------------------------------------------------
                # PENETRATION
                # -------------------------------------------------------------

                with perf1:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            (
                                '<div class="section-label">'
                                'PENETRATION'
                                '</div>'
                            ),
                            unsafe_allow_html=True,
                        )

                        st.metric(
                            "Expected penetration",
                            f"{best_penetration:.2f} mm",
                        )

                        st.markdown(
                            (
                                '<div class="result-range">'
                                '<strong>Likely range</strong><br>'
                                f"{best_penetration_lower:.2f}"
                                " – "
                                f"{best_penetration_upper:.2f}"
                                " mm"
                                "<br>"
                                "Required: "
                                f"{target_penetration:.2f} mm"
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )


                # -------------------------------------------------------------
                # RETENTION
                # -------------------------------------------------------------

                with perf2:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            (
                                '<div class="section-label">'
                                'CCA RETENTION'
                                '</div>'
                            ),
                            unsafe_allow_html=True,
                        )

                        st.metric(
                            "Expected retention",
                            f"{best_retention:.2f} kg/m³",
                        )

                        st.markdown(
                            (
                                '<div class="result-range">'
                                '<strong>Likely range</strong><br>'
                                f"{best_retention_lower:.2f}"
                                " – "
                                f"{best_retention_upper:.2f}"
                                " kg/m³"
                                "<br>"
                                "Required: "
                                f"{target_retention:.2f} kg/m³"
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )


# =============================================================================
# 18. FOOTER
# =============================================================================

st.markdown(
    (
        '<div class="app-footer">'
        'CCA Wood Treatment Decision Support'
        '</div>'
    ),
    unsafe_allow_html=True,
)

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
# 2. COLOUR PALETTE
# =============================================================================

JUNGLE = "#14532D"
JUNGLE_DARK = "#0B3D22"
GREEN = "#1F7A4D"
SKY = "#DDF3FF"
SKY_BORDER = "#A9D8EF"
BROWN = "#704522"
TEXT = "#24382E"
MUTED = "#66776E"
PAGE_BG = "#F4F8F5"
CARD_BG = "#FFFFFF"
SOFT_GREEN = "#EEF8F1"
SOFT_BLUE = "#F1F9FD"
BORDER = "#D9E5DE"


# =============================================================================
# 3. CSS
# =============================================================================

st.markdown(
    """
<style>

/* --------------------------------------------------------------------------
   APP
   -------------------------------------------------------------------------- */

.stApp {
    background: #F4F8F5;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
    color: #24382E;
}

.block-container {
    max-width: 1180px;
    padding-top: 0.8rem;
    padding-bottom: 1.2rem;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}


/* --------------------------------------------------------------------------
   TOP STREAMLIT AREA
   -------------------------------------------------------------------------- */

header[data-testid="stHeader"] {
    background: rgba(255,255,255,0.96);
}


/* --------------------------------------------------------------------------
   SIDEBAR
   -------------------------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: #14532D !important;
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
    border-right: none !important;
}

section[data-testid="stSidebar"] > div:first-child {
    width: 300px !important;
    background: #14532D !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2.2rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}

section[data-testid="stSidebar"] h3 {
    color: white !important;
    font-size: 0.82rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.12em !important;
    margin-bottom: 0.9rem !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #F8FFF9 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.55rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.17);
    border-radius: 9px;
    padding: 0.65rem 0.65rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.15);
}


/* ============================================================
   MASTER FRAMEWORK HEADER
   ============================================================ */

.framework-banner {
    background: linear-gradient(
        105deg,
        #CDEEFF 0%,
        #E8F7FF 100%
    );

    border: 1px solid #9FD4ED;
    border-left: 7px solid #14532D;

    border-radius: 12px;

    padding: 18px 24px 17px 24px;

    margin-top: 0.35rem;
    margin-bottom: 1.25rem;

    box-shadow:
        0 3px 10px
        rgba(20, 83, 45, 0.07);

    position: relative;
    z-index: 1;
}

.framework-name {
    color: #704522 !important;

    font-size: 1.55rem !important;
    line-height: 1.25 !important;

    font-weight: 850 !important;
    font-style: italic;

    letter-spacing: -0.01em;

    margin: 0 0 5px 0 !important;
    padding: 0 !important;

    opacity: 1 !important;
}

.framework-subtitle {
    color: #14532D !important;

    font-size: 0.96rem !important;
    line-height: 1.30 !important;

    font-weight: 750 !important;

    margin: 0 !important;
    padding: 0 !important;

    opacity: 1 !important;
}
/* --------------------------------------------------------------------------
   PAGE TITLES
   -------------------------------------------------------------------------- */

h1 {
    color: #173F2C !important;
    font-size: 1.70rem !important;
    font-weight: 780 !important;
    letter-spacing: -0.025em;
    margin-top: 0.1rem !important;
    margin-bottom: 0.2rem !important;
}

h2 {
    color: #173F2C !important;
    font-size: 1.16rem !important;
    font-weight: 750 !important;
    margin-top: 0.25rem !important;
    margin-bottom: 0.20rem !important;
}

h3 {
    color: #214D39 !important;
    font-size: 1.02rem !important;
    font-weight: 730 !important;
}


/* --------------------------------------------------------------------------
   CARDS
   -------------------------------------------------------------------------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #D9E5DE !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(20,83,45,0.045);
}


/* --------------------------------------------------------------------------
   SECTION LABEL
   -------------------------------------------------------------------------- */

.section-label {
    color: #704522;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.11em;
    margin-top: 0.10rem;
    margin-bottom: -0.10rem;
}


/* --------------------------------------------------------------------------
   INPUTS
   -------------------------------------------------------------------------- */

div[data-baseweb="input"] {
    background: #F8FAF9 !important;
    border-radius: 8px !important;
}

div[data-testid="stNumberInput"] input {
    font-size: 0.94rem;
    color: #24382E;
}

div[data-testid="stNumberInput"] label p {
    color: #344B40;
    font-weight: 650;
}


/* --------------------------------------------------------------------------
   BUTTONS
   -------------------------------------------------------------------------- */

div[data-testid="stButton"] button {
    min-height: 44px;
    background: #176B45 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.94rem;
    font-weight: 750;
    box-shadow: 0 2px 5px rgba(20,83,45,0.14);
}

div[data-testid="stButton"] button:hover {
    background: #0F5132 !important;
    color: white !important;
}


/* --------------------------------------------------------------------------
   METRICS
   -------------------------------------------------------------------------- */

div[data-testid="stMetric"] {
    background: #F4FAF6;
    border: 1px solid #D7E9DD;
    border-top: 4px solid #258657;
    border-radius: 10px;
    padding: 12px 14px;
}

div[data-testid="stMetricLabel"] {
    color: #53675D;
    font-weight: 650;
}

div[data-testid="stMetricValue"] {
    color: #143D2A;
    font-weight: 760;
}


/* --------------------------------------------------------------------------
   RESULT RANGE
   -------------------------------------------------------------------------- */

.result-range {
    background: #F1F9FD;
    border-left: 3px solid #5BA7C9;
    border-radius: 6px;
    padding: 8px 11px;
    margin-top: 5px;
    font-size: 0.88rem;
    color: #455B50;
}


/* --------------------------------------------------------------------------
   STATUS CARD
   -------------------------------------------------------------------------- */

.status-good {
    background: #EAF7EE;
    border-left: 5px solid #218653;
    border-radius: 8px;
    padding: 11px 14px;
    color: #17492F;
    font-weight: 650;
}

.status-watch {
    background: #FFF7E5;
    border-left: 5px solid #C58A19;
    border-radius: 8px;
    padding: 11px 14px;
    color: #6A4B10;
    font-weight: 650;
}

.status-low {
    background: #FFF0EE;
    border-left: 5px solid #B94A3D;
    border-radius: 8px;
    padding: 11px 14px;
    color: #713229;
    font-weight: 650;
}


/* --------------------------------------------------------------------------
   FOOTER
   -------------------------------------------------------------------------- */

.app-footer {
    text-align: center;
    color: #819087;
    font-size: 0.75rem;
    margin-top: 1rem;
}


/* --------------------------------------------------------------------------
   MOBILE / SMALL SCREENS
   -------------------------------------------------------------------------- */

@media (max-width: 900px) {

    section[data-testid="stSidebar"] {
        width: 270px !important;
        min-width: 270px !important;
    }

    .framework-name {
        font-size: 1.08rem;
    }

}


/* --------------------------------------------------------------------------
   HIDE UNNECESSARY STREAMLIT DECORATION
   -------------------------------------------------------------------------- */

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
# 4. PROJECT PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent


def resolve_model_path(filename):

    model_folder = ROOT / "models" / filename
    root_file = ROOT / filename

    if model_folder.exists():
        return model_folder

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
# 5. MODEL FEATURES
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
# 6. VALID INPUT RANGES
# =============================================================================

BOUNDS = {

    "Mc_percent": (14.0, 45.0),

    "rho_od": (
        445.2003,
        1005.3631
    ),

    "CCA_percent": (
        1.0,
        10.0
    ),

    "pressure": (
        1.0,
        14.0
    ),

    "vacuum": (
        1.0,
        8.0
    ),

    "duration": (
        10.0,
        120.0
    ),
}


# =============================================================================
# 7. CONSTANTS
# =============================================================================

RETENTION_THRESHOLD = 20.0

Z_95 = 1.96

MIN_STD = 1e-12


# =============================================================================
# 8. LOAD MODELS
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
# 9. LOAD CALIBRATION FACTORS
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
            "Calibration_Factor"
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
# 10. INITIALISE
# =============================================================================

try:

    MODELS = load_models()

    UNCERTAINTY_FACTORS = (
        load_uncertainty_factors()
    )

except Exception:

    st.error(
        "The prediction models could not be loaded."
    )

    st.stop()


# =============================================================================
# 11. SUPPORT FUNCTIONS
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
            MIN_STD
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

        "mean": mean,

        "std": calibrated_std,

        "margin": margin,

        "lower": mean - margin,

        "upper": mean + margin,
    }


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
        MIN_STD
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
# 12. SIDEBAR
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
# 13. FRAMEWORK HEADER
# =============================================================================
#
# IMPORTANT:
# Keep this HTML on ONE physical Python string.
# This prevents Streamlit from displaying the HTML tags.
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
# 14. FORWARD PREDICTION
# =============================================================================

if mode == "Predict Treatment Results":

    st.title(
        "Predict Treatment Results"
    )

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
            '<div class="section-label">WOOD</div>',
            unsafe_allow_html=True,
        )

        wood1, wood2 = st.columns(
            2,
            gap="medium"
        )

        with wood1:

            mc_percent = st.number_input(
                "Moisture content (%)",
                min_value=BOUNDS["Mc_percent"][0],
                max_value=BOUNDS["Mc_percent"][1],
                value=28.0,
                step=1.0,
            )

        with wood2:

            rho_od = st.number_input(
                "Wood density (kg/m³)",
                min_value=BOUNDS["rho_od"][0],
                max_value=BOUNDS["rho_od"][1],
                value=725.0,
                step=5.0,
            )

        # ---------------------------------------------------------------------
        # TREATMENT
        # ---------------------------------------------------------------------

        st.markdown(
            '<div class="section-label">TREATMENT</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(
            4,
            gap="medium"
        )

        with c1:

            cca_percent = st.number_input(
                "CCA concentration (%)",
                min_value=BOUNDS["CCA_percent"][0],
                max_value=BOUNDS["CCA_percent"][1],
                value=5.5,
                step=0.1,
            )

        with c2:

            pressure = st.number_input(
                "Pressure (kPa)",
                min_value=BOUNDS["pressure"][0],
                max_value=BOUNDS["pressure"][1],
                value=7.5,
                step=0.1,
            )

        with c3:

            vacuum = st.number_input(
                "Vacuum (kPa)",
                min_value=BOUNDS["vacuum"][0],
                max_value=BOUNDS["vacuum"][1],
                value=4.5,
                step=0.1,
            )

        with c4:

            duration = st.number_input(
                "Vacuum time (min)",
                min_value=BOUNDS["duration"][0],
                max_value=BOUNDS["duration"][1],
                value=60.0,
                step=1.0,
            )

        run_prediction = st.button(
            "Predict Treatment Performance",
            type="primary",
            use_container_width=True,
        )


    # =========================================================================
    # RESULTS
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
        # PERFORMANCE
        # ---------------------------------------------------------------------

        st.subheader(
            "Predicted Performance"
        )

        result1, result2 = st.columns(
            2,
            gap="medium"
        )


        # ---------------------------------------------------------------------
        # PENETRATION
        # ---------------------------------------------------------------------

        with result1:

            with st.container(
                border=True
            ):

                st.markdown(
                    '<div class="section-label">PENETRATION</div>',
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Expected penetration",
                    f"{lp['mean']:.2f} mm",
                )

                st.markdown(
                    (
                        '<div class="result-range">'
                        '<strong>Likely range</strong><br>'
                        f"{lp['lower']:.2f} – "
                        f"{lp['upper']:.2f} mm"
                        '</div>'
                    ),
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
                    '<div class="section-label">CCA RETENTION</div>',
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Expected retention",
                    f"{rcca['mean']:.2f} kg/m³",
                )

                st.markdown(
                    (
                        '<div class="result-range">'
                        '<strong>Likely range</strong><br>'
                        f"{rcca['lower']:.2f} – "
                        f"{rcca['upper']:.2f} kg/m³"
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

        check1, check2 = st.columns(
            2,
            gap="medium"
        )

        with check1:

            st.metric(
                "Required retention",
                f"{RETENTION_THRESHOLD:.0f} kg/m³",
            )

        with check2:

            st.metric(
                "Chance of falling below requirement",
                f"{p_below * 100:.1f}%",
            )


        # ---------------------------------------------------------------------
        # SIMPLE USER MESSAGE
        # ---------------------------------------------------------------------

        if (
            rcca["lower"]
            >= RETENTION_THRESHOLD
        ):

            st.markdown(
                '<div class="status-good">'
                '✓ This treatment is expected to meet the retention requirement.'
                '</div>',
                unsafe_allow_html=True,
            )

        elif (
            rcca["mean"]
            >= RETENTION_THRESHOLD
        ):

            st.markdown(
                '<div class="status-watch">'
                '⚠ The expected retention meets the requirement, '
                'but there is some uncertainty.'
                '</div>',
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<div class="status-low">'
                '✕ This treatment is expected to fall below '
                'the retention requirement.'
                '</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# 15. TREATMENT FINDER
# =============================================================================

else:

    st.title(
        "Find Treatment Settings"
    )

    with st.container(
        border=True
    ):

        st.subheader(
            "Set Your Target"
        )

        # ---------------------------------------------------------------------
        # WOOD
        # ---------------------------------------------------------------------

        st.markdown(
            '<div class="section-label">WOOD</div>',
            unsafe_allow_html=True,
        )

        w1, w2 = st.columns(
            2,
            gap="medium"
        )

        with w1:

            finder_mc = st.number_input(
                "Moisture content (%)",
                min_value=BOUNDS["Mc_percent"][0],
                max_value=BOUNDS["Mc_percent"][1],
                value=28.0,
                step=1.0,
                key="finder_mc",
            )

        with w2:

            finder_density = st.number_input(
                "Wood density (kg/m³)",
                min_value=BOUNDS["rho_od"][0],
                max_value=BOUNDS["rho_od"][1],
                value=725.0,
                step=5.0,
                key="finder_density",
            )


        # ---------------------------------------------------------------------
        # TARGETS
        # ---------------------------------------------------------------------

        st.markdown(
            '<div class="section-label">TARGET PERFORMANCE</div>',
            unsafe_allow_html=True,
        )

        t1, t2 = st.columns(
            2,
            gap="medium"
        )

        with t1:

            target_lp = st.number_input(
                "Required penetration (mm)",
                min_value=0.0,
                value=10.0,
                step=0.5,
            )

        with t2:

            target_rcca = st.number_input(
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
    # SEARCH
    # =========================================================================

    if find_button:

        with st.spinner(
            "Finding suitable treatment settings..."
        ):

            rng = np.random.default_rng(
                42
            )

            N_CANDIDATES = 12000


            # -----------------------------------------------------------------
            # CANDIDATES
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
            # MODEL MATRIX
            # -----------------------------------------------------------------

            candidates = pd.DataFrame(
                {
                    "Mc": np.full(
                        N_CANDIDATES,
                        finder_mc / 100.0,
                    ),

                    "ρod": np.full(
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
            # PREDICTIONS
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
            # CALIBRATED UNCERTAINTY
            # -----------------------------------------------------------------

            lp_std = (
                np.maximum(
                    lp_std,
                    MIN_STD
                )
                *
                UNCERTAINTY_FACTORS["Lp"]
            )

            rcca_std = (
                np.maximum(
                    rcca_std,
                    MIN_STD
                )
                *
                UNCERTAINTY_FACTORS["R(cca)"]
            )


            # -----------------------------------------------------------------
            # CONSERVATIVE LOWER LIMITS
            # -----------------------------------------------------------------

            lp_lower = (
                lp_mean
                -
                Z_95 * lp_std
            )

            rcca_lower = (
                rcca_mean
                -
                Z_95 * rcca_std
            )


            # -----------------------------------------------------------------
            # SUITABLE CANDIDATES
            # -----------------------------------------------------------------

            feasible = (
                (lp_lower >= target_lp)
                &
                (rcca_lower >= target_rcca)
            )

            feasible_indices = np.where(
                feasible
            )[0]


            # -----------------------------------------------------------------
            # NONE FOUND
            # -----------------------------------------------------------------

            if len(feasible_indices) == 0:

                st.warning(
                    "No treatment setting was found that meets "
                    "both targets with the required confidence. "
                    "Try reducing one of the target values."
                )


            # -----------------------------------------------------------------
            # SELECT LOWER-INTENSITY SOLUTION
            # -----------------------------------------------------------------

            else:

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

                treatment_intensity = (
                    cca_norm
                    +
                    pressure_norm
                    +
                    vacuum_norm
                    +
                    duration_norm
                ) / 4.0

                best_index = feasible_indices[
                    np.argmin(
                        treatment_intensity[
                            feasible_indices
                        ]
                    )
                ]


                # -------------------------------------------------------------
                # SELECTED SETTINGS
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
                # SELECTED PERFORMANCE
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


                best_lp_lower = (
                    best_lp
                    -
                    Z_95 * best_lp_std
                )

                best_lp_upper = (
                    best_lp
                    +
                    Z_95 * best_lp_std
                )

                best_rcca_lower = (
                    best_rcca
                    -
                    Z_95 * best_rcca_std
                )

                best_rcca_upper = (
                    best_rcca
                    +
                    Z_95 * best_rcca_std
                )


                # =============================================================
                # DISPLAY
                # =============================================================

                st.markdown(
                    '<div class="status-good">'
                    '✓ Suitable treatment found.'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.subheader(
                    "Recommended Treatment"
                )

                s1, s2, s3, s4 = st.columns(
                    4,
                    gap="medium"
                )

                s1.metric(
                    "CCA concentration",
                    f"{best_cca:.2f}%"
                )

                s2.metric(
                    "Pressure",
                    f"{best_pressure:.2f} kPa"
                )

                s3.metric(
                    "Vacuum",
                    f"{best_vacuum:.2f} kPa"
                )

                s4.metric(
                    "Vacuum time",
                    f"{best_duration:.0f} min"
                )


                # -------------------------------------------------------------
                # EXPECTED PERFORMANCE
                # -------------------------------------------------------------

                st.subheader(
                    "Expected Performance"
                )

                p1, p2 = st.columns(
                    2,
                    gap="medium"
                )

                with p1:

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
                            f"{best_lp:.2f} mm",
                        )

                        st.markdown(
                            (
                                '<div class="result-range">'
                                '<strong>Likely range</strong><br>'
                                f"{best_lp_lower:.2f} – "
                                f"{best_lp_upper:.2f} mm"
                                '<br>'
                                f"Required: {target_lp:.2f} mm"
                                '</div>'
                            ),
                            unsafe_allow_html=True,
                        )


                with p2:

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
                            f"{best_rcca:.2f} kg/m³",
                        )

                        st.markdown(
                            (
                                '<div class="result-range">'
                                '<strong>Likely range</strong><br>'
                                f"{best_rcca_lower:.2f} – "
                                f"{best_rcca_upper:.2f} kg/m³"
                                '<br>'
                                f"Required: {target_rcca:.2f} kg/m³"
                                '</div>'
                            ),
                            unsafe_allow_html=True,
                        )


# =============================================================================
# 16. FOOTER
# =============================================================================

st.markdown(
    '<div class="app-footer">'
    'CCA Wood Treatment Decision Support'
    '</div>',
    unsafe_allow_html=True,
)

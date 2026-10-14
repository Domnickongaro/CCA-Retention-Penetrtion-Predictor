#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# CCA WOOD TREATMENT DECISION SUPPORT
# Frozen GP surrogate deployment
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
    page_title="CCA Treatment Decision Support",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 2. PROFESSIONAL UI STYLING
# =============================================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------------
       GLOBAL LAYOUT
       ------------------------------------------------------------- */

    .stApp {
        background-color: #F6F8F7;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.25rem;
        padding-bottom: 1.5rem;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif;
    }


    /* -------------------------------------------------------------
       SIDEBAR
       ------------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #EEF4F0;
        border-right: 1px solid #D7E2DB;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }


    /* -------------------------------------------------------------
       HEADINGS
       ------------------------------------------------------------- */

    h1 {
        color: #153C2E;
        font-weight: 750;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        color: #1C4938;
        font-weight: 700;
    }


    /* -------------------------------------------------------------
       INPUT CONTAINERS
       ------------------------------------------------------------- */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }


    /* -------------------------------------------------------------
       INPUT FIELDS
       ------------------------------------------------------------- */

    div[data-baseweb="input"] {
        border-radius: 9px;
    }

    div[data-testid="stNumberInput"] input {
        font-size: 0.96rem;
    }


    /* -------------------------------------------------------------
       BUTTON
       ------------------------------------------------------------- */

    div[data-testid="stButton"] button,
    div[data-testid="stFormSubmitButton"] button {

        min-height: 45px;

        border-radius: 9px;

        font-weight: 700;

        border: none;
    }


    /* -------------------------------------------------------------
       METRIC CARDS
       ------------------------------------------------------------- */

    div[data-testid="stMetric"] {

        background-color: #FFFFFF;

        border: 1px solid #DCE5DF;

        border-top: 4px solid #287653;

        border-radius: 12px;

        padding: 15px 17px;

        box-shadow:
            0 2px 7px
            rgba(20, 60, 45, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #53645C;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        color: #153C2E;
        font-weight: 720;
    }


    /* -------------------------------------------------------------
       ALERTS
       ------------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* -------------------------------------------------------------
       EXPANDERS
       ------------------------------------------------------------- */

    div[data-testid="stExpander"] {

        background-color: #FFFFFF;

        border: 1px solid #DFE6E1;

        border-radius: 10px;
    }


    /* -------------------------------------------------------------
       REDUCE EXCESSIVE SPACING
       ------------------------------------------------------------- */

    div[data-testid="stVerticalBlock"] {
        gap: 0.75rem;
    }


    /* -------------------------------------------------------------
       HIDE STREAMLIT DECORATION
       ------------------------------------------------------------- */

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


# Support both repository structures:
#
# repository/
# ├── app.py
# ├── final_gp_Lp.joblib
# └── final_gp_Rcca.joblib
#
# OR
#
# repository/
# ├── app.py
# └── models/
#     ├── final_gp_Lp.joblib
#     └── final_gp_Rcca.joblib


def resolve_model_path(filename):

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
# 4. FROZEN FEATURE ORDER
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
# 5. VALIDATED EXPERIMENTAL DOMAIN
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
# 6. STUDY CONSTANTS
# =============================================================================

RETENTION_THRESHOLD = 20.0

Z_95 = 1.96

MIN_STD = 1e-12


# =============================================================================
# 7. LOAD FROZEN MODELS
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

        "Lp":
            joblib.load(
                GP_LP_FILE
            ),

        "R(cca)":
            joblib.load(
                GP_RCCA_FILE
            ),
    }


# =============================================================================
# 8. LOAD UNCERTAINTY CALIBRATION
# =============================================================================

@st.cache_resource
def load_uncertainty_factors():

    # Frozen Stage 05 default
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
                table["Response"]
                == response
            ]


            if not match.empty:

                factors[response] = float(
                    match[
                        "Calibration_Factor"
                    ]
                    .iloc[0]
                )


    except Exception:

        return factors


    return factors


# =============================================================================
# 9. LOAD ARTIFACTS
# =============================================================================

try:

    MODELS = load_models()

    UNCERTAINTY_FACTORS = (
        load_uncertainty_factors()
    )


except Exception:

    st.error(
        "The frozen prediction models could not be loaded."
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
    Convert user-facing percentages to the exact
    fractional representation expected by the frozen models.
    """

    return pd.DataFrame(
        [[

            # 28% -> 0.28
            mc_percent / 100.0,

            rho_od,

            # 5.5% -> 0.055
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
        np.asarray(mean)
        .ravel()[0]
    )


    raw_std = float(
        np.asarray(raw_std)
        .ravel()[0]
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

        "mean":
            mean,

        "std":
            calibrated_std,

        "margin":
            margin,

        "lower":
            lower,

        "upper":
            upper,
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


    # Conditional probability for this exact
    # material/treatment combination.

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

        "P_below":
            float(p_below),
    }


# =============================================================================
# 14. SIDEBAR
# =============================================================================

with st.sidebar:

    st.title(
        "🌲 CCA Treatment"
    )

    st.caption(
        "Wood treatment decision support"
    )


    st.divider()


    mode = st.radio(

        "Tool",

        [
            "Predict performance",
            "Treatment finder",
        ],

        captions=[

            "Evaluate known treatment settings",

            "Find settings for required performance",
        ],
    )


    st.divider()


    if mode == "Predict performance":

        st.markdown(
            "**Current task**"
        )

        st.caption(
            "Enter the timber condition and treatment "
            "settings to estimate penetration, retention "
            "and prediction uncertainty."
        )


    else:

        st.markdown(
            "**Current task**"
        )

        st.caption(
            "Enter the timber condition and required "
            "performance to identify suitable treatment "
            "settings within the experimental domain."
        )


    st.divider()


    with st.expander(
        "Validated ranges"
    ):

        st.caption(
            "Moisture: 14–45%"
        )

        st.caption(
            "Density: 445.2–1005.4 kg/m³"
        )

        st.caption(
            "CCA: 1–10%"
        )

        st.caption(
            "Pressure: 1–14 kPa"
        )

        st.caption(
            "Vacuum: 1–8 kPa"
        )

        st.caption(
            "Vacuum time: 10–120 min"
        )


    with st.expander(
        "About uncertainty"
    ):

        st.caption(
            "The 95% prediction interval represents "
            "uncertainty in the GP prediction for the "
            "exact conditions entered."
        )


# =============================================================================
# 15. MAIN HEADER
# =============================================================================

st.title(
    "CCA Wood Treatment"
)

st.caption(
    "Prediction and treatment-setting support for "
    "Eucalyptus CCA treatment."
)


# =============================================================================
# 16. FORWARD PREDICTION
# =============================================================================

if mode == "Predict performance":

    # -------------------------------------------------------------------------
    # INPUT PANEL
    # -------------------------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "Treatment conditions"
        )


        # =====================================================================
        # MATERIAL
        # =====================================================================

        st.caption(
            "WOOD CONDITION"
        )


        wood1, wood2 = st.columns(2)


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

                "Oven-dry density (kg/m³)",

                min_value=BOUNDS[
                    "rho_od"
                ][0],

                max_value=BOUNDS[
                    "rho_od"
                ][1],

                value=725.0,

                step=5.0,
            )


        st.caption(
            "TREATMENT SETTINGS"
        )


        c1, c2, c3, c4 = (
            st.columns(4)
        )


        with c1:

            cca_percent = (
                st.number_input(

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
            )


        with c2:

            pressure = (
                st.number_input(

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
            )


        with c3:

            vacuum = (
                st.number_input(

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
            )


        with c4:

            duration = (
                st.number_input(

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
            )


        run_prediction = st.button(

            "Predict performance",

            type="primary",

            use_container_width=True,
        )


    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------

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


        except Exception as error:

            st.error(
                "Prediction could not be completed."
            )

            with st.expander(
                "Technical details"
            ):

                st.exception(
                    error
                )

            st.stop()


        lp = result["Lp"]

        rcca = result["Rcca"]

        p_below = result["P_below"]


        # =====================================================================
        # MAIN RESULTS
        # =====================================================================

        st.subheader(
            "Predicted performance"
        )


        result1, result2 = (
            st.columns(2)
        )


        with result1:

            with st.container(
                border=True
            ):

                st.caption(
                    "PRESERVATIVE PENETRATION"
                )

                st.metric(

                    "Expected penetration",

                    f"{lp['mean']:.2f} mm",
                )


                st.write(
                    "**95% prediction interval**"
                )

                st.write(
                    f"{lp['lower']:.2f} – "
                    f"{lp['upper']:.2f} mm"
                )


                st.caption(
                    "This range expresses prediction "
                    "uncertainty for the selected conditions."
                )


        with result2:

            with st.container(
                border=True
            ):

                st.caption(
                    "CCA RETENTION"
                )

                st.metric(

                    "Expected retention",

                    f"{rcca['mean']:.2f} kg/m³",
                )


                st.write(
                    "**95% prediction interval**"
                )

                st.write(
                    f"{rcca['lower']:.2f} – "
                    f"{rcca['upper']:.2f} kg/m³"
                )


                st.caption(
                    "This range expresses prediction "
                    "uncertainty for the selected conditions."
                )


        # =====================================================================
        # RETENTION REQUIREMENT
        # =====================================================================

        st.subheader(
            "Retention assessment"
        )


        a1, a2 = st.columns(
            [1, 1]
        )


        with a1:

            st.metric(

                "Study requirement",

                f"{RETENTION_THRESHOLD:.0f} kg/m³",
            )


        with a2:

            st.metric(

                "Probability below requirement",

                f"{p_below * 100:.1f}%",
            )


        # ---------------------------------------------------------------------
        # USER-FRIENDLY INTERPRETATION
        # ---------------------------------------------------------------------

        if rcca["lower"] >= RETENTION_THRESHOLD:

            st.success(
                "The complete 95% prediction interval "
                "is above the study retention requirement "
                "for these specific conditions."
            )


        elif rcca["mean"] >= RETENTION_THRESHOLD:

            st.warning(
                "The expected retention is above the study "
                "requirement, but prediction uncertainty "
                "extends below the requirement."
            )


        else:

            st.error(
                "The expected retention is below the study "
                "requirement. Consider adjusting the treatment "
                "conditions."
            )


        # =====================================================================
        # TECHNICAL DETAILS — HIDDEN
        # =====================================================================

        with st.expander(
            "Prediction details"
        ):

            d1, d2 = st.columns(2)


            with d1:

                st.write(
                    "**Penetration uncertainty**"
                )

                st.write(
                    f"Predictive SD: "
                    f"{lp['std']:.3f} mm"
                )

                st.write(
                    f"95% margin: "
                    f"±{lp['margin']:.3f} mm"
                )


            with d2:

                st.write(
                    "**Retention uncertainty**"
                )

                st.write(
                    f"Predictive SD: "
                    f"{rcca['std']:.3f} kg/m³"
                )

                st.write(
                    f"95% margin: "
                    f"±{rcca['margin']:.3f} kg/m³"
                )


# =============================================================================
# 17. TREATMENT FINDER
# =============================================================================

else:

    st.info(
        "Treatment Finder searches the validated treatment "
        "domain for settings that match the requested performance "
        "for the specified wood condition. It is a conditional "
        "target-matching tool, not the population-level robust "
        "NSGA-II decision analysis."
    )


    # -------------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "Required treatment performance"
        )


        st.caption(
            "WOOD CONDITION"
        )


        w1, w2 = st.columns(2)


        with w1:

            finder_mc = st.number_input(

                "Moisture content (%)",

                min_value=14.0,

                max_value=45.0,

                value=28.0,

                step=1.0,

                key="finder_mc",
            )


        with w2:

            finder_density = (
                st.number_input(

                    "Oven-dry density (kg/m³)",

                    min_value=445.2003,

                    max_value=1005.3631,

                    value=725.0,

                    step=5.0,

                    key="finder_density",
                )
            )


        st.caption(
            "PERFORMANCE TARGETS"
        )


        t1, t2 = st.columns(2)


        with t1:

            target_lp = (
                st.number_input(

                    "Minimum penetration (mm)",

                    min_value=0.0,

                    value=10.0,

                    step=0.5,
                )
            )


        with t2:

            target_rcca = (
                st.number_input(

                    "Minimum retention (kg/m³)",

                    min_value=RETENTION_THRESHOLD,

                    value=RETENTION_THRESHOLD,

                    step=0.5,

                    help=(
                        "The frozen study uses "
                        "20 kg/m³ as the retention requirement."
                    ),
                )
            )


        find_button = st.button(

            "Find suitable settings",

            type="primary",

            use_container_width=True,
        )


    # -------------------------------------------------------------------------
    # TARGET SEARCH
    # -------------------------------------------------------------------------

    if find_button:

        with st.spinner(
            "Evaluating treatment settings..."
        ):

            # =============================================================
            # REPRODUCIBLE CANDIDATE GRID
            # =============================================================
            #
            # This is NOT labelled NSGA-II.
            #
            # It performs deterministic target matching over a
            # reproducible space-filling candidate set.
            # =============================================================

            rng = np.random.default_rng(
                42
            )


            N_CANDIDATES = 12000


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


            # =============================================================
            # MODEL MATRIX
            # =============================================================

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


            # =============================================================
            # VECTOR PREDICTIONS
            # =============================================================

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


            # =============================================================
            # CONSERVATIVE LOWER BOUNDS
            # =============================================================

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


            # =============================================================
            # FEASIBILITY SCREEN
            # =============================================================

            feasible = (

                (lp_lower >= target_lp)

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


            # =============================================================
            # NO FEASIBLE SOLUTION
            # =============================================================

            if len(
                feasible_indices
            ) == 0:

                st.warning(
                    "No candidate treatment setting was found "
                    "whose complete 95% prediction interval "
                    "satisfied both requested targets for this "
                    "wood condition."
                )


            # =============================================================
            # SELECT LOWEST RELATIVE TREATMENT INTENSITY
            # =============================================================

            else:

                # Normalize controllable settings
                # using experimental bounds.

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


                # =========================================================
                # BEST CANDIDATE
                # =========================================================

                best_cca = (
                    cca_candidates[
                        best_index
                    ]
                )

                best_pressure = (
                    pressure_candidates[
                        best_index
                    ]
                )

                best_vacuum = (
                    vacuum_candidates[
                        best_index
                    ]
                )

                best_duration = (
                    duration_candidates[
                        best_index
                    ]
                )


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


                best_lp_lower = float(
                    lp_lower[
                        best_index
                    ]
                )


                best_lp_upper = (

                    best_lp
                    +
                    Z_95
                    *
                    best_lp_std
                )


                best_rcca_lower = float(
                    rcca_lower[
                        best_index
                    ]
                )


                best_rcca_upper = (

                    best_rcca
                    +
                    Z_95
                    *
                    best_rcca_std
                )


                # =========================================================
                # RESULTS
                # =========================================================

                st.success(
                    "A candidate setting satisfying both "
                    "targets across its 95% prediction "
                    "interval was identified."
                )


                st.subheader(
                    "Suggested settings"
                )


                s1, s2, s3, s4 = (
                    st.columns(4)
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


                # =========================================================
                # PERFORMANCE
                # =========================================================

                st.subheader(
                    "Expected performance"
                )


                p1, p2 = (
                    st.columns(2)
                )


                with p1:

                    with st.container(
                        border=True
                    ):

                        st.metric(

                            "Penetration",

                            f"{best_lp:.2f} mm",
                        )


                        st.write(
                            "**95% prediction interval**"
                        )

                        st.write(
                            f"{best_lp_lower:.2f} – "
                            f"{best_lp_upper:.2f} mm"
                        )


                        st.caption(
                            f"Required minimum: "
                            f"{target_lp:.2f} mm"
                        )


                with p2:

                    with st.container(
                        border=True
                    ):

                        st.metric(

                            "CCA retention",

                            f"{best_rcca:.2f} kg/m³",
                        )


                        st.write(
                            "**95% prediction interval**"
                        )

                        st.write(
                            f"{best_rcca_lower:.2f} – "
                            f"{best_rcca_upper:.2f} kg/m³"
                        )


                        st.caption(
                            f"Required minimum: "
                            f"{target_rcca:.2f} kg/m³"
                        )


                st.caption(
                    "The suggested setting is the lowest-intensity "
                    "candidate among the sampled settings satisfying "
                    "both conservative prediction requirements. "
                    "It should not be interpreted as the Stage 07 "
                    "population-level robust setpoint."
                )


# =============================================================================
# 18. ABOUT
# =============================================================================

with st.expander(
    "About this tool"
):

    st.write(
        "This application uses frozen Gaussian Process surrogate "
        "models developed from the experimental Eucalyptus CCA "
        "treatment dataset."
    )


    st.write(
        "Predict Performance evaluates a specific combination of "
        "wood properties and treatment settings. Treatment Finder "
        "searches for candidate settings for a specified wood "
        "condition while remaining within the validated "
        "experimental domain."
    )


    st.caption(
        "The application is a research decision-support tool and "
        "does not replace treatment standards, laboratory "
        "verification, engineering judgement or safety procedures."
    )

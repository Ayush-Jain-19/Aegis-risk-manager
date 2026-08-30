"""
src/ui/app.py

Streamlit dashboard for the AI Risk Manager fraud API. Sends a transaction
payload to POST /v1/predict-fraud and renders the result: a prominent
action banner, the SHAP explanation for the score, and — critically — a
visibly correct handling of the API's 422 graceful-failure path so a demo
can prove the system doesn't crash on bad input.

IMPORTANT NOTE ON FIELD NAMES
--------------------------------
The task description names the payload fields after the RAW Sparkov CSV
columns (amt, lat, long, merch_lat, merch_long, trans_date_trans_time).
Those are the dataset's column names, not the API's contract — the actual
`TransactionInput` schema built in src/api/schemas.py uses `amount`,
`customer_lat`, `customer_long`, `merchant_lat`, `merchant_long`, and
`trans_time` (a checkout SDK doesn't send "cc_num-style" raw dataset
columns; it sends a synchronous request with directly relevant fields).
This file builds requests using the REAL schema's field names, since a
payload built with the raw CSV names would fail the API's OWN validation
with a 422 "field required" error on `amount`/`customer_lat`/
`customer_long`/`trans_time` before ever reaching the model — sending
mislabeled fields would make the "malformed payload" demo trigger by
accident rather than by the deliberate bad-input tests it's meant to show.
The UI labels below are written in the more familiar/raw terms where it
helps usability, with a caption clarifying the mapping.

UI NOTE (this revision)
--------------------------------
This pass only touches presentation: the injected CSS, the widget layout
(primary form moved from the sidebar into the main body; the sidebar is
now reserved for backend health, the input-mode toggle, and read-only
global settings), the decision banner markup, and the Altair color/theme
config for the SHAP chart. None of the request-building, schema field
names, status-code branching, or fallback-rendering logic below has been
changed from the original.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
API_BASE_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/v1/predict-fraud"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
REQUEST_TIMEOUT_SECONDS = 10  # a checkout-flow-facing call must fail fast,
                                # not hang the merchant's page — 10s is a
                                # generous ceiling for a synchronous
                                # inference call, not a "wait forever" value.

# Design tokens for the decision banner + SHAP chart. Centralized here so
# the banner color and the chart's "toward FRAUD" color always agree.
RISK_PALETTE = {
    "APPROVE": {
        "label": "Approve",
        "emoji": "✅",
        "text": "#22C55E",
        "bg": "rgba(34, 197, 94, 0.12)",
        "border": "rgba(34, 197, 94, 0.35)",
    },
    "REVIEW": {
        "label": "Review",
        "emoji": "🕵️",
        "text": "#F59E0B",
        "bg": "rgba(245, 158, 11, 0.12)",
        "border": "rgba(245, 158, 11, 0.35)",
    },
    "BLOCK": {
        "label": "Block",
        "emoji": "⛔",
        "text": "#EF4444",
        "bg": "rgba(239, 68, 68, 0.14)",
        "border": "rgba(239, 68, 68, 0.40)",
    },
    "UNKNOWN": {
        "label": "Unknown",
        "emoji": "ℹ️",
        "text": "#8A94A6",
        "bg": "rgba(138, 148, 166, 0.12)",
        "border": "rgba(138, 148, 166, 0.35)",
    },
}

st.set_page_config(page_title="AI Risk Manager", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-base: #111318;
        --bg-elevated: #1A1D24;
        --bg-elevated-2: #20242C;
        --border-subtle: #262B35;
        --text-primary: #F5F7FA;
        --text-muted: #8A94A6;
        --accent-blue-start: #2E6BFF;
        --accent-blue-end: #1348D6;
    }

    html, body, [class*="css"], .stApp, button, input, textarea, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Rich charcoal, not flat black — a faint radial gradient for depth */
    .stApp {
        background: radial-gradient(circle at 15% 0%, #14161d 0%, #111318 45%, #0F1116 100%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: #0D0F14;
        border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        color: var(--text-muted);
    }
    .brand-mark {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.01em;
        padding: 0.25rem 0 0.75rem 0;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }
    h1 { font-weight: 800; }
    p, span, .stCaption, small {
        color: var(--text-muted);
    }
    .section-eyebrow {
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin: 0.9rem 0 0.35rem 0;
    }

    /* ---------- Elevated cards (bordered containers) ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
    }

    /* ---------- Inputs ---------- */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stDateInput>div>div>input,
    .stTimeInput>div>div>input,
    .stTextArea textarea {
        background-color: var(--bg-elevated-2);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        font-weight: 500;
    }
    .stSelectbox>div>div, div[data-baseweb="select"]>div {
        background-color: var(--bg-elevated-2) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    .stTextInput label, .stNumberInput label, .stDateInput label,
    .stTimeInput label, .stSelectbox label, .stTextArea label {
        color: var(--text-muted) !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stWidgetLabel"] p {
        color: var(--text-muted) !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ---------- Primary button ---------- */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent-blue-start), var(--accent-blue-end));
        color: white;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.75rem;
        box-shadow: 0 8px 22px rgba(46, 107, 255, 0.32);
        transition: all 0.18s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(46, 107, 255, 0.42);
        color: white;
    }
    .stButton>button p { color: white; font-weight: 600; }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background-color: var(--bg-elevated-2);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    /* ---------- Decision banner ---------- */
    .decision-banner {
        width: 100%;
        border-radius: 14px;
        padding: 1.35rem 1.75rem;
        margin-bottom: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .decision-banner .decision-emoji { font-size: 1.6rem; line-height: 1; }
    .decision-banner .decision-copy .decision-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.85;
    }
    .decision-banner .decision-copy .decision-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        line-height: 1.15;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Backend connectivity check
# WHY: a merchant-facing risk dashboard should never silently hang if the
# API is down — surfacing connectivity status up front turns "why is
# nothing happening when I click the button" into an immediate, legible
# signal instead of a confusing timeout later.
# --------------------------------------------------------------------------- #
def check_backend_health() -> tuple[bool, dict | str]:
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=3)
        resp.raise_for_status()
        return True, resp.json()
    except requests.exceptions.RequestException as exc:
        return False, str(exc)


# --------------------------------------------------------------------------- #
# Sidebar — health status, input-mode toggle, and read-only global settings
# only. The transaction form itself now lives in the main body (see
# render_structured_form / render_raw_json_editor / main).
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    st.sidebar.markdown('<div class="brand-mark">🛡️ AI Risk Manager</div>', unsafe_allow_html=True)

    healthy, health_info = check_backend_health()
    with st.sidebar.container(border=True):
        if healthy:
            st.success(f"Backend connected\n\nthreshold = {health_info.get('threshold')}")
        else:
            st.error(f"Backend unreachable at {API_BASE_URL}\n\n{health_info}")

    st.sidebar.markdown('<div class="section-eyebrow">Input mode</div>', unsafe_allow_html=True)
    input_mode = st.sidebar.radio(
        "Input mode",
        options=["Structured form", "Raw JSON"],
        label_visibility="collapsed",
        help="Use the form for normal testing. Use Raw JSON to hand-craft "
             "malformed payloads and verify the API's graceful-failure path.",
    )

    st.sidebar.markdown('<div class="section-eyebrow">Global settings</div>', unsafe_allow_html=True)
    with st.sidebar.container(border=True):
        st.caption(f"**Endpoint**  \n`{PREDICT_ENDPOINT}`")
        st.caption(f"**Request timeout**  \n{REQUEST_TIMEOUT_SECONDS}s")

    return input_mode


def render_structured_form() -> dict | None:
    """
    Individual widgets rather than one big free-text field, so a normal
    demo doesn't require hand-writing JSON — but note that NONE of these
    widgets clamp `amount` to a positive minimum. That's deliberate: if the
    widget itself blocked negative numbers, this UI could never actually
    demonstrate requirement #5 (proving a negative amount gets caught and
    gracefully handled by the API) — the bad input has to reach the
    backend for that proof to mean anything.
    """
    row1 = st.columns([1.3, 1.5, 1.2])
    amount = row1[0].number_input(
        "Amount ($)", value=249.99, step=1.0, format="%.2f",
        help="Try a negative value here to test the API's validation gate.",
    )
    merchant = row1[1].text_input("Merchant", value="fraud_Kutch_and_Sons")
    category = row1[2].selectbox(
        "Category",
        options=["shopping_net", "grocery_pos", "misc_net", "travel", "gas_transport", "entertainment"],
    )

    st.markdown('<div class="section-eyebrow">Merchant location</div>', unsafe_allow_html=True)
    row2 = st.columns(2)
    merchant_lat = row2[0].number_input("Merchant lat", value=40.7128, format="%.4f")
    merchant_long = row2[1].number_input("Merchant long", value=-74.0060, format="%.4f")

    st.markdown('<div class="section-eyebrow">Cardholder location</div>', unsafe_allow_html=True)
    row3 = st.columns(2)
    customer_lat = row3[0].number_input("Customer lat", value=34.0522, format="%.4f")
    customer_long = row3[1].number_input("Customer long", value=-118.2437, format="%.4f")

    st.markdown('<div class="section-eyebrow">Cardholder &amp; transaction timing</div>', unsafe_allow_html=True)
    row4 = st.columns(3)
    dob = row4[0].date_input(
        "Date of birth", value=date(1990, 5, 14),
        min_value=date(1900, 1, 1), max_value=date.today(),
    )
    now = datetime.now()
    # Defaulting to "right now" (rather than a fixed placeholder like
    # 14:30) matters here: the API's trans_time validator rejects
    # timestamps more than a few minutes in the future (see schemas.py),
    # so a fixed default time could accidentally trip that check on the
    # very first click depending on what time the demo happens to run at
    # — an unhelpful accidental failure on first use, unrelated to what
    # the demo is meant to show. Defaulting to "now" makes a first click
    # with unmodified inputs reliably land on the success path; a
    # deliberately future time is still one click away for testing #5.
    trans_date = row4[1].date_input("Date", value=now.date(), key="trans_date")
    trans_time_of_day = row4[2].time_input("Time", value=now.time(), key="trans_time")

    trans_datetime = datetime.combine(trans_date, trans_time_of_day)

    return {
        "amount": amount,
        "merchant": merchant,
        "category": category,
        "merchant_lat": merchant_lat,
        "merchant_long": merchant_long,
        "customer_lat": customer_lat,
        "customer_long": customer_long,
        "dob": dob.isoformat(),
        "trans_time": trans_datetime.isoformat(),
    }


def render_raw_json_editor() -> dict | None:
    """
    Free-text JSON editor for deliberately crafting malformed payloads
    (missing fields, wrong types, negative amounts) — the fastest way to
    demo the 422 graceful-failure path end-to-end.

    WHY `key=` IS NOT OPTIONAL HERE: without an explicit key, Streamlit
    derives a widget's identity partly from its `value=` argument. The
    example payload below includes `datetime.now().isoformat()`, which is
    a different string on every single script rerun — so an unkeyed
    text_area would look like a brand-new widget each time Streamlit
    reruns the script (which happens on ANY interaction, not just
    clicking submit), silently discarding whatever the user had typed and
    snapping back to a fresh example. Pinning `key="raw_json_editor"`
    makes Streamlit track this widget by that stable key instead, so
    `value=` only seeds it once and user edits persist across reruns.
    """
    example_payload = {
        "amount": 249.99,
        "merchant": "fraud_Kutch_and_Sons",
        "category": "shopping_net",
        "merchant_lat": 40.7128,
        "merchant_long": -74.0060,
        "customer_lat": 34.0522,
        "customer_long": -118.2437,
        "dob": "1990-05-14",
        "trans_time": datetime.now().isoformat(timespec="seconds"),
    }
    raw_text = st.text_area(
        "JSON payload", value=json.dumps(example_payload, indent=2), height=340,
        key="raw_json_editor",
        help="Edit freely — e.g. set amount to -50, or delete a field entirely — "
             "to test the API's validation and fallback behavior.",
    )
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # A JSON parse failure happens entirely client-side, before any
        # request is sent — this is NOT the same thing as the API's 422
        # response, so it gets its own distinct message rather than being
        # dressed up to look like a backend validation failure.
        st.error(f"Invalid JSON — fix before submitting: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def render_decision_banner(action: str) -> None:
    style = RISK_PALETTE.get(action, RISK_PALETTE["UNKNOWN"])
    st.markdown(
        f"""
        <div class="decision-banner" style="background-color:{style['bg']};
             border: 1px solid {style['border']}; color:{style['text']};">
            <div class="decision-emoji">{style['emoji']}</div>
            <div class="decision-copy">
                <div class="decision-eyebrow">Decision</div>
                <div class="decision-title">{style['label'].upper()}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_success_result(data: dict) -> None:
    action = data.get("action_taken", "UNKNOWN")
    render_decision_banner(action)

    if data.get("fallback_triggered"):
        st.warning("⚠️ **ML Pipeline Unavailable.** System automatically degraded to high-availability rules-based fallback.")
        st.info("Because the transaction was routed to the fallback engine, probability metrics and SHAP explainability are not available.")
    else:
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Fraud Probability", f"{data.get('fraud_probability', 0):.2%}")
            col2.metric("Decision Threshold", f"{data.get('threshold_used', 0):.3f}")
            col3.metric("Is Fraud (model verdict)", "Yes" if data.get("is_fraud") else "No")

        st.markdown('<div class="section-eyebrow">Top contributing features (SHAP)</div>', unsafe_allow_html=True)
        shap_explanation: dict = data.get("shap_explanation", {})
        if not shap_explanation:
            st.info("No SHAP explanation was returned for this prediction.")
        else:
            render_shap_chart(shap_explanation)
    with st.expander("Raw API response"):
        st.json(data)


def render_shap_chart(shap_explanation: dict) -> None:
    """
    Renders the top-3 SHAP contributions as both a diverging horizontal
    bar chart (immediate visual read: which way did each feature push the
    score) and a plain table (exact values, for anyone who wants the
    numbers rather than the picture).

    WHY A DIVERGING CHART SPECIFICALLY: a positive SHAP value pushed the
    prediction TOWARD fraud, a negative one pushed it AWAY from fraud —
    that sign is the single most important piece of information in this
    chart, so color (not just bar length) encodes it directly rather than
    making the analyst read the axis to figure out direction.

    The two colors below are pulled directly from RISK_PALETTE (BLOCK's
    red / APPROVE's green) so the chart's semantics agree with the
    decision banner's, rather than introducing a second unrelated color
    language on a dark background.
    """
    df = pd.DataFrame(
        [{"feature": k, "shap_value": v} for k, v in shap_explanation.items()]
    ).sort_values("shap_value", key=abs, ascending=False)
    df["direction"] = df["shap_value"].apply(
        lambda v: "Pushes toward FRAUD" if v > 0 else "Pushes toward LEGITIMATE"
    )

    grid_color = "rgba(255, 255, 255, 0.06)"
    axis_color = "#8A94A6"

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X(
                "shap_value:Q",
                title="SHAP contribution",
                axis=alt.Axis(labelColor=axis_color, titleColor=axis_color,
                               gridColor=grid_color, domainColor="#262B35"),
            ),
            y=alt.Y(
                "feature:N", sort="-x", title=None,
                axis=alt.Axis(labelColor="#F5F7FA", domainColor="#262B35", tickColor="#262B35"),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Pushes toward FRAUD", "Pushes toward LEGITIMATE"],
                    range=[RISK_PALETTE["BLOCK"]["text"], RISK_PALETTE["APPROVE"]["text"]],
                ),
                legend=alt.Legend(title=None, orient="bottom", labelColor=axis_color),
            ),
            tooltip=["feature", "shap_value", "direction"],
        )
        .properties(height=160, background="transparent")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")

    st.dataframe(
        df[["feature", "shap_value", "direction"]].rename(
            columns={"feature": "Feature", "shap_value": "SHAP Value", "direction": "Direction"}
        ),
        hide_index=True, width="stretch",
    )


def render_validation_failure(response: requests.Response) -> None:
    """
    Handles the API's 422 response — the exact scenario requirement #5
    asks us to prove doesn't crash the frontend. The response body's shape
    (see src/api/exception_handlers.py) always carries a `decision` object
    with a rules-based fallback, even though the model was never scored —
    we render that fallback exactly as prominently as a real prediction,
    so it's visually obvious the system degraded gracefully rather than
    silently failing.
    """
    try:
        body = response.json()
    except json.JSONDecodeError:
        st.error(f"Received HTTP {response.status_code} with a non-JSON body:\n\n{response.text}")
        return

    st.warning(
        "⚠️  **Validation failed — the API rejected this payload before scoring it.** "
        "The system did NOT crash: it returned a structured error and a safe fallback decision."
    )

    validation_errors = body.get("validation_errors", [])
    if validation_errors:
        st.markdown("**What failed:**")
        st.table(pd.DataFrame(validation_errors))

    decision = body.get("decision", {})
    if decision:
        st.markdown("**Fallback decision (rules-based, no model involved):**")
        col1, col2 = st.columns(2)
        col1.metric("Action Taken", decision.get("action_taken", "UNKNOWN"))
        col2.metric("Fallback Triggered", "Yes" if decision.get("fallback_triggered") else "No")
        if decision.get("reason"):
            st.caption(decision["reason"])

    with st.expander("Raw API response"):
        st.json(body)


def render_server_error(response: requests.Response) -> None:
    """
    Handles a 500 from the API — an internal scoring error rather than a
    bad payload (see generic_exception_handler in exception_handlers.py).
    Same principle as the 422 case: show the fallback decision the API
    still managed to return, rather than just showing a raw error.
    """
    try:
        body = response.json()
    except json.JSONDecodeError:
        st.error(f"Received HTTP {response.status_code} with a non-JSON body:\n\n{response.text}")
        return

    st.warning(
        "⚠️  **An internal error occurred while scoring this transaction.** "
        "The API did NOT crash: it returned a safe fallback decision instead."
    )
    decision = body.get("decision", {})
    if decision:
        st.metric("Fallback Action", decision.get("action_taken", "UNKNOWN"))
        if decision.get("reason"):
            st.caption(decision["reason"])

    with st.expander("Raw API response"):
        st.json(body)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    input_mode = render_sidebar()

    st.title("🛡️ AI Risk Manager")
    st.caption(
        "Submit a transaction payload below. The API scores it with "
        "LightGBM, applies a cost-optimized decision threshold, and returns "
        "a SHAP-explained decision — or, for malformed input, a safe "
        "rules-based fallback."
    )

    with st.container(border=True):
        st.markdown("#### Transaction details")
        if input_mode == "Structured form":
            payload = render_structured_form()
        else:
            payload = render_raw_json_editor()

        st.write("")
        _, btn_col = st.columns([3, 1])
        submitted = btn_col.button("🔍 Check Transaction", type="primary", width="stretch")

    if not submitted:
        st.info("Fill in a transaction above and click **Check Transaction**.")
        return

    if payload is None:
        # Raw JSON mode with a parse error — already surfaced above by
        # render_raw_json_editor(); nothing valid to send.
        st.error("Fix the JSON payload above before submitting.")
        return

    # ------------------------------------------------------------------ #
    # THE ACTUAL REQUEST. Every failure mode below is caught explicitly —
    # a merchant-facing risk console going down because the fraud API had
    # a bad day would be a worse outcome than the fraud API itself being
    # briefly unavailable, so this layer needs its own graceful handling,
    # independent of (and in addition to) the API's own.
    # ------------------------------------------------------------------ #
    with st.spinner("Scoring transaction..."):
        try:
            response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Could not connect to the API at `{API_BASE_URL}`. "
                "Is the FastAPI backend running (`uvicorn api.main:app`)?"
            )
            return
        except requests.exceptions.Timeout:
            st.error(f"❌ The API did not respond within {REQUEST_TIMEOUT_SECONDS}s.")
            return
        except requests.exceptions.RequestException as exc:
            st.error(f"❌ Request to the API failed: {exc}")
            return

    if response.status_code == 200:
        render_success_result(response.json())
    elif response.status_code == 422:
        render_validation_failure(response)
    elif response.status_code >= 500:
        render_server_error(response)
    else:
        st.error(f"Unexpected HTTP {response.status_code}:\n\n{response.text}")


if __name__ == "__main__":
    main()
"""
app.py — FourSight AI · Military Vehicle HUMS Dashboard
========================================================
Reads the CSV produced by hums_gen.cpp and renders a Streamlit web dashboard
with a prioritized maintenance work-order list and IBM watsonx.ai Granite
AI explanations.

CSV schema (from hums_gen.cpp):
  timestamp, vehicle_id, engine_temp_c, vibration_mm_s, run_hours, status

Run:
    streamlit run src/app.py -- --csv data/hums_data.csv

AI features (optional):
    Set the following environment variables (or copy src/.env.example → .env):
      WATSONX_API_KEY      — IBM Cloud API key
      WATSONX_PROJECT_ID   — watsonx.ai project ID
      WATSONX_URL          — e.g. https://us-south.ml.cloud.ibm.com
      WATSONX_MODEL_ID     — defaults to ibm/granite-3-3-8b-instruct

    When configured:
      • FAULT / CRITICAL vehicles receive AI readiness briefings on page load.
      • Every work-order row has an "🤖 Ask Granite" on-demand explain button.
      • A Fleet AI Summary section generates a commanding-officer briefing.

    When not configured the dashboard loads normally; AI panels show a
    configuration notice instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import watsonx_client

# ---------------------------------------------------------------------------
# Thresholds — kept in sync with hums_gen.cpp constants
# ---------------------------------------------------------------------------

WARN_TEMP_C     = 105.0   # °C
FAULT_TEMP_C    = 120.0   # °C
WARN_VIBRATION  =   8.0   # mm/s
FAULT_VIBRATION =  14.0   # mm/s

# Run-hours threshold that triggers a mandatory scheduled-maintenance flag
RUN_HOURS_SERVICE_INTERVAL = 500.0   # hours between services

# Scoring weights (tunable)
WEIGHT_STATUS    = 0.50   # heaviest — discrete fault state
WEIGHT_TEMP      = 0.25   # continuous temperature excess
WEIGHT_VIBRATION = 0.25   # continuous vibration excess

# Priority bands derived from composite score (0–100)
PRIORITY_CRITICAL  = 75
PRIORITY_HIGH      = 50
PRIORITY_MODERATE  = 25
# Load the predictive model
try:
   
    model = joblib.load('predictive_model.pkl')
    st.sidebar.success("AI Predictive Model Active ✅")
except FileNotFoundError:
    
    try:
        model = joblib.load('../predictive_model.pkl')
        st.sidebar.success("AI Predictive Model Active ✅")
    except:
        st.sidebar.error("Model file not found! Please check the path.")

# ---------------------------------------------------------------------------
# Section 1 — CLI argument parsing (Streamlit forwards args after '--')
# ---------------------------------------------------------------------------

def _parse_cli_csv() -> Optional[str]:
    """Return the --csv path if supplied on the command line, else None."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", default=None)
    try:
        args, _ = parser.parse_known_args(sys.argv[1:])
        return args.csv
    except SystemExit:
        return None


# ---------------------------------------------------------------------------
# Section 2 — Data loading & validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "timestamp", "vehicle_id", "engine_temp_c",
    "vibration_mm_s", "run_hours", "status",
}


@st.cache_data(show_spinner="Loading HUMS data…")
def load_csv(path: str) -> pd.DataFrame:
    """Load and lightly validate the HUMS CSV file."""
    df = pd.read_csv(path)

    # Map alternate column names (like hums_data2.csv) to standard names
    rename_map = {
        "Vehicle": "vehicle_id",
        "Status": "status",
        "Temp (C)": "engine_temp_c",
        "Vibration (mm/s)": "vibration_mm_s",
        "Run Hours": "run_hours",
        "Last Reading": "timestamp"
    }
    df = df.rename(columns=rename_map)

    required = {"vehicle_id", "engine_temp_c", "vibration_mm_s", "run_hours", "status"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.stop()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime("now")

    df["status"] = df["status"].str.upper().str.strip()
    df["engine_temp_c"]   = pd.to_numeric(df["engine_temp_c"],   errors="coerce")
    df["vibration_mm_s"]  = pd.to_numeric(df["vibration_mm_s"],  errors="coerce")
    df["run_hours"]        = pd.to_numeric(df["run_hours"],        errors="coerce")
    df = df.dropna(subset=["engine_temp_c", "vibration_mm_s", "run_hours"])

    return df



# ---------------------------------------------------------------------------
# Section 3 — Per-vehicle aggregation & composite risk scoring
# ---------------------------------------------------------------------------

STATUS_SCORE = {"NOMINAL": 0, "WARNING": 50, "FAULT": 100}


def _latest_status_score(statuses: pd.Series) -> float:
    last = statuses.iloc[-1] if not statuses.empty else "NOMINAL"
    return float(STATUS_SCORE.get(last, 0))


def _temp_excess_score(temps: pd.Series) -> float:
    latest = float(temps.iloc[-1])
    if latest < WARN_TEMP_C:
        return 0.0
    return min(100.0, (latest - WARN_TEMP_C) / (FAULT_TEMP_C - WARN_TEMP_C) * 100.0)


def _vib_excess_score(vibs: pd.Series) -> float:
    latest = float(vibs.iloc[-1])
    if latest < WARN_VIBRATION:
        return 0.0
    return min(100.0, (latest - WARN_VIBRATION) / (FAULT_VIBRATION - WARN_VIBRATION) * 100.0)


def compute_work_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw HUMS rows into one work-order row per vehicle, sorted by
    descending composite risk score.
    """
    df_sorted = df.sort_values("timestamp")
    groups = df_sorted.groupby("vehicle_id", sort=False)

    records = []
    for vid, grp in groups:
        latest = grp.iloc[-1]

        status_score = _latest_status_score(grp["status"])
        temp_score   = _temp_excess_score(grp["engine_temp_c"])
        vib_score    = _vib_excess_score(grp["vibration_mm_s"])

        composite = (
            WEIGHT_STATUS    * status_score
            + WEIGHT_TEMP    * temp_score
            + WEIGHT_VIBRATION * vib_score
        )

        total_hours = float(latest["run_hours"])
        service_due = (total_hours % RUN_HOURS_SERVICE_INTERVAL) < 10.0 or \
                      total_hours >= RUN_HOURS_SERVICE_INTERVAL

        fault_count   = int((grp["status"] == "FAULT").sum())
        warning_count = int((grp["status"] == "WARNING").sum())

        records.append({
            "vehicle_id":            vid,
            "latest_timestamp":      latest["timestamp"],
            "latest_status":         latest["status"],
            "latest_temp_c":         round(float(latest["engine_temp_c"]),  2),
            "latest_vibration_mm_s": round(float(latest["vibration_mm_s"]), 2),
            "total_run_hours":       round(total_hours, 1),
            "fault_count":           fault_count,
            "warning_count":         warning_count,
            "service_due":           service_due,
            "composite_score":       round(composite, 1),
        })

    wo = pd.DataFrame(records).sort_values("composite_score", ascending=False)
    wo = wo.reset_index(drop=True)
    wo.index += 1  # 1-based work-order rank

    def _band(score: float) -> str:
        if score >= PRIORITY_CRITICAL:  return "🔴 CRITICAL"
        if score >= PRIORITY_HIGH:      return "🟠 HIGH"
        if score >= PRIORITY_MODERATE:  return "🟡 MODERATE"
        return "🟢 LOW"

    wo["priority"] = wo["composite_score"].apply(_band)
    return wo


# ---------------------------------------------------------------------------
# Section 4 — Streamlit UI helpers (colours, KPI row, charts)
# ---------------------------------------------------------------------------

STATUS_COLOURS = {
    "FAULT":   "#d62728",
    "WARNING": "#ff7f0e",
    "NOMINAL": "#2ca02c",
}

PRIORITY_COLOURS = {
    "🔴 CRITICAL": "#d62728",
    "🟠 HIGH":     "#ff7f0e",
    "🟡 MODERATE": "#f0c040",
    "🟢 LOW":      "#2ca02c",
}


def _status_colour(val: str) -> str:
    return f"color: {STATUS_COLOURS.get(val, '#888')}; font-weight: bold"


def _priority_colour(val: str) -> str:
    return f"color: {PRIORITY_COLOURS.get(val, '#888')}; font-weight: bold"


def render_kpi_row(df: pd.DataFrame, wo: pd.DataFrame) -> None:
    total    = wo.shape[0]
    critical = (wo["priority"] == "🔴 CRITICAL").sum()
    high     = (wo["priority"] == "🟠 HIGH").sum()
    faults   = (df["status"] == "FAULT").sum()
    warnings = (df["status"] == "WARNING").sum()
    svc_due  = wo["service_due"].sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Vehicles",         total)
    c2.metric("🔴 Critical",      int(critical))
    c3.metric("🟠 High",          int(high))
    c4.metric("FAULT readings",   int(faults))
    c5.metric("WARNING readings", int(warnings))
    c6.metric("Service Due",      int(svc_due))


def render_charts(df: pd.DataFrame, wo: pd.DataFrame) -> None:
    st.subheader("Fleet Overview")

    col_a, col_b = st.columns(2)

    with col_a:
        status_counts = df["status"].value_counts().reindex(
            ["FAULT", "WARNING", "NOMINAL"], fill_value=0
        )
        status_df = status_counts.reset_index()
        status_df.columns = ["Status", "Count"]
        st.bar_chart(status_df.set_index("Status"), color="#3b82d4")
        st.caption("All-time readings by status")

    with col_b:
        priority_counts = wo["priority"].value_counts().reset_index()
        priority_counts.columns = ["Priority", "Count"]
        st.bar_chart(priority_counts.set_index("Priority"), color="#7c5cd8")
        st.caption("Vehicles by priority band")

    st.subheader("Engine Temperature Trend — Top 5 Risk Vehicles")
    top5_ids = wo.head(5)["vehicle_id"].tolist()
    trend_df = (
        df[df["vehicle_id"].isin(top5_ids)]
        .sort_values("timestamp")
        .pivot_table(index="timestamp", columns="vehicle_id",
                     values="engine_temp_c", aggfunc="mean")
    )
    st.line_chart(trend_df)

    st.subheader("Vibration Trend — Top 5 Risk Vehicles")
    vib_df = (
        df[df["vehicle_id"].isin(top5_ids)]
        .sort_values("timestamp")
        .pivot_table(index="timestamp", columns="vehicle_id",
                     values="vibration_mm_s", aggfunc="mean")
    )
    st.line_chart(vib_df)


# ---------------------------------------------------------------------------
# Section 5 — watsonx.ai / Granite integration
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Ensure required session-state keys are present."""
    if "granite_explanations" not in st.session_state:
        st.session_state["granite_explanations"] = {}
    if "granite_fleet_summary" not in st.session_state:
        st.session_state["granite_fleet_summary"] = None


def generate_auto_explanations(wo: pd.DataFrame) -> None:
    """
    Pre-generate Granite explanations for every FAULT / CRITICAL vehicle on
    page load.  Results are cached in st.session_state so the API is called at
    most once per vehicle per browser session.

    Runs silently when watsonx.ai is not configured.
    """
    if not watsonx_client.is_configured():
        return

    # Identify vehicles that need auto-explanation and are not already cached
    auto_mask = (wo["latest_status"] == "FAULT") | (wo["priority"] == "🔴 CRITICAL")
    targets = wo[auto_mask]
    pending = [
        row for _, row in targets.iterrows()
        if row["vehicle_id"] not in st.session_state["granite_explanations"]
    ]

    if not pending:
        return

    with st.spinner(f"Generating AI readiness briefings for {len(pending)} vehicle(s)…"):
        for row in pending:
            vid = row["vehicle_id"]
            try:
                explanation = watsonx_client.explain_vehicle(row.to_dict())
                st.session_state["granite_explanations"][vid] = explanation
            except watsonx_client.WatsonxError as exc:
                st.session_state["granite_explanations"][vid] = (
                    f"⚠️ AI explanation unavailable: {exc}"
                )


def render_fleet_summary(wo: pd.DataFrame) -> None:
    """
    Render the collapsible Fleet AI Summary expander below the KPI strip.
    Shows a Granite-generated commanding-officer readiness brief.
    """
    with st.expander("🤖 Fleet AI Summary", expanded=False):
        if not watsonx_client.is_configured():
            st.caption(
                "AI fleet summary is disabled. "
                "Set WATSONX_API_KEY, WATSONX_PROJECT_ID, and WATSONX_URL "
                "in your environment or .env file to enable it."
            )
            return

        cached = st.session_state.get("granite_fleet_summary")
        if cached:
            st.info(cached)
            if st.button("🔄 Regenerate Fleet Summary", key="regen_fleet"):
                st.session_state["granite_fleet_summary"] = None
                st.rerun()
        else:
            st.caption(
                "Generate a plain-English fleet readiness brief for the current "
                "work-order data, powered by IBM Granite."
            )
            if st.button("📋 Generate Fleet Summary", key="gen_fleet"):
                with st.spinner("Asking Granite for a fleet readiness brief…"):
                    try:
                        summary = watsonx_client.explain_fleet(wo)
                        st.session_state["granite_fleet_summary"] = summary
                        st.rerun()
                    except watsonx_client.WatsonxError as exc:
                        st.error(f"Fleet summary failed: {exc}")


def render_work_order_expanders(wo: pd.DataFrame) -> None:
    """
    Render each vehicle as a collapsible expander.

    Inside each expander:
      • Two rows of st.metric show all key sensor / score fields.
      • A Granite AI explanation is shown if already cached, or an
        "🤖 Ask Granite" button fetches it on demand.
    """
    for _, row in wo.iterrows():
        vid      = row["vehicle_id"]
        priority = row["priority"]
        status   = row["latest_status"]
        score    = row["composite_score"]

        # Colour the expander label by priority
        label_colour = PRIORITY_COLOURS.get(priority, "#888")
        label = f"{vid}  —  {priority}  |  Risk Score: {score:.1f}"

        # Auto-expand FAULT vehicles so critical items are immediately visible
        auto_expand = (status == "FAULT") or (priority == "🔴 CRITICAL")

        with st.expander(label, expanded=auto_expand):
            # --- Metric grid ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status",         status)
            c2.metric("Temp (°C)",      f"{row['latest_temp_c']:.1f}")
            c3.metric("Vibration",      f"{row['latest_vibration_mm_s']:.2f} mm/s")
            c4.metric("Run Hours",      f"{row['total_run_hours']:.1f} h")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Risk Score",     f"{score:.1f} / 100")
            c6.metric("# Faults",       row["fault_count"])
            c7.metric("# Warnings",     row["warning_count"])
            c8.metric("Svc Due",        "✅ Yes" if row["service_due"] else "No")

            st.caption(f"Last reading: {row['latest_timestamp']}")
            st.divider()

            # --- Granite AI explanation ---
            cached_explanation = st.session_state["granite_explanations"].get(vid)

            if cached_explanation:
                st.markdown("**🤖 Granite AI Assessment**")
                st.info(cached_explanation)
            elif watsonx_client.is_configured():
                if st.button("🤖 Ask Granite", key=f"explain_{vid}"):
                    with st.spinner(f"Generating AI assessment for {vid}…"):
                        try:
                            explanation = watsonx_client.explain_vehicle(row.to_dict())
                            st.session_state["granite_explanations"][vid] = explanation
                            st.rerun()
                        except watsonx_client.WatsonxError as exc:
                            st.error(f"AI explanation failed: {exc}")
            else:
                st.caption(
                    "🔒 Configure WATSONX_API_KEY, WATSONX_PROJECT_ID, and "
                    "WATSONX_URL to enable AI explanations."
                )


# ---------------------------------------------------------------------------
# Section 6 — Page layout & main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="FourSight AI — HUMS Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()

    # ---- Sidebar ---------------------------------------------------------------
    with st.sidebar:
        st.title("🛡️ FourSight AI")
        st.caption("Military Vehicle Health & Usage Monitoring")
        st.divider()

        st.subheader("Data Source")
        cli_csv = _parse_cli_csv()
        if cli_csv:
            st.info(f"Using CLI path:\n`{cli_csv}`")
            csv_path: Optional[str] = cli_csv
        else:
            typed_path = st.text_input(
                "CSV file path", value="data/hums_data2.csv",
                help="Path to the CSV generated by hums_gen.cpp"
            )
            uploaded = st.file_uploader(
                "…or upload CSV", type=["csv"],
                help="Upload a CSV file directly"
            )
            csv_path = None
            if uploaded is not None:
                tmp = Path("/tmp/hums_upload.csv")
                tmp.write_bytes(uploaded.read())
                csv_path = str(tmp)
            elif typed_path:
                csv_path = typed_path

        st.divider()

        st.subheader("Filters")
        min_score = st.slider(
            "Minimum risk score", min_value=0, max_value=100, value=0, step=5
        )
        show_svc_only = st.checkbox("Show service-due vehicles only", value=False)

        st.divider()
        st.caption("Sensor Thresholds")
        st.caption(f"Warn temp: {WARN_TEMP_C} °C | Fault temp: {FAULT_TEMP_C} °C")
        st.caption(f"Warn vib: {WARN_VIBRATION} mm/s | Fault vib: {FAULT_VIBRATION} mm/s")
        st.caption(f"Service interval: {RUN_HOURS_SERVICE_INTERVAL} h")

        st.divider()
        # Granite status indicator
        if watsonx_client.is_configured():
            st.success("🤖 Granite AI: connected")
        else:
            st.warning("🤖 Granite AI: not configured")

    # ---- Main content ----------------------------------------------------------
    st.title("🛡️ Military Vehicle HUMS — Maintenance Work-Order Dashboard")

    if not csv_path:
        st.info("Provide a CSV path in the sidebar or pass `--csv <path>` on the command line.")
        st.stop()

    if not Path(csv_path).exists():
        st.error(f"File not found: `{csv_path}`")
        st.stop()

    df = load_csv(csv_path)

    if df.empty:
        st.warning("The CSV file is empty or contained no parseable rows.")
        st.stop()

    wo = compute_work_orders(df)

    # Auto-generate Granite explanations for FAULT/CRITICAL vehicles
    generate_auto_explanations(wo)

    # Apply sidebar filters
    wo_filtered = wo[wo["composite_score"] >= min_score]
    if show_svc_only:
        wo_filtered = wo_filtered[wo_filtered["service_due"]]

    # ---- KPI row ---------------------------------------------------------------
    st.subheader("Fleet Summary")
    render_kpi_row(df, wo)

    # ---- Fleet AI Summary (Granite) --------------------------------------------
    render_fleet_summary(wo)

    st.divider()

    # ---- Work-order expanders --------------------------------------------------
    st.subheader(
        f"Prioritized Maintenance Work Orders"
        f"  — {len(wo_filtered)} vehicle(s)"
        + (" (filtered)" if len(wo_filtered) < len(wo) else "")
    )
    st.caption(
        "Sorted by **Risk Score** (0–100). "
        "Score = 50 % status + 25 % temperature excess + 25 % vibration excess. "
        "FAULT / CRITICAL vehicles auto-expand."
    )

    if wo_filtered.empty:
        st.info("No vehicles match the current filter settings.")
    else:
        render_work_order_expanders(wo_filtered)

        csv_bytes = wo_filtered.to_csv(index_label="rank").encode()
        st.download_button(
            label="⬇ Export work orders as CSV",
            data=csv_bytes,
            file_name="work_orders.csv",
            mime="text/csv",
        )

    st.divider()

    # ---- Charts ----------------------------------------------------------------
    render_charts(df, wo)

    # ---- Raw data explorer -----------------------------------------------------
    with st.expander("🔍 Raw HUMS data explorer"):
        vehicles = sorted(df["vehicle_id"].unique().tolist())
        selected = st.multiselect("Filter by vehicle", options=vehicles, default=[])
        raw_view = df[df["vehicle_id"].isin(selected)] if selected else df
        st.dataframe(raw_view.sort_values("timestamp", ascending=False),
                     use_container_width=True, height=300)

    st.caption(
        "FourSight AI · HUMS Dashboard · "
        f"Loaded {len(df):,} readings across {df['vehicle_id'].nunique()} vehicles "
        f"from `{Path(csv_path).name}`"
    )


if __name__ == "__main__":
    main()

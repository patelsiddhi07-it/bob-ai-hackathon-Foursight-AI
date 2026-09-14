"""
watsonx_client.py — FourSight AI · IBM watsonx.ai Granite integration
======================================================================
Provides a thin wrapper around the ibm_watsonx_ai SDK so that app.py never
imports the SDK directly.  All SDK imports are lazy (inside functions) so
the module loads cleanly even when ibm_watsonx_ai is not installed or
credentials are absent — the dashboard degrades gracefully in that case.

Public API
----------
is_configured() -> bool
    Returns True when all required WATSONX_* env vars are present.

explain_vehicle(row: dict, extra_context: str = "") -> str
    Build a technician-scoped prompt from a work-order row dict and call
    Granite.  Returns the generated explanation string.

explain_fleet(wo: pd.DataFrame) -> str
    Build a fleet-level prompt from the full work-order DataFrame and call
    Granite.  Returns the generated fleet readiness brief.

build_vehicle_prompt(row: dict) -> str
build_fleet_prompt(wo: pd.DataFrame) -> str
    Prompt builders exposed for testing / inspection.

Exceptions
----------
WatsonxError
    Raised (wrapping the underlying exception) when an SDK call fails.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

# Load .env if present — must happen before any os.environ reads.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv optional at import time; app still works from env

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID = "ibm/granite-3-3-8b-instruct"
_REQUIRED_ENV_VARS = ("WATSONX_API_KEY", "WATSONX_PROJECT_ID", "WATSONX_URL")

# Thresholds mirrored from app.py / hums_gen.cpp (for prompt context)
_WARN_TEMP_C      = 105.0
_FAULT_TEMP_C     = 120.0
_WARN_VIBRATION   =   8.0
_FAULT_VIBRATION  =  14.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

class WatsonxError(RuntimeError):
    """Raised when an IBM watsonx.ai API call fails."""


def is_configured() -> bool:
    """Return True when all required WATSONX_* env vars are non-empty."""
    return all(os.environ.get(v, "").strip() for v in _REQUIRED_ENV_VARS)


# ---------------------------------------------------------------------------
# Internal: SDK initialisation
# ---------------------------------------------------------------------------

# Module-level cache so the ModelInference object is constructed once per
# Python process (Streamlit reruns the script per interaction, but module
# globals persist across reruns within the same worker process).
_model_cache: dict = {}


def _build_model(max_new_tokens: int = 400):
    """
    Construct (or return a cached) ModelInference instance.

    Raises WatsonxError if the SDK is not installed or credentials are wrong.
    """
    cache_key = max_new_tokens
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        from ibm_watsonx_ai.foundation_models import ModelInference          # noqa: PLC0415
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # noqa: PLC0415
    except ImportError as exc:
        raise WatsonxError(
            "ibm-watsonx-ai package is not installed. "
            "Run: pip install ibm-watsonx-ai"
        ) from exc

    api_key    = os.environ["WATSONX_API_KEY"].strip()
    project_id = os.environ["WATSONX_PROJECT_ID"].strip()
    url        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip()
    model_id   = os.environ.get("WATSONX_MODEL_ID", _DEFAULT_MODEL_ID).strip()

    try:
        model = ModelInference(
            model_id=model_id,
            credentials={"apikey": api_key, "url": url},
            project_id=project_id,
            params={GenParams.MAX_NEW_TOKENS: max_new_tokens},
        )
    except Exception as exc:
        raise WatsonxError(f"Failed to initialise ModelInference: {exc}") from exc

    _model_cache[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_vehicle_prompt(row: dict, extra_context: str = "") -> str:
    """
    Build a military-maintenance-technician-scoped prompt for a single vehicle.

    Parameters
    ----------
    row : dict
        One row from the work-order DataFrame as a plain dict.
    extra_context : str
        Any free-text the operator wants to inject (e.g. recent maintenance
        notes).  Appended after the sensor table if non-empty.
    """
    svc_label = "YES — overdue" if row.get("service_due") else "No"

    prompt = (
        "You are a military vehicle maintenance advisor. "
        "Given the following Health and Usage Monitoring System (HUMS) sensor "
        "data, explain in plain English what is wrong with this vehicle, why it "
        "is a maintenance concern, and what the technician should inspect or "
        "replace first. Be concise (3–5 sentences). Do not repeat the raw "
        "numbers verbatim; interpret them.\n\n"
        f"Vehicle ID:          {row.get('vehicle_id', 'N/A')}\n"
        f"Priority Band:       {row.get('priority', 'N/A')}\n"
        f"Latest Status:       {row.get('latest_status', 'N/A')}\n"
        f"Engine Temp (°C):    {row.get('latest_temp_c', 'N/A')}"
        f"  [Warning >{_WARN_TEMP_C} °C / Fault >{_FAULT_TEMP_C} °C]\n"
        f"Vibration (mm/s):    {row.get('latest_vibration_mm_s', 'N/A')}"
        f"  [Warning >{_WARN_VIBRATION} / Fault >{_FAULT_VIBRATION}]\n"
        f"Total Run Hours:     {row.get('total_run_hours', 'N/A')}\n"
        f"FAULT readings:      {row.get('fault_count', 0)}\n"
        f"WARNING readings:    {row.get('warning_count', 0)}\n"
        f"Scheduled Svc Due:   {svc_label}\n"
        f"Composite Risk Score:{row.get('composite_score', 0):.1f} / 100\n"
    )

    if extra_context.strip():
        prompt += f"\nAdditional context from operator:\n{extra_context.strip()}\n"

    prompt += "\nMaintenance Assessment:\n"
    return prompt


def build_fleet_prompt(wo) -> str:  # wo: pd.DataFrame
    """
    Build a fleet-level readiness brief prompt from the full work-order frame.

    Lists counts per priority band, total fleet run hours, and the top-5
    highest-risk vehicles with their key metrics.
    """
    import pandas as pd  # local import — module may load before pandas is available

    total      = len(wo)
    critical   = int((wo["priority"] == "🔴 CRITICAL").sum())
    high       = int((wo["priority"] == "🟠 HIGH").sum())
    moderate   = int((wo["priority"] == "🟡 MODERATE").sum())
    low        = int((wo["priority"] == "🟢 LOW").sum())
    total_rh   = float(wo["total_run_hours"].sum())
    svc_due    = int(wo["service_due"].sum())

    top5 = wo.head(5)
    top5_lines = []
    for _, r in top5.iterrows():
        top5_lines.append(
            f"  • {r['vehicle_id']}: {r['priority']} | "
            f"Temp {r['latest_temp_c']:.1f}°C | "
            f"Vib {r['latest_vibration_mm_s']:.2f} mm/s | "
            f"Score {r['composite_score']:.1f}/100"
        )
    top5_block = "\n".join(top5_lines) if top5_lines else "  (no vehicles)"

    prompt = (
        "You are a military fleet readiness officer. "
        "Write a concise 5-sentence fleet health briefing for a commanding officer "
        "based on the HUMS data summary below. Cover: overall readiness status, "
        "the most critical vehicles, dominant failure modes, immediate action "
        "recommendations, and any scheduled maintenance backlog.\n\n"
        f"Fleet Size:           {total} vehicles\n"
        f"🔴 CRITICAL:          {critical}\n"
        f"🟠 HIGH:              {high}\n"
        f"🟡 MODERATE:          {moderate}\n"
        f"🟢 LOW:               {low}\n"
        f"Service Overdue:      {svc_due}\n"
        f"Total Fleet Run Hrs:  {total_rh:,.1f} h\n\n"
        f"Top 5 Highest-Risk Vehicles:\n{top5_block}\n\n"
        "Fleet Readiness Briefing:\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# Public inference calls
# ---------------------------------------------------------------------------

def explain_vehicle(row: dict, extra_context: str = "") -> str:
    """
    Generate a plain-English maintenance explanation for a single vehicle.

    Parameters
    ----------
    row : dict
        Work-order row dict (all keys from compute_work_orders output).
    extra_context : str
        Optional free-text injected after the sensor table.

    Returns
    -------
    str
        Granite-generated explanation, stripped of leading/trailing whitespace.

    Raises
    ------
    WatsonxError
        If the SDK is unavailable or the API call fails.
    """
    model  = _build_model(max_new_tokens=400)
    prompt = build_vehicle_prompt(row, extra_context)
    try:
        result = model.generate_text(prompt=prompt)
    except Exception as exc:
        raise WatsonxError(f"generate_text failed: {exc}") from exc
    return result.strip() if isinstance(result, str) else str(result).strip()


def explain_fleet(wo) -> str:  # wo: pd.DataFrame
    """
    Generate a fleet-level readiness brief paragraph.

    Parameters
    ----------
    wo : pd.DataFrame
        Full work-order DataFrame from compute_work_orders().

    Returns
    -------
    str
        Granite-generated fleet briefing, stripped of leading/trailing
        whitespace.

    Raises
    ------
    WatsonxError
        If the SDK is unavailable or the API call fails.
    """
    model  = _build_model(max_new_tokens=300)
    prompt = build_fleet_prompt(wo)
    try:
        result = model.generate_text(prompt=prompt)
    except Exception as exc:
        raise WatsonxError(f"generate_text failed for fleet summary: {exc}") from exc
    return result.strip() if isinstance(result, str) else str(result).strip()

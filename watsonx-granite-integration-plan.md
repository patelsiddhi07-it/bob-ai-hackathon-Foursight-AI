# Plan: watsonx.ai Granite Integration — Vehicle Readiness Explainer

## Overview

Add an IBM watsonx.ai Granite LLM layer to `src/app.py` that explains vehicle
readiness issues in plain language for maintenance technicians.

Two entry points:
1. **Auto-explain on load** — for every vehicle whose latest status is FAULT
   or whose priority band is 🔴 CRITICAL, a Granite explanation is
   pre-generated and cached when the page loads.
2. **On-demand "Explain" button** — every row in the work-order table has an
   expander containing an "Ask Granite" button; clicking it fetches an
   explanation for that specific vehicle on demand.

A **Fleet AI Summary** section at the top of the main content area shows a
single Granite-generated paragraph describing overall fleet readiness,
drawing on the aggregated work-order data.

Credentials (`WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`,
`WATSONX_MODEL_ID`) are loaded from the environment / `.env` file.
If credentials are absent the Granite features degrade gracefully — the
dashboard still loads and the AI panels show a configuration notice instead.

---

## Sub-Task 1 — Granite client module (`src/watsonx_client.py`)

**Status:** [x] done

### Intent
Isolate all watsonx.ai SDK calls in a single module so `app.py` never imports
`ibm_watsonx_ai` directly.  This keeps the dashboard functional when the
package is missing or credentials are not set, and makes the AI layer easy to
swap or test independently.

### Expected Outcomes
- A new file `src/watsonx_client.py` exists.
- It exports two public functions:
  - `is_configured() -> bool` — returns True only when all three required env
    vars are present.
  - `explain_vehicle(vehicle_row: dict, extra_context: str = "") -> str` —
    builds a prompt, calls `ModelInference.generate_text()`, returns the
    generated string. Raises `WatsonxError` on SDK failure so the caller can
    surface a user-friendly message.
- `ibm_watsonx_ai` is imported inside the functions (lazy import), so the
  module loads without error even if the package is not installed.
- Model ID defaults to `ibm/granite-3-3-8b-instruct` but is overridden by
  `WATSONX_MODEL_ID` env var.

### Todo List
- [ ] Create `src/watsonx_client.py`.
- [ ] Read `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`,
      `WATSONX_MODEL_ID` from `os.environ` (with `python-dotenv` load at
      module level).
- [ ] Implement `is_configured()`.
- [ ] Implement `_build_model()` — constructs and returns a cached
      `ModelInference` instance using the SDK pattern:
      `ModelInference(model_id=..., credentials={"apikey":..., "url":...},
      project_id=..., params={GenParams.MAX_NEW_TOKENS: 400})`.
- [ ] Implement `build_vehicle_prompt(row: dict) -> str` — formats the
      structured vehicle data into a role-scoped prompt (see Prompt Design
      section below).
- [ ] Implement `explain_vehicle(row, extra_context) -> str` — calls
      `model.generate_text(prompt)` and returns the stripped result.
- [ ] Add `WatsonxError` exception class.
- [ ] Add `src/watsonx_client.py` to `src/requirements.txt` entry for
      `ibm-watsonx-ai>=1.1.0`.

### Prompt Design
The prompt sent to Granite must be written for a military maintenance
technician persona and must include all computed fields from the work-order row:

```
You are a military vehicle maintenance advisor. Given the following HUMS
sensor data for vehicle {vehicle_id}, explain in plain English what is wrong,
why it is a maintenance concern, and what the technician should inspect or
replace first. Be concise (3-5 sentences).

Vehicle ID:        {vehicle_id}
Priority:          {priority}
Latest Status:     {latest_status}
Engine Temp (°C):  {latest_temp_c}  [Warning: 105°C / Fault: 120°C]
Vibration (mm/s):  {latest_vibration_mm_s}  [Warning: 8 / Fault: 14]
Run Hours:         {total_run_hours}
FAULT readings:    {fault_count}
WARNING readings:  {warning_count}
Service Due:       {service_due}
Risk Score:        {composite_score}/100
```

### Relevant Context
- IBM SDK pattern confirmed from docs:
  `ModelInference(model_id=..., credentials={"apikey": ..., "url": ...},
  project_id=..., params={GenParams.MAX_NEW_TOKENS: 400})`
- `generate_text(prompt)` returns a plain string.
- `GenParams` is imported from
  `ibm_watsonx_ai.metanames.GenTextParamsMetaNames`.

---

## Sub-Task 2 — Auto-explain for FAULT/CRITICAL vehicles on load

**Status:** [x] done

### Intent
When the dashboard loads, silently call `explain_vehicle` for every vehicle
whose `latest_status == "FAULT"` or `priority == "🔴 CRITICAL"` and store the
results in `st.session_state`.  Subsequent renders read from session state, so
the API is called at most once per vehicle per browser session.

### Expected Outcomes
- A `generate_auto_explanations(wo: pd.DataFrame) -> None` function exists in
  `app.py`.
- It runs only if `is_configured()` is True.
- Results are stored in `st.session_state["granite_explanations"]` as a
  `dict[vehicle_id -> str]`.
- A `st.spinner` is shown while the batch runs; if any call fails the error is
  caught, logged to `st.warning`, and the loop continues for other vehicles.
- Function is called from `main()` immediately after `compute_work_orders()`.

### Todo List
- [ ] Add `generate_auto_explanations(wo)` to `app.py`.
- [ ] Initialise `st.session_state["granite_explanations"]` to `{}` if absent.
- [ ] Loop over rows where `latest_status == "FAULT"` or
      `priority == "🔴 CRITICAL"`.
- [ ] Skip vehicles already present in session state (already explained).
- [ ] Call `watsonx_client.explain_vehicle(row.to_dict())` inside a try/except.
- [ ] Store result or error string in session state.
- [ ] Wrap the loop in `st.spinner("Generating AI readiness briefings…")`.
- [ ] Call `generate_auto_explanations(wo)` in `main()` before rendering the
      work-order table.

### Relevant Context
- `wo` DataFrame has one row per vehicle, with all the fields listed in the
  prompt design above.
- `st.session_state` persists across rerenders within the same browser
  session; `@st.cache_data` is for pure data, not side-effectful API calls.

---

## Sub-Task 3 — Per-row "Explain" expander in the work-order table

**Status:** [x] done

### Intent
Replace the current `st.dataframe(styled, ...)` call with a row-by-row
rendering loop that places each vehicle in an `st.expander`. Inside the
expander: the styled row data is shown, and a button labelled
"🤖 Ask Granite" triggers an on-demand explanation if one is not already
cached.

### Expected Outcomes
- Each vehicle in `wo_filtered` is rendered as an `st.expander` with the
  vehicle ID and priority band as its label.
- Inside the expander, the vehicle's metrics are displayed with
  `st.columns` / `st.metric`.
- An "🤖 Ask Granite" button is present.
  - If `is_configured()` is False, the button is replaced by a grey notice:
    *"Configure WATSONX_* env vars to enable AI explanations."*
  - If a cached explanation exists in session state it is displayed directly
    (no button needed).
  - Clicking the button calls `explain_vehicle` and writes to session state.
- The fleet-level CSV export download button moves to below the expanders.

### Todo List
- [ ] Add `render_work_order_expanders(wo: pd.DataFrame) -> None` to `app.py`.
- [ ] Iterate `wo_filtered.iterrows()`.
- [ ] For each row create `st.expander(f"{row.vehicle_id}  —  {row.priority}")`.
- [ ] Inside the expander, display metrics in two `st.columns` rows:
      Status, Temp, Vibration, Run Hours, Risk Score, Fault count,
      Warning count, Service Due.
- [ ] Check `st.session_state["granite_explanations"].get(vid)`:
  - Present → `st.info(explanation)`.
  - Absent + configured → show "🤖 Ask Granite" button; on click call
    `explain_vehicle`, store result, `st.rerun()`.
  - Absent + not configured → `st.caption("…configure env vars…")`.
- [ ] Replace the `render_work_order_table(wo_filtered)` call in `main()` with
      `render_work_order_expanders(wo_filtered)`.

### Relevant Context
- `st.rerun()` is the Streamlit 1.27+ replacement for `st.experimental_rerun`.
- Per-button state must use unique keys: `f"explain_{vid}"`.

---

## Sub-Task 4 — Fleet AI Summary section

**Status:** [x] done

### Intent
Add a collapsible "🤖 Fleet AI Summary" section just below the KPI strip that
uses Granite to generate a single-paragraph fleet health briefing from the
aggregated work-order data.  The summary covers: total vehicles, how many are
critical/high/nominal, which vehicles are highest risk, and what the top
recommended actions are.

### Expected Outcomes
- A "🤖 Fleet AI Summary" expander appears after the KPI strip.
- Inside: a "Generate Fleet Summary" button.
- Clicking generates a summary prompt that describes the fleet state
  numerically and asks Granite to write a 5-sentence readiness brief.
- The result is cached in `st.session_state["granite_fleet_summary"]`.
- If not configured, the expander shows a configuration notice.

### Todo List
- [ ] Add `build_fleet_prompt(wo: pd.DataFrame) -> str` to
      `watsonx_client.py`.  It receives the full work-order frame and
      produces a prompt listing top-5 vehicles by risk score, counts per
      priority band, and total fleet run hours.
- [ ] Add `explain_fleet(wo: pd.DataFrame) -> str` to `watsonx_client.py`
      that calls `generate_text` with the fleet prompt.
- [ ] Add `render_fleet_summary(wo: pd.DataFrame) -> None` to `app.py`.
- [ ] Call `render_fleet_summary(wo)` in `main()` between the KPI row and
      the divider before the work-order table.

### Relevant Context
- Same `ModelInference` instance reused from Sub-Task 1.
- Fleet prompt should request `MAX_NEW_TOKENS: 300`.

---

## Sub-Task 5 — Configuration & dependency updates

**Status:** [x] done

### Intent
Ensure the project can be installed and run by a new contributor with just
`pip install -r src/requirements.txt` and a `.env` file.

### Expected Outcomes
- `src/requirements.txt` includes `ibm-watsonx-ai>=1.1.0` and
  `python-dotenv>=1.0.0`.
- `src/.env.example` documents `WATSONX_MODEL_ID` alongside the existing
  `WATSONX_*` vars.
- The app module docstring is updated to mention the new `--csv` run command
  and the env vars required for AI features.

### Todo List
- [ ] Add `ibm-watsonx-ai>=1.1.0` and `python-dotenv>=1.0.0` to
      `src/requirements.txt`.
- [ ] Add `WATSONX_MODEL_ID=ibm/granite-3-3-8b-instruct` to
      `src/.env.example`.
- [ ] Update the docstring at the top of `src/app.py` to describe the AI
      features and required env vars.

### Relevant Context
- `src/.env.example` already has `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`,
  and `WATSONX_URL`.
- `src/requirements.txt` was created alongside `app.py` with
  `streamlit`, `pandas`, `numpy`.

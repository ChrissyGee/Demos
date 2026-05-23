"""
================================================================================
AI-POWERED INVENTORY MANAGEMENT — STREAMLIT MVP
================================================================================

A self-contained demo of a multi-agent inventory system.

ARCHITECTURE (matches the whiteboard sketch):

    +---------+        +-----------+        +---------------------+
    |  Chat   |  --->  |   Lead    |  --->  |  Inventory Display  |
    | (User)  |        |   Agent   |        |     + Metrics       |
    +---------+        +-----+-----+        +----------+----------+
                             |                         ^
                             v                         |
                       +----------+              +-----+-----+
                       | Console  |              | Assistant |
                       | (Reason) |              |   Agent   |
                       +----------+              +-----+-----+
                                                       |
                                            +----------+----------+
                                            |       Tools         |
                                            |  - Weather API      |
                                            |  - Social Trends    |
                                            |  - Demand Patterns  |
                                            |  - Supply Chain     |
                                            +---------------------+

KEY IDEAS
    * The Lead Agent runs on a tick, monitoring inventory and proposing
      replenishment based on tool-augmented forecasts from the Assistant Agent.
    * The Console captures the agent's step-by-step reasoning so the user
      can see *why* the system is acting.
    * Forecasts use numpy polyfit (degree-1 linear regression) trained on
      synthetic historical demand (plus a moving-average fallback).
    * If an OPENAI_API_KEY is present, the Chat tab uses OpenAI; otherwise
      it falls back to a deterministic rule-based responder so the demo
      always runs.

USAGE
    pip install streamlit pandas numpy scikit-learn openai
    streamlit run app.py
================================================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# OpenAI is optional — the app works without an API key.
try:
    from openai import OpenAI
    _OPENAI_IMPORTED = True
except Exception:
    _OPENAI_IMPORTED = False


# ============================================================
# PAGE CONFIG  (must be the first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="AI Inventory Manager",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS & SEED DATA
# ============================================================

# A small but realistic product catalog. Each row seeds a SKU.
SEED_PRODUCTS = [
    # (sku,    name,             category,      unit_cost, price,  lead_time_days)
    ("SKU001", "Umbrella",       "Outdoor",     4.50,      14.99,  3),
    ("SKU002", "Sunscreen SPF50","Health",      3.20,      11.49,  2),
    ("SKU003", "Wireless Earbuds","Electronics",22.00,     79.00,  7),
    ("SKU004", "Yoga Mat",       "Fitness",     8.75,      29.99,  4),
    ("SKU005", "Coffee Beans 1kg","Grocery",    9.10,      22.50,  2),
    ("SKU006", "LED Desk Lamp",  "Home",        11.40,     34.99,  5),
    ("SKU007", "Running Shoes",  "Fitness",     28.00,     89.00,  6),
    ("SKU008", "Notebook A5",    "Stationery",  1.20,      5.99,   3),
]

# How many days of synthetic history to generate per SKU.
HISTORY_DAYS = 60

# Daily holding cost as a % of unit cost (used in savings calc).
HOLDING_COST_RATE = 0.001  # 0.1% per day

# Stockout penalty (lost margin per unit that goes unfulfilled).
STOCKOUT_PENALTY_MULT = 1.5


# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def generate_historical_demand(sku: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """
    Build a synthetic daily-demand series for a single SKU.

    The series combines:
        * A baseline mean (per-SKU, deterministic from the sku hash)
        * A weekly seasonality wave
        * Gaussian noise
        * A mild upward trend

    Parameters
    ----------
    sku : str
        Product identifier — used to seed the baseline so each SKU is unique
        but reproducible across reruns.
    days : int
        Number of days of history to generate.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'sku', 'units_sold'].
    """
    # Deterministic per-SKU seed so the same SKU always gets the same history.
    rng = np.random.default_rng(abs(hash(sku)) % (2**32))
    baseline = rng.integers(8, 30)
    trend = rng.uniform(0.02, 0.10)

    dates = pd.date_range(end=datetime.today().date(), periods=days)
    t = np.arange(days)

    seasonal = 3 * np.sin(2 * np.pi * t / 7)          # weekly cycle
    noise = rng.normal(0, 2, size=days)
    series = baseline + trend * t + seasonal + noise
    series = np.clip(series, 0, None).round().astype(int)

    return pd.DataFrame({"date": dates, "sku": sku, "units_sold": series})


def init_inventory() -> pd.DataFrame:
    """
    Build the starting inventory table from `SEED_PRODUCTS`.

    Stock levels are randomised within a reasonable range so the demo has
    some SKUs near reorder threshold and others well-stocked.
    """
    rows = []
    for sku, name, category, cost, price, lead in SEED_PRODUCTS:
        rows.append({
            "sku": sku,
            "name": name,
            "category": category,
            "unit_cost": cost,
            "price": price,
            "lead_time_days": lead,
            "stock": random.randint(20, 180),
            "reorder_point": random.randint(40, 70),
            "on_order": 0,
        })
    return pd.DataFrame(rows)


def init_signals() -> Dict[str, dict]:
    """
    Build a fake external-signals dictionary keyed by category.

    Each category gets a weather-style signal and a social-buzz score.
    Categories like 'Outdoor' care about weather; 'Fitness' cares about
    social trends. The Assistant Agent reads from this dict.
    """
    return {
        "Outdoor":     {"weather": "Heavy Rain Forecast", "social_buzz": 0.65},
        "Health":      {"weather": "Heatwave Incoming",   "social_buzz": 0.80},
        "Electronics": {"weather": "Stable",              "social_buzz": 0.55},
        "Fitness":     {"weather": "Mild",                "social_buzz": 0.90},
        "Grocery":     {"weather": "Stable",              "social_buzz": 0.40},
        "Home":        {"weather": "Cold Snap",           "social_buzz": 0.45},
        "Stationery":  {"weather": "Stable",              "social_buzz": 0.25},
    }


# ============================================================
# SESSION STATE BOOTSTRAP
# ============================================================

def init_session_state() -> None:
    """
    Initialise every key used by the app exactly once per session.

    Streamlit reruns the whole script on every interaction, so anything that
    must survive across reruns lives in `st.session_state`.
    """
    if "initialised" in st.session_state:
        return

    # Core data.
    st.session_state.inventory = init_inventory()
    st.session_state.signals = init_signals()

    # Per-SKU demand history — a dict of DataFrames for fast lookup.
    st.session_state.history = {
        sku: generate_historical_demand(sku) for sku, *_ in SEED_PRODUCTS
    }

    # Operational metrics that the dashboard surfaces.
    st.session_state.metrics = {
        "stockouts_avoided": 0,
        "holding_cost_saved": 0.0,
        "replenishment_orders": 0,
        "ticks": 0,
    }

    # Console log — list of {timestamp, agent, message} dicts.
    st.session_state.console: List[dict] = []

    # Chat transcript — list of {role, content} dicts.
    st.session_state.chat: List[dict] = []

    # Auto-tick toggle (Lead Agent runs every interaction when on).
    st.session_state.auto_tick = False

    st.session_state.initialised = True


# ============================================================
# FORECASTING
# ============================================================

def forecast_demand(history: pd.DataFrame, horizon_days: int = 7) -> float:
    """
    Predict total demand for the next `horizon_days` days.

    Uses numpy polyfit (degree-1 polynomial) for linear regression on the
    historical series. If there are too few points (e.g. early in the
    simulation), falls back to a 7-day moving average — a common technique
    in retail forecasting.

    Parameters
    ----------
    history : pd.DataFrame
        Must contain a 'units_sold' column ordered by date ascending.
    horizon_days : int
        Forecast horizon — how many days ahead to total.

    Returns
    -------
    float
        Expected total units sold across the horizon (clipped at zero).
    """
    series = history["units_sold"].to_numpy()

    if len(series) < 14:
        # Not enough data — use a simple moving average.
        return float(max(0, np.mean(series[-7:]) * horizon_days))

    X = np.arange(len(series))
    coeffs = np.polyfit(X, series, 1)   # degree-1 polynomial = linear regression

    future_X = np.arange(len(series), len(series) + horizon_days)
    predictions = np.polyval(coeffs, future_X)
    return float(max(0, predictions.sum()))


# ============================================================
# ASSISTANT AGENT — TOOLS
# ============================================================
# Each tool is a pure function that returns a small dict the Lead Agent
# can reason over. In a real deployment these would call external APIs.

def tool_weather_api(category: str) -> dict:
    """Simulated weather signal for a product category."""
    signal = st.session_state.signals.get(category, {}).get("weather", "Stable")
    # Map weather text to a demand multiplier the planner can use directly.
    multiplier = {
        "Heavy Rain Forecast": 1.6,  # umbrellas, raincoats spike
        "Heatwave Incoming":   1.8,  # sunscreen, drinks spike
        "Cold Snap":           1.3,
        "Mild":                1.0,
        "Stable":              1.0,
    }.get(signal, 1.0)
    return {"tool": "weather_api", "signal": signal, "demand_multiplier": multiplier}


def tool_social_trends(category: str) -> dict:
    """Simulated social-media buzz score (0..1) for a product category."""
    buzz = st.session_state.signals.get(category, {}).get("social_buzz", 0.3)
    # Buzz scales between 1x (no buzz) and 1.5x (viral).
    multiplier = 1.0 + buzz * 0.5
    return {"tool": "social_trends", "buzz": buzz, "demand_multiplier": multiplier}


def tool_demand_patterns(sku: str) -> dict:
    """Look at the last 14 days of history to spot a trend."""
    hist = st.session_state.history[sku].tail(14)["units_sold"].to_numpy()
    recent_avg = float(np.mean(hist))
    older_avg = float(np.mean(hist[:7])) if len(hist) >= 14 else recent_avg
    trend = "rising" if recent_avg > older_avg * 1.05 else (
        "falling" if recent_avg < older_avg * 0.95 else "flat"
    )
    return {
        "tool": "demand_patterns",
        "recent_avg_units_per_day": round(recent_avg, 2),
        "trend": trend,
    }


def tool_supply_chain(sku: str) -> dict:
    """Simulated supplier reliability + lead-time risk."""
    inv = st.session_state.inventory
    row = inv.loc[inv["sku"] == sku].iloc[0]
    lead = int(row["lead_time_days"])
    # Random reliability per SKU — seeded for stability within a session.
    rng = random.Random(sku)
    reliability = round(rng.uniform(0.80, 0.99), 2)
    risk = "low" if reliability > 0.9 and lead <= 4 else (
        "medium" if reliability > 0.85 else "high"
    )
    return {
        "tool": "supply_chain",
        "lead_time_days": lead,
        "supplier_reliability": reliability,
        "risk": risk,
    }


# ============================================================
# CONSOLE LOGGING
# ============================================================

def log(agent: str, message: str) -> None:
    """
    Append a reasoning step to the console log.

    The console is rendered in both the sidebar (compact) and the Console
    tab (full). Each entry is timestamped so the user can follow the agent
    over time.
    """
    st.session_state.console.append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "message": message,
    })
    # Cap the buffer so very long sessions don't grow without bound.
    if len(st.session_state.console) > 250:
        st.session_state.console = st.session_state.console[-250:]


# ============================================================
# LEAD AGENT
# ============================================================

def lead_agent_tick() -> None:
    """
    One 'tick' of the Lead Agent.

    For each SKU it:
        1. Simulates a day's sales (depleting stock).
        2. Calls the Assistant Agent's tools to gather signals.
        3. Combines a polyfit forecast with the signal multipliers.
        4. Decides whether to place a replenishment order.
        5. Logs every step to the console for transparency.
    """
    st.session_state.metrics["ticks"] += 1
    tick_num = st.session_state.metrics["ticks"]
    log("LeadAgent", f"--- Tick #{tick_num} starting ---")

    inv = st.session_state.inventory

    for idx, row in inv.iterrows():
        sku, name, category = row["sku"], row["name"], row["category"]

        # --- Step 1: simulate today's actual sales ---------------------
        recent_mean = float(np.mean(
            st.session_state.history[sku].tail(7)["units_sold"]
        ))
        units_sold_today = max(0, int(np.random.normal(recent_mean, 3)))
        units_sold_today = min(units_sold_today, int(row["stock"]))

        # Update the history with today's actual.
        new_row = pd.DataFrame([{
            "date": datetime.today().date(),
            "sku": sku,
            "units_sold": units_sold_today,
        }])
        st.session_state.history[sku] = pd.concat(
            [st.session_state.history[sku], new_row], ignore_index=True
        ).tail(HISTORY_DAYS + 30)  # keep memory bounded

        inv.at[idx, "stock"] = int(row["stock"]) - units_sold_today

        # --- Step 2: gather signals via Assistant tools ----------------
        weather = tool_weather_api(category)
        social = tool_social_trends(category)
        pattern = tool_demand_patterns(sku)
        supply = tool_supply_chain(sku)

        # --- Step 3: forecast and adjust with signals ------------------
        base_forecast = forecast_demand(st.session_state.history[sku], horizon_days=7)
        adjusted_forecast = base_forecast * weather["demand_multiplier"] * social["demand_multiplier"]

        # --- Step 4: decide whether to reorder -------------------------
        projected_stock = inv.at[idx, "stock"] - adjusted_forecast
        reorder_point = int(row["reorder_point"])
        on_order = int(row["on_order"])

        if projected_stock < reorder_point and on_order == 0:
            # Reorder enough to cover 2x the forecast horizon plus safety.
            reorder_qty = int(adjusted_forecast * 2 + reorder_point - projected_stock)
            inv.at[idx, "on_order"] = reorder_qty
            st.session_state.metrics["replenishment_orders"] += 1
            st.session_state.metrics["stockouts_avoided"] += 1
            # Holding cost saved is approximated as the cost we *didn't* incur
            # by ordering exactly what's needed vs. a naive overstock policy.
            saved = float(row["unit_cost"]) * reorder_qty * HOLDING_COST_RATE * 30
            st.session_state.metrics["holding_cost_saved"] += saved

            log("LeadAgent",
                f"[{sku} {name}] stock={inv.at[idx,'stock']}, "
                f"7d forecast={base_forecast:.0f} → adjusted={adjusted_forecast:.0f} "
                f"(weather: {weather['signal']}, buzz: {social['buzz']:.2f}, "
                f"trend: {pattern['trend']}, supply risk: {supply['risk']}). "
                f"Reordering {reorder_qty} units.")
        else:
            log("LeadAgent",
                f"[{sku} {name}] stock={inv.at[idx,'stock']}, forecast OK "
                f"(projected={projected_stock:.0f} ≥ reorder={reorder_point}). No action.")

        # --- Step 5: receive a delivery if one was on order ------------
        # Simple model: every tick has a chance of an outstanding order arriving.
        if on_order > 0 and random.random() < 0.4:
            inv.at[idx, "stock"] = int(inv.at[idx, "stock"]) + on_order
            log("LeadAgent",
                f"[{sku} {name}] Delivery received: +{on_order} units "
                f"(new stock={inv.at[idx,'stock']}).")
            inv.at[idx, "on_order"] = 0

    st.session_state.inventory = inv
    log("LeadAgent", f"--- Tick #{tick_num} complete ---")


# ============================================================
# CHAT AGENT — uses OpenAI if available, falls back otherwise
# ============================================================

def get_openai_client() -> Optional["OpenAI"]:
    """Return an OpenAI client if the key is set, else None."""
    if not _OPENAI_IMPORTED:
        return None
    key = os.environ.get("OPENAI_API_KEY") or st.session_state.get("openai_key")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


def build_context_snapshot() -> str:
    """
    Render a compact text snapshot of inventory + signals.

    Used as system context for the chat agent so its answers are grounded
    in the current simulation state.
    """
    inv = st.session_state.inventory
    lines = ["CURRENT INVENTORY:"]
    for _, r in inv.iterrows():
        lines.append(
            f"  - {r['sku']} {r['name']} ({r['category']}): "
            f"stock={r['stock']}, reorder_point={r['reorder_point']}, "
            f"on_order={r['on_order']}"
        )
    lines.append("\nEXTERNAL SIGNALS:")
    for cat, sig in st.session_state.signals.items():
        lines.append(f"  - {cat}: weather={sig['weather']}, social_buzz={sig['social_buzz']}")
    m = st.session_state.metrics
    lines.append(
        f"\nMETRICS: stockouts_avoided={m['stockouts_avoided']}, "
        f"holding_cost_saved=${m['holding_cost_saved']:.2f}, "
        f"replenishment_orders={m['replenishment_orders']}, ticks={m['ticks']}"
    )
    return "\n".join(lines)


def chat_respond(user_message: str) -> str:
    """
    Answer the user's question about inventory.

    Tries OpenAI first; falls back to a deterministic rule-based responder
    that scans the inventory snapshot for keywords. This keeps the demo
    fully functional offline.
    """
    log("ChatAgent", f"User asked: {user_message}")
    context = build_context_snapshot()

    client = get_openai_client()
    if client is not None:
        try:
            log("ChatAgent", "Routing question to OpenAI with live inventory context.")
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content":
                        "You are an inventory analyst. Answer using ONLY the "
                        "supplied context. Be concise (under 120 words). "
                        "If the user asks for a recommendation, justify it "
                        "with the numbers in the context."},
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log("ChatAgent", f"OpenAI call failed ({e}); falling back to local responder.")

    # --- Offline fallback: simple keyword router ---------------------------
    log("ChatAgent", "Using offline rule-based responder (no API key detected).")
    q = user_message.lower()
    inv = st.session_state.inventory

    if "low" in q or "stockout" in q or "running out" in q:
        low = inv[inv["stock"] < inv["reorder_point"]]
        if low.empty:
            return "No SKUs are currently below their reorder point. Inventory is healthy."
        items = ", ".join(f"{r['name']} ({r['stock']} left)" for _, r in low.iterrows())
        return f"These SKUs are below reorder point: {items}."

    if "metric" in q or "savings" in q or "performance" in q:
        m = st.session_state.metrics
        return (f"So far: {m['stockouts_avoided']} stockouts avoided, "
                f"${m['holding_cost_saved']:.2f} in holding cost saved across "
                f"{m['replenishment_orders']} replenishment orders.")

    if "weather" in q or "social" in q or "trend" in q:
        return (f"Current external signals:\n{context.split('EXTERNAL SIGNALS:')[1].split('METRICS')[0]}")

    if "forecast" in q or "predict" in q:
        # Forecast the highest-stock SKU as an example.
        sku = inv.iloc[0]["sku"]
        f = forecast_demand(st.session_state.history[sku])
        return f"7-day forecast for {inv.iloc[0]['name']} ({sku}): ~{f:.0f} units."

    return ("I can answer questions about stock levels, low SKUs, current "
            "metrics, external signals (weather/social), and demand forecasts. "
            "Try: 'What's running low?' or 'Show me the savings so far.'")


# ============================================================
# UI HELPERS
# ============================================================

def render_console_panel(container, max_entries: int = 50) -> None:
    """Render the most recent N console entries inside any container."""
    entries = st.session_state.console[-max_entries:][::-1]
    if not entries:
        container.info("Console is empty. Run a Lead Agent tick to populate.")
        return
    for entry in entries:
        prefix = "🧠" if entry["agent"] == "LeadAgent" else "💬"
        container.markdown(
            f"`{entry['ts']}` {prefix} **{entry['agent']}** — {entry['message']}"
        )


def colour_stock(val, reorder):
    """Style helper: red if below reorder point, amber if within 20%."""
    if val < reorder:
        return "background-color: #ffcccc"
    if val < reorder * 1.2:
        return "background-color: #fff2cc"
    return ""


# ============================================================
# TAB: DASHBOARD
# ============================================================

def render_dashboard() -> None:
    """The main real-time view: metrics, inventory table, controls."""
    st.subheader("📊 Real-time Inventory Dashboard")

    # --- Top-line KPIs --------------------------------------------------
    m = st.session_state.metrics
    inv = st.session_state.inventory
    total_units = int(inv["stock"].sum())
    total_value = float((inv["stock"] * inv["unit_cost"]).sum())
    low_stock_count = int((inv["stock"] < inv["reorder_point"]).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Units in Stock", f"{total_units:,}")
    c2.metric("Inventory Value", f"${total_value:,.0f}")
    c3.metric("SKUs Below Reorder", low_stock_count,
              delta=None if low_stock_count == 0 else "Action needed",
              delta_color="inverse")
    c4.metric("Stockouts Avoided", m["stockouts_avoided"])
    c5.metric("Holding Cost Saved", f"${m['holding_cost_saved']:,.2f}")

    st.divider()

    # --- Agent controls -------------------------------------------------
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    if cc1.button("▶ Run Lead Agent Tick", type="primary", use_container_width=True):
        lead_agent_tick()
        st.rerun()

    auto = cc2.toggle("Auto-tick on rerun", value=st.session_state.auto_tick,
                      help="When on, the Lead Agent runs every time the page reruns.")
    st.session_state.auto_tick = auto

    if cc3.button("🔄 Reset Simulation", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.divider()

    # --- Inventory table with conditional formatting --------------------
    st.markdown("#### Live Inventory")
    display = inv[[
        "sku", "name", "category", "stock", "reorder_point",
        "on_order", "unit_cost", "price", "lead_time_days"
    ]].copy()
    styled = display.style.apply(
        lambda r: [colour_stock(r["stock"], r["reorder_point"]) if c == "stock" else ""
                   for c in display.columns],
        axis=1,
    ).format({"unit_cost": "${:.2f}", "price": "${:.2f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Demand chart for a selected SKU --------------------------------
    st.markdown("#### Demand History & Forecast")
    pick = st.selectbox("Select a SKU", inv["sku"] + " — " + inv["name"])
    sku = pick.split(" — ")[0]
    hist = st.session_state.history[sku].copy()
    # Append the forecast as a dotted continuation.
    forecast_total = forecast_demand(hist, horizon_days=7)
    forecast_daily = forecast_total / 7.0
    future_dates = pd.date_range(
        start=hist["date"].max() + timedelta(days=1), periods=7
    )
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "units_sold": [forecast_daily] * 7,
        "kind": "forecast",
    })
    hist["kind"] = "actual"
    chart_df = pd.concat([hist[["date", "units_sold", "kind"]], forecast_df])
    chart_df = chart_df.pivot(index="date", columns="kind", values="units_sold")
    st.line_chart(chart_df, use_container_width=True)


# ============================================================
# TAB: CHAT
# ============================================================

def render_chat() -> None:
    """Conversational interface backed by OpenAI (or offline fallback)."""
    st.subheader("💬 Ask the Inventory Assistant")
    st.caption(
        "Ask about stock levels, signals, savings, or forecasts. "
        "Set `OPENAI_API_KEY` for live LLM answers — otherwise a built-in "
        "rule-based responder handles the questions."
    )

    # Optional in-app key entry so the demo can be tested without env vars.
    with st.expander("🔑 OpenAI API key (optional)"):
        st.text_input(
            "API key (stored only in this session)",
            type="password",
            key="openai_key",
        )
        if not _OPENAI_IMPORTED:
            st.info("The `openai` package is not installed. Using offline responder.")

    # Render past messages.
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user input.
    user_input = st.chat_input("e.g. What's running low?")
    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        reply = chat_respond(user_input)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


# ============================================================
# TAB: CONSOLE
# ============================================================

def render_console_tab() -> None:
    """Full-width console view of every reasoning step."""
    st.subheader("🧠 Agent Reasoning Console")
    st.caption(
        "Every decision the Lead Agent makes is logged here, including the "
        "tool calls and signals it consulted. This is what makes the system "
        "auditable rather than a black box."
    )
    col1, col2 = st.columns([1, 1])
    if col1.button("Clear Console"):
        st.session_state.console = []
        st.rerun()
    col2.metric("Log Entries", len(st.session_state.console))
    render_console_panel(st, max_entries=250)


# ============================================================
# TAB: INTEGRATION STEPS
# ============================================================

def render_integration() -> None:
    """Guidance on swapping the simulated parts for real systems."""
    st.subheader("🔌 Connect to Real APIs")
    st.markdown(
        """
        This MVP simulates every external system. To productionise it,
        replace the four tool functions and the data sources below.

        ---

        ### 1. Inventory source of truth
        Replace `init_inventory()` with a query against your ERP / WMS:

        ```python
        # Example: pull from Shopify, NetSuite, SAP, Odoo, etc.
        inventory_df = erp_client.fetch_inventory()
        ```

        ### 2. Weather API
        Swap `tool_weather_api()` for a real call:

        ```python
        import requests
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"q": store_city, "appid": API_KEY},
        )
        ```

        ### 3. Social trends
        Swap `tool_social_trends()` for a real signal:
        - Google Trends (via `pytrends`)
        - X/Twitter API search volume
        - TikTok hashtag analytics

        ### 4. Supply-chain telemetry
        Swap `tool_supply_chain()` for vendor EDI feeds or a supplier
        scorecard service (e.g. project44, FourKites).

        ### 5. Demand forecasting
        The current `np.polyfit` linear model is intentionally simple. Production
        alternatives:
        - Prophet for seasonality
        - LightGBM / XGBoost with engineered features
        - Amazon Forecast / Vertex AI Forecast managed services

        ### 6. LLM provider
        The `chat_respond()` function uses OpenAI when `OPENAI_API_KEY`
        is set. You can swap in Anthropic, Bedrock, or a self-hosted model
        by changing only that one function.

        ### 7. Orchestration
        For a multi-tenant deployment, move the Lead Agent loop out of
        Streamlit and into a background worker (Celery, Temporal, or a
        cron job) and let Streamlit subscribe to the metrics it produces.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    """Always-visible mini console + run controls."""
    with st.sidebar:
        st.markdown("## 📦 AI Inventory Manager")
        st.caption("Multi-agent demo · Streamlit MVP")

        st.divider()
        st.markdown("### ⚙️ Quick Controls")
        if st.button("Run Tick", use_container_width=True):
            lead_agent_tick()
            st.rerun()

        st.divider()
        st.markdown("### 🧠 Console (latest)")
        # Compact console — newest first, last 20 entries.
        with st.container(height=400):
            render_console_panel(st, max_entries=20)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Top-level app entry point."""
    init_session_state()

    # Auto-tick: run the Lead Agent on each rerun if the toggle is on.
    # This gives the appearance of a live, self-running system.
    if st.session_state.auto_tick:
        lead_agent_tick()

    render_sidebar()

    st.title("📦 AI-Powered Inventory Management")
    st.caption(
        "Lead Agent monitors stock in real time · Assistant Agent gathers "
        "external signals via tools · Chat lets you ask questions · "
        "Console shows every reasoning step."
    )

    tab_dashboard, tab_chat, tab_console, tab_integration = st.tabs(
        ["📊 Dashboard", "💬 Chat", "🧠 Console", "🔌 Integration Steps"]
    )

    with tab_dashboard:
        render_dashboard()
    with tab_chat:
        render_chat()
    with tab_console:
        render_console_tab()
    with tab_integration:
        render_integration()


if __name__ == "__main__":
    main()

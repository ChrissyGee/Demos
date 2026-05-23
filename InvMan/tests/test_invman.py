"""
Unit tests for InvMan/app.py.

Key coverage:
  - forecast_demand: the refactored function (LinearRegression → np.polyfit)
  - generate_historical_demand: synthetic data generator
  - init_signals / init_inventory: seed data helpers
  - tool_weather_api, tool_social_trends: signal tools (mocked session state)
  - tool_demand_patterns: trend detection (mocked history)

Streamlit is mocked before import.  Tool functions that access session state
are given a properly structured mock so they run against realistic data.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock streamlit before importing app
# ---------------------------------------------------------------------------
_st = MagicMock()
_st.session_state = MagicMock()
_st.session_state.console = []
_st.session_state.get = MagicMock(return_value="")
sys.modules["streamlit"] = _st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app  # noqa: E402


# ---------------------------------------------------------------------------
# Module-level session-state fixtures for tool tests
# ---------------------------------------------------------------------------
# Build real data once so every tool test class can reference it.
_SIGNALS = app.init_signals()
_INVENTORY = app.init_inventory()
_HISTORY = {sku: app.generate_historical_demand(sku)
            for sku, *_ in app.SEED_PRODUCTS}


def _setup_session():
    """Wire the shared data into the mocked session_state."""
    _st.session_state.signals = _SIGNALS
    _st.session_state.inventory = _INVENTORY
    _st.session_state.history = _HISTORY


_setup_session()


# ---------------------------------------------------------------------------
# forecast_demand  ← THE REFACTORED FUNCTION
# ---------------------------------------------------------------------------

class TestForecastDemand:

    def test_returns_float(self):
        hist = app.generate_historical_demand("SKU001")
        result = app.forecast_demand(hist, horizon_days=7)
        assert isinstance(result, float)

    def test_result_is_nonnegative(self):
        hist = app.generate_historical_demand("SKU001")
        result = app.forecast_demand(hist, horizon_days=7)
        assert result >= 0.0

    def test_all_zero_history_returns_zero(self):
        df = pd.DataFrame({"units_sold": [0] * 20})
        result = app.forecast_demand(df, horizon_days=7)
        assert result == 0.0

    def test_short_history_uses_moving_average_fallback(self):
        # < 14 points → moving-average branch
        df = pd.DataFrame({"units_sold": [10] * 7})
        result = app.forecast_demand(df, horizon_days=7)
        # Moving average of 7 × 10s over 7 days = 70
        assert result == pytest.approx(70.0, abs=1e-6)

    def test_exactly_14_points_uses_polyfit(self):
        # Exactly 14 points: uses the polyfit branch (>= 14 check)
        df = pd.DataFrame({"units_sold": [5] * 14})
        result = app.forecast_demand(df, horizon_days=7)
        assert result >= 0.0

    def test_linear_upward_trend_projects_forward(self):
        # Perfectly linear series 1..20 — forecast should be > mean
        series = list(range(1, 21))
        df = pd.DataFrame({"units_sold": series})
        result = app.forecast_demand(df, horizon_days=7)
        mean_7 = float(np.mean(series[-7:])) * 7  # naive MA
        assert result > mean_7  # linear fit extrapolates above MA for rising series

    def test_larger_horizon_gives_larger_total(self):
        hist = app.generate_historical_demand("SKU002")
        f7 = app.forecast_demand(hist, horizon_days=7)
        f14 = app.forecast_demand(hist, horizon_days=14)
        assert f14 >= f7

    def test_known_constant_series_gives_correct_forecast(self):
        # Constant series of 10 → linear fit slope=0, intercept=10
        # Forecast for any horizon = 10 * horizon_days
        df = pd.DataFrame({"units_sold": [10] * 20})
        result = app.forecast_demand(df, horizon_days=5)
        assert result == pytest.approx(50.0, abs=1.0)

    def test_polyfit_result_matches_manual_calculation(self):
        # Manual verification: series = [1,2,...,20], horizon=7
        n = 20
        series = np.arange(1, n + 1, dtype=float)
        df = pd.DataFrame({"units_sold": series})
        coeffs = np.polyfit(np.arange(n), series, 1)
        future_x = np.arange(n, n + 7)
        expected = float(max(0, np.polyval(coeffs, future_x).sum()))
        result = app.forecast_demand(df, horizon_days=7)
        assert result == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# generate_historical_demand
# ---------------------------------------------------------------------------

class TestGenerateHistoricalDemand:

    def test_returns_dataframe(self):
        df = app.generate_historical_demand("SKU001")
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        df = app.generate_historical_demand("SKU001", days=30)
        assert len(df) == 30

    def test_default_days_equals_history_days_constant(self):
        df = app.generate_historical_demand("SKU001")
        assert len(df) == app.HISTORY_DAYS

    def test_required_columns(self):
        df = app.generate_historical_demand("SKU001")
        assert {"date", "sku", "units_sold"}.issubset(set(df.columns))

    def test_sku_column_matches_input(self):
        df = app.generate_historical_demand("SKU003")
        assert (df["sku"] == "SKU003").all()

    def test_units_sold_are_nonnegative(self):
        df = app.generate_historical_demand("SKU001")
        assert (df["units_sold"] >= 0).all()

    def test_units_sold_are_integers(self):
        df = app.generate_historical_demand("SKU001")
        assert df["units_sold"].dtype in (np.int32, np.int64, int)

    def test_deterministic_for_same_sku(self):
        df1 = app.generate_historical_demand("SKU005")
        df2 = app.generate_historical_demand("SKU005")
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_skus_produce_different_series(self):
        df1 = app.generate_historical_demand("SKU001")
        df2 = app.generate_historical_demand("SKU002")
        assert not df1["units_sold"].equals(df2["units_sold"])


# ---------------------------------------------------------------------------
# init_signals
# ---------------------------------------------------------------------------

class TestInitSignals:

    def test_returns_dict(self):
        assert isinstance(app.init_signals(), dict)

    def test_contains_all_expected_categories(self):
        signals = app.init_signals()
        for category in ["Outdoor", "Health", "Electronics", "Fitness",
                         "Grocery", "Home", "Stationery"]:
            assert category in signals

    def test_all_entries_have_weather_key(self):
        for cat, sig in app.init_signals().items():
            assert "weather" in sig, f"Missing 'weather' for {cat}"

    def test_all_entries_have_social_buzz_key(self):
        for cat, sig in app.init_signals().items():
            assert "social_buzz" in sig, f"Missing 'social_buzz' for {cat}"

    def test_social_buzz_in_valid_range(self):
        for cat, sig in app.init_signals().items():
            assert 0.0 <= sig["social_buzz"] <= 1.0, f"Out-of-range buzz for {cat}"

    def test_weather_values_are_strings(self):
        for cat, sig in app.init_signals().items():
            assert isinstance(sig["weather"], str)


# ---------------------------------------------------------------------------
# init_inventory
# ---------------------------------------------------------------------------

class TestInitInventory:

    def test_returns_dataframe(self):
        assert isinstance(app.init_inventory(), pd.DataFrame)

    def test_correct_number_of_rows(self):
        assert len(app.init_inventory()) == len(app.SEED_PRODUCTS)

    def test_required_columns(self):
        expected = {"sku", "name", "category", "unit_cost", "price",
                    "lead_time_days", "stock", "reorder_point", "on_order"}
        assert expected.issubset(set(app.init_inventory().columns))

    def test_all_stock_values_positive(self):
        assert (app.init_inventory()["stock"] > 0).all()

    def test_all_reorder_points_positive(self):
        assert (app.init_inventory()["reorder_point"] > 0).all()

    def test_on_order_initialized_to_zero(self):
        assert (app.init_inventory()["on_order"] == 0).all()

    def test_unit_costs_positive(self):
        assert (app.init_inventory()["unit_cost"] > 0).all()

    def test_skus_match_seed_products(self):
        inv = app.init_inventory()
        expected_skus = {row[0] for row in app.SEED_PRODUCTS}
        assert set(inv["sku"]) == expected_skus


# ---------------------------------------------------------------------------
# tool_weather_api
# ---------------------------------------------------------------------------

class TestToolWeatherApi:

    def test_returns_required_keys(self):
        result = app.tool_weather_api("Outdoor")
        assert {"tool", "signal", "demand_multiplier"}.issubset(set(result.keys()))

    def test_tool_key_is_correct(self):
        assert app.tool_weather_api("Outdoor")["tool"] == "weather_api"

    def test_heavy_rain_gives_multiplier_above_1(self):
        # Outdoor has "Heavy Rain Forecast" → multiplier 1.6
        result = app.tool_weather_api("Outdoor")
        assert result["demand_multiplier"] > 1.0

    def test_stable_weather_gives_multiplier_of_1(self):
        # Electronics has "Stable" → multiplier 1.0
        result = app.tool_weather_api("Electronics")
        assert result["demand_multiplier"] == pytest.approx(1.0)

    def test_unknown_category_defaults_to_multiplier_1(self):
        result = app.tool_weather_api("UnknownCategory")
        assert result["demand_multiplier"] == pytest.approx(1.0)

    def test_heatwave_gives_multiplier_above_1(self):
        # Health has "Heatwave Incoming" → multiplier 1.8
        result = app.tool_weather_api("Health")
        assert result["demand_multiplier"] > 1.0


# ---------------------------------------------------------------------------
# tool_social_trends
# ---------------------------------------------------------------------------

class TestToolSocialTrends:

    def test_returns_required_keys(self):
        result = app.tool_social_trends("Fitness")
        assert {"tool", "buzz", "demand_multiplier"}.issubset(set(result.keys()))

    def test_tool_key_is_correct(self):
        assert app.tool_social_trends("Fitness")["tool"] == "social_trends"

    def test_high_buzz_gives_multiplier_above_1(self):
        # Fitness has buzz=0.90, multiplier = 1 + 0.9 * 0.5 = 1.45
        result = app.tool_social_trends("Fitness")
        assert result["demand_multiplier"] > 1.0

    def test_low_buzz_gives_multiplier_close_to_1(self):
        # Stationery has buzz=0.25, multiplier = 1 + 0.25 * 0.5 = 1.125
        result = app.tool_social_trends("Stationery")
        assert 1.0 < result["demand_multiplier"] < 1.3

    def test_multiplier_formula(self):
        # Verify formula: multiplier = 1.0 + buzz * 0.5
        result = app.tool_social_trends("Fitness")
        expected = 1.0 + result["buzz"] * 0.5
        assert result["demand_multiplier"] == pytest.approx(expected)

    def test_unknown_category_uses_default_buzz(self):
        result = app.tool_social_trends("UnknownCategory")
        # Default buzz is 0.3 → multiplier = 1.0 + 0.3 * 0.5 = 1.15
        assert result["demand_multiplier"] == pytest.approx(1.15)


# ---------------------------------------------------------------------------
# tool_demand_patterns
# ---------------------------------------------------------------------------

class TestToolDemandPatterns:

    def _make_history(self, series: list, sku: str = "TSKU") -> None:
        """Inject a synthetic history series into the mocked session state."""
        df = pd.DataFrame({
            "date": pd.date_range(end="2026-05-23", periods=len(series)),
            "sku": sku,
            "units_sold": series,
        })
        _st.session_state.history = {**_HISTORY, sku: df}

    def test_returns_required_keys(self):
        _setup_session()
        result = app.tool_demand_patterns("SKU001")
        assert {"tool", "recent_avg_units_per_day", "trend"}.issubset(set(result.keys()))

    def test_tool_key_is_correct(self):
        _setup_session()
        assert app.tool_demand_patterns("SKU001")["tool"] == "demand_patterns"

    def test_rising_trend_detected(self):
        # Last 7 days clearly higher than first 7 days → "rising"
        series = [5] * 7 + [20] * 7
        self._make_history(series, "RSKU")
        result = app.tool_demand_patterns("RSKU")
        assert result["trend"] == "rising"

    def test_falling_trend_detected(self):
        # Last 7 days clearly lower than first 7 days → "falling"
        series = [20] * 7 + [5] * 7
        self._make_history(series, "FSKU")
        result = app.tool_demand_patterns("FSKU")
        assert result["trend"] == "falling"

    def test_flat_trend_detected(self):
        # Identical values → no trend → "flat"
        series = [10] * 14
        self._make_history(series, "FLATSKU")
        result = app.tool_demand_patterns("FLATSKU")
        assert result["trend"] == "flat"

    def test_recent_avg_is_positive_for_positive_series(self):
        _setup_session()
        result = app.tool_demand_patterns("SKU001")
        assert result["recent_avg_units_per_day"] > 0

"""
Unit tests for DecisionMaker/app.py.

All pure-logic functions are tested in isolation.
Streamlit is mocked before import so module-level st calls are safe.
"""

import os
import sys
import importlib.util
import sqlite3
import datetime
import pandas as pd
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock streamlit + matplotlib BEFORE importing app.
# Use importlib so the module is loaded as 'decision_app' and does not
# collide with other 'app' modules when the full test suite runs together.
# ---------------------------------------------------------------------------
_st = MagicMock()
_st.session_state = MagicMock()
_st.session_state.ai_detected = []
sys.modules["streamlit"] = _st

_mpl = MagicMock()
sys.modules["matplotlib"] = _mpl
sys.modules["matplotlib.pyplot"] = MagicMock()

_app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
_spec = importlib.util.spec_from_file_location("decision_app", _app_path)
app = importlib.util.module_from_spec(_spec)
sys.modules["decision_app"] = app
_spec.loader.exec_module(app)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_conn() -> sqlite3.Connection:
    """Fresh in-memory SQLite DB with the app schema."""
    conn = sqlite3.connect(":memory:")
    app._create_schema(conn)
    return conn


def insert_sample(conn: sqlite3.Connection, n: int = 5) -> None:
    """Insert n sample decisions with predictable data."""
    departments = ["Engineering", "Sales", "HR"]
    impacts = ["Low", "Medium", "High"]
    for i in range(n):
        app.save_decision(conn, {
            "who": f"Person {i % 3}",
            "what": f"Decision {i}",
            "why": f"Rationale {i}",
            "department": departments[i % len(departments)],
            "impact": impacts[i % len(impacts)],
            "decided_at": datetime.datetime.now().isoformat(),
            "source": "manual",
        })


# ===========================================================================
# validate_decision
# ===========================================================================

class TestValidateDecision:

    def test_valid_record_passes(self):
        ok, errs = app.validate_decision(
            {"who": "Alice", "what": "Buy server", "why": "Need capacity"}
        )
        assert ok is True
        assert errs == []

    def test_missing_who_fails(self):
        ok, errs = app.validate_decision({"who": "", "what": "X", "why": "Y"})
        assert ok is False
        assert any("maker" in e.lower() for e in errs)

    def test_whitespace_who_fails(self):
        ok, errs = app.validate_decision({"who": "   ", "what": "X", "why": "Y"})
        assert ok is False

    def test_missing_what_fails(self):
        ok, errs = app.validate_decision({"who": "Alice", "what": "", "why": "Y"})
        assert ok is False
        assert any("description" in e.lower() for e in errs)

    def test_missing_why_fails(self):
        ok, errs = app.validate_decision({"who": "Alice", "what": "X", "why": "  "})
        assert ok is False
        assert any("rationale" in e.lower() for e in errs)

    def test_all_missing_returns_three_errors(self):
        _, errs = app.validate_decision({})
        assert len(errs) == 3

    def test_returns_tuple(self):
        result = app.validate_decision({"who": "A", "what": "B", "why": "C"})
        assert isinstance(result, tuple)
        assert len(result) == 2


# ===========================================================================
# save_decision & load_decisions
# ===========================================================================

class TestSaveAndLoad:

    def test_save_returns_integer_id(self):
        conn = make_conn()
        row_id = app.save_decision(conn, {"who": "A", "what": "B", "why": "C"})
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_ids_are_sequential(self):
        conn = make_conn()
        id1 = app.save_decision(conn, {"who": "A", "what": "B", "why": "C"})
        id2 = app.save_decision(conn, {"who": "A", "what": "B", "why": "C"})
        assert id2 == id1 + 1

    def test_load_empty_db_returns_empty_dataframe(self):
        conn = make_conn()
        df = app.load_decisions(conn)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_load_returns_correct_row_count(self):
        conn = make_conn()
        insert_sample(conn, 4)
        df = app.load_decisions(conn)
        assert len(df) == 4

    def test_required_columns_present(self):
        conn = make_conn()
        insert_sample(conn, 1)
        df = app.load_decisions(conn)
        expected = {"id", "who", "what", "why", "department", "impact", "decided_at", "source"}
        assert expected.issubset(set(df.columns))

    def test_default_source_is_manual(self):
        conn = make_conn()
        app.save_decision(conn, {"who": "A", "what": "B", "why": "C"})
        df = app.load_decisions(conn)
        assert df.iloc[0]["source"] == "manual"

    def test_custom_source_persisted(self):
        conn = make_conn()
        app.save_decision(conn, {"who": "A", "what": "B", "why": "C", "source": "ai_monitor"})
        df = app.load_decisions(conn)
        assert df.iloc[0]["source"] == "ai_monitor"

    def test_who_is_stripped(self):
        conn = make_conn()
        app.save_decision(conn, {"who": "  Alice  ", "what": "B", "why": "C"})
        df = app.load_decisions(conn)
        assert df.iloc[0]["who"] == "Alice"


# ===========================================================================
# analyze_decisions
# ===========================================================================

class TestAnalyzeDecisions:

    def _empty_df(self):
        return pd.DataFrame(
            columns=["id", "who", "what", "why", "department", "impact", "decided_at", "source"]
        )

    def test_empty_df_total_is_zero(self):
        result = app.analyze_decisions(self._empty_df())
        assert result["total"] == 0

    def test_empty_df_insights_is_list(self):
        result = app.analyze_decisions(self._empty_df())
        assert isinstance(result["insights"], list)
        assert len(result["insights"]) >= 1

    def test_total_matches_df_length(self):
        conn = make_conn()
        insert_sample(conn, 4)
        df = app.load_decisions(conn)
        assert app.analyze_decisions(df)["total"] == 4

    def test_top_decision_maker_is_most_frequent(self):
        conn = make_conn()
        for name in ["Alice", "Alice", "Alice", "Bob", "Charlie"]:
            app.save_decision(conn, {"who": name, "what": "X", "why": "Y"})
        df = app.load_decisions(conn)
        assert app.analyze_decisions(df)["top_decision_maker"] == "Alice"

    def test_high_impact_count_correct(self):
        conn = make_conn()
        for impact in ["High", "High", "Low", "Medium", "High"]:
            app.save_decision(conn, {"who": "A", "what": "X", "why": "Y", "impact": impact})
        df = app.load_decisions(conn)
        assert app.analyze_decisions(df)["high_impact_count"] == 3

    def test_required_keys_present(self):
        conn = make_conn()
        insert_sample(conn, 2)
        df = app.load_decisions(conn)
        result = app.analyze_decisions(df)
        expected = {"total", "top_decision_maker", "top_department", "high_impact_count", "insights"}
        assert expected.issubset(result.keys())

    def test_insights_is_list_of_strings(self):
        conn = make_conn()
        insert_sample(conn, 3)
        df = app.load_decisions(conn)
        result = app.analyze_decisions(df)
        assert all(isinstance(s, str) for s in result["insights"])

    def test_top_department_detected(self):
        conn = make_conn()
        for dept in ["Engineering", "Engineering", "Engineering", "Sales"]:
            app.save_decision(conn, {"who": "A", "what": "X", "why": "Y", "department": dept})
        df = app.load_decisions(conn)
        assert app.analyze_decisions(df)["top_department"] == "Engineering"


# ===========================================================================
# get_department_summary
# ===========================================================================

class TestGetDepartmentSummary:

    def _empty_df(self):
        return pd.DataFrame(
            columns=["id", "who", "what", "why", "department", "impact", "decided_at", "source"]
        )

    def test_empty_df_returns_empty_dataframe(self):
        result = app.get_department_summary(self._empty_df())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_correct_columns_returned(self):
        conn = make_conn()
        insert_sample(conn, 3)
        df = app.load_decisions(conn)
        summary = app.get_department_summary(df)
        for col in ["Department", "Total", "High", "Medium", "Low"]:
            assert col in summary.columns

    def test_totals_sum_to_df_length(self):
        conn = make_conn()
        insert_sample(conn, 6)
        df = app.load_decisions(conn)
        summary = app.get_department_summary(df)
        assert summary["Total"].sum() == len(df)

    def test_engineering_count_is_correct(self):
        conn = make_conn()
        for dept in ["Engineering", "Engineering", "Sales"]:
            app.save_decision(conn, {
                "who": "A", "what": "X", "why": "Y",
                "department": dept, "impact": "Low",
            })
        df = app.load_decisions(conn)
        summary = app.get_department_summary(df)
        eng_row = summary[summary["Department"] == "Engineering"]
        assert int(eng_row["Total"].iloc[0]) == 2

    def test_sorted_descending_by_total(self):
        conn = make_conn()
        for dept, n in [("Engineering", 5), ("Sales", 2), ("HR", 1)]:
            for _ in range(n):
                app.save_decision(conn, {
                    "who": "A", "what": "X", "why": "Y",
                    "department": dept, "impact": "Low",
                })
        df = app.load_decisions(conn)
        summary = app.get_department_summary(df)
        totals = summary["Total"].tolist()
        assert totals == sorted(totals, reverse=True)


# ===========================================================================
# simulate_ai_detection
# ===========================================================================

class TestSimulateAiDetection:

    def test_returns_list(self):
        assert isinstance(app.simulate_ai_detection(n=2), list)

    def test_correct_count_two(self):
        assert len(app.simulate_ai_detection(n=2)) == 2

    def test_n_one_returns_one_item(self):
        assert len(app.simulate_ai_detection(n=1)) == 1

    def test_required_keys_present(self):
        result = app.simulate_ai_detection(n=1)
        for key in ["who", "what", "why", "decided_at", "source"]:
            assert key in result[0], f"Missing key: {key}"

    def test_source_is_ai_monitor(self):
        result = app.simulate_ai_detection(n=1)
        assert result[0]["source"] == "ai_monitor"

    def test_n_zero_returns_empty_list(self):
        assert app.simulate_ai_detection(n=0) == []

    def test_n_exceeds_pool_capped_at_pool_size(self):
        pool_size = len(app.AI_SIMULATED_DECISIONS)
        result = app.simulate_ai_detection(n=1000)
        assert len(result) <= pool_size

    def test_does_not_mutate_pool(self):
        original_who = [d["who"] for d in app.AI_SIMULATED_DECISIONS]
        app.simulate_ai_detection(n=len(app.AI_SIMULATED_DECISIONS))
        after_who = [d["who"] for d in app.AI_SIMULATED_DECISIONS]
        assert original_who == after_who

    def test_decided_at_is_string(self):
        result = app.simulate_ai_detection(n=1)
        assert isinstance(result[0]["decided_at"], str)


# ===========================================================================
# generate_pdf_report
# ===========================================================================

class TestGeneratePdfReport:

    def _empty_df(self):
        return pd.DataFrame(
            columns=["id", "who", "what", "why", "department", "impact", "decided_at", "source"]
        )

    def test_returns_none_when_reportlab_unavailable(self, monkeypatch):
        monkeypatch.setattr(app, "REPORTLAB_AVAILABLE", False)
        result = app.generate_pdf_report(self._empty_df(), {"total": 0, "insights": []})
        assert result is None

    def test_returns_bytes_with_data(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        conn = make_conn()
        insert_sample(conn, 3)
        df = app.load_decisions(conn)
        analysis = app.analyze_decisions(df)
        pdf = app.generate_pdf_report(df, analysis)
        assert isinstance(pdf, bytes)
        assert len(pdf) > 100

    def test_pdf_starts_with_pdf_magic_bytes(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        conn = make_conn()
        insert_sample(conn, 2)
        df = app.load_decisions(conn)
        pdf = app.generate_pdf_report(df, app.analyze_decisions(df))
        assert pdf[:4] == b"%PDF"

    def test_empty_df_still_generates_pdf(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        df = self._empty_df()
        analysis = app.analyze_decisions(df)
        pdf = app.generate_pdf_report(df, analysis)
        assert pdf is not None
        assert pdf[:4] == b"%PDF"

    def test_analysis_with_all_keys(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        conn = make_conn()
        insert_sample(conn, 5)
        df = app.load_decisions(conn)
        analysis = {
            "total": 5,
            "top_decision_maker": "Person 0",
            "top_department": "Engineering",
            "high_impact_count": 2,
            "insights": ["Insight one.", "Insight two."],
        }
        pdf = app.generate_pdf_report(df, analysis)
        assert pdf[:4] == b"%PDF"

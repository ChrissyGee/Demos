"""
Unit tests for PayrollTimeEntry/app.py.

All pure-logic functions are tested in isolation.
Streamlit is mocked before import so module-level st calls are safe.
"""

import os
import sys
import importlib.util
import datetime
import pytest
import pandas as pd
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock streamlit BEFORE importing app.
# Use importlib to load by file path so the module name 'payroll_app' is
# unique and does not collide with other 'app' modules in the test suite.
# ---------------------------------------------------------------------------
_st = MagicMock()
_st.session_state = MagicMock()
sys.modules["streamlit"] = _st

_app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
_spec = importlib.util.spec_from_file_location("payroll_app", _app_path)
app = importlib.util.module_from_spec(_spec)
sys.modules["payroll_app"] = app
_spec.loader.exec_module(app)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_employee(
    name="Alice",
    emp_id="EMP001",
    allowed_hours=160.0,
    rate=25.0,
    role="Engineer",
) -> dict:
    return {
        "name": name,
        "employee_id": emp_id,
        "allowed_hours_per_month": allowed_hours,
        "pay_rate": rate,
        "role": role,
    }


def make_entry(
    emp_id="EMP001",
    name="Alice",
    hours_dict: dict = None,
    week_label="2026-05-20",
    month="2026-05",
    status="pending",
) -> dict:
    """Build a minimal time-entry record."""
    if hours_dict is None:
        hours_dict = {f"Day{i}": 8.0 for i in range(5)}  # 40h total
    return {
        "employee_id": emp_id,
        "employee_name": name,
        "week_label": week_label,
        "month": month,
        "hours": hours_dict,
        "notes": "",
        "status": status,
        "submitted_at": datetime.datetime.now().isoformat(),
    }


# ===========================================================================
# validate_employee
# ===========================================================================

class TestValidateEmployee:

    def test_valid_employee_passes(self):
        ok, errs = app.validate_employee(make_employee())
        assert ok is True
        assert errs == []

    def test_missing_name_fails(self):
        ok, errs = app.validate_employee({**make_employee(), "name": ""})
        assert ok is False
        assert any("name" in e.lower() for e in errs)

    def test_whitespace_name_fails(self):
        ok, errs = app.validate_employee({**make_employee(), "name": "   "})
        assert ok is False

    def test_missing_employee_id_fails(self):
        ok, errs = app.validate_employee({**make_employee(), "employee_id": ""})
        assert ok is False
        assert any("id" in e.lower() for e in errs)

    def test_whitespace_employee_id_fails(self):
        ok, errs = app.validate_employee({**make_employee(), "employee_id": "  "})
        assert ok is False

    def test_zero_allowed_hours_fails(self):
        ok, errs = app.validate_employee({**make_employee(), "allowed_hours_per_month": 0})
        assert ok is False
        assert any("hour" in e.lower() for e in errs)

    def test_negative_allowed_hours_fails(self):
        ok, errs = app.validate_employee({**make_employee(), "allowed_hours_per_month": -10})
        assert ok is False

    def test_non_numeric_hours_fails(self):
        ok, errs = app.validate_employee({**make_employee(), "allowed_hours_per_month": "lots"})
        assert ok is False
        assert any("number" in e.lower() for e in errs)

    def test_returns_tuple(self):
        result = app.validate_employee(make_employee())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_record_has_multiple_errors(self):
        ok, errs = app.validate_employee({})
        assert ok is False
        assert len(errs) >= 2


# ===========================================================================
# validate_time_entry
# ===========================================================================

class TestValidateTimeEntry:

    def test_valid_entry_passes(self):
        entry = make_entry(hours_dict={"Mon": 8.0, "Tue": 8.0, "Wed": 8.0, "Thu": 8.0, "Fri": 8.0})
        ok, errs = app.validate_time_entry(entry, allowed_hours_per_week=40.0)
        assert ok is True
        assert errs == []

    def test_zero_hours_fails(self):
        entry = make_entry(hours_dict={"Mon": 0.0, "Tue": 0.0})
        ok, errs = app.validate_time_entry(entry, allowed_hours_per_week=40.0)
        assert ok is False
        assert any("zero" in e.lower() or "greater" in e.lower() for e in errs)

    def test_exceeds_ceiling_fails(self):
        # 50h/day × 8 days = 400h; ceiling = 40 * 7 = 280h
        entry = make_entry(hours_dict={f"D{i}": 50.0 for i in range(8)})
        ok, errs = app.validate_time_entry(entry, allowed_hours_per_week=40.0)
        assert ok is False
        assert any("exceed" in e.lower() or "ceiling" in e.lower() for e in errs)

    def test_negative_hours_fails(self):
        entry = make_entry(hours_dict={"Mon": -2.0, "Tue": 8.0})
        ok, errs = app.validate_time_entry(entry, allowed_hours_per_week=40.0)
        assert ok is False
        assert any("negative" in e.lower() for e in errs)

    def test_over_24h_per_day_fails(self):
        entry = make_entry(hours_dict={"Mon": 25.0})
        ok, errs = app.validate_time_entry(entry, allowed_hours_per_week=40.0)
        assert ok is False
        assert any("24" in e or "exceed" in e.lower() for e in errs)

    def test_non_dict_hours_fails(self):
        entry = {**make_entry(), "hours": None}
        ok, errs = app.validate_time_entry(entry)
        assert ok is False

    def test_returns_tuple(self):
        result = app.validate_time_entry(make_entry())
        assert isinstance(result, tuple)
        assert len(result) == 2


# ===========================================================================
# get_total_hours
# ===========================================================================

class TestGetTotalHours:

    def test_sums_correctly(self):
        entry = make_entry(hours_dict={"Mon": 8.0, "Tue": 7.5, "Wed": 6.0})
        assert app.get_total_hours(entry) == 21.5

    def test_empty_hours_returns_zero(self):
        assert app.get_total_hours({"hours": {}}) == 0.0

    def test_missing_hours_key_returns_zero(self):
        assert app.get_total_hours({}) == 0.0

    def test_none_hours_returns_zero(self):
        assert app.get_total_hours({"hours": None}) == 0.0

    def test_single_day(self):
        assert app.get_total_hours({"hours": {"Mon": 8.0}}) == 8.0

    def test_returns_float(self):
        result = app.get_total_hours({"hours": {"Mon": 8}})
        assert isinstance(result, float)


# ===========================================================================
# calculate_pay
# ===========================================================================

class TestCalculatePay:

    def test_basic_calculation(self):
        assert app.calculate_pay(40.0, 25.0) == 1000.0

    def test_zero_hours_returns_zero(self):
        assert app.calculate_pay(0.0, 25.0) == 0.0

    def test_zero_rate_returns_zero(self):
        assert app.calculate_pay(40.0, 0.0) == 0.0

    def test_negative_hours_returns_zero(self):
        assert app.calculate_pay(-5.0, 25.0) == 0.0

    def test_negative_rate_returns_zero(self):
        assert app.calculate_pay(40.0, -10.0) == 0.0

    def test_decimal_hours(self):
        assert app.calculate_pay(0.5, 20.0) == 10.0

    def test_result_is_rounded_to_cents(self):
        result = app.calculate_pay(7.5, 15.333)
        assert result == round(7.5 * 15.333, 2)

    def test_large_values(self):
        result = app.calculate_pay(744.0, 999.99)
        assert result == round(744.0 * 999.99, 2)


# ===========================================================================
# ai_validate_time_entry
# ===========================================================================

class TestAiValidateTimeEntry:

    def test_ok_within_limits(self):
        emp = make_employee(allowed_hours=160.0)
        entry = make_entry(hours_dict={f"D{i}": 8.0 for i in range(5)})  # 40h
        result = app.ai_validate_time_entry(entry, emp)
        assert result["status"] == "ok"

    def test_reject_when_exceeds_monthly_limit(self):
        emp = make_employee(allowed_hours=40.0)
        entry = make_entry(hours_dict={f"D{i}": 8.0 for i in range(7)})  # 56h > 40h
        result = app.ai_validate_time_entry(entry, emp)
        assert result["status"] == "reject"

    def test_warn_when_close_to_limit(self):
        emp = make_employee(allowed_hours=40.0)
        # 37h / 40h = 92.5% → above 90% threshold → warn
        entry = make_entry(hours_dict={"Mon": 37.0})
        result = app.ai_validate_time_entry(entry, emp)
        assert result["status"] == "warn"

    def test_reject_zero_hours(self):
        emp = make_employee(allowed_hours=160.0)
        entry = make_entry(hours_dict={"Mon": 0.0})
        result = app.ai_validate_time_entry(entry, emp)
        assert result["status"] == "reject"

    def test_returns_required_keys(self):
        emp = make_employee()
        entry = make_entry()
        result = app.ai_validate_time_entry(entry, emp)
        assert {"employee_id", "status", "message"}.issubset(result.keys())

    def test_employee_id_matches_entry(self):
        emp = make_employee(emp_id="EMP099")
        entry = make_entry(emp_id="EMP099")
        result = app.ai_validate_time_entry(entry, emp)
        assert result["employee_id"] == "EMP099"

    def test_message_is_non_empty_string(self):
        emp = make_employee()
        entry = make_entry()
        result = app.ai_validate_time_entry(entry, emp)
        assert isinstance(result["message"], str)
        assert len(result["message"]) > 0

    def test_exactly_at_limit_is_ok_not_reject(self):
        emp = make_employee(allowed_hours=40.0)
        # Exactly 40h = 100% of 40h limit → reject (exceeds allowed)
        entry = make_entry(hours_dict={"Mon": 40.0})
        result = app.ai_validate_time_entry(entry, emp)
        # 40h == 40h allowed, not > allowed → should NOT reject
        assert result["status"] in ("ok", "warn")


# ===========================================================================
# get_employee_by_id
# ===========================================================================

class TestGetEmployeeById:

    def test_finds_correct_employee(self):
        employees = [make_employee("Alice", "EMP001"), make_employee("Bob", "EMP002")]
        result = app.get_employee_by_id(employees, "EMP002")
        assert result is not None
        assert result["name"] == "Bob"

    def test_returns_none_for_unknown_id(self):
        employees = [make_employee("Alice", "EMP001")]
        assert app.get_employee_by_id(employees, "EMP999") is None

    def test_empty_list_returns_none(self):
        assert app.get_employee_by_id([], "EMP001") is None

    def test_returns_first_match(self):
        # If duplicates exist, return the first one encountered
        employees = [make_employee("Alice", "EMP001"), make_employee("Alice2", "EMP001")]
        result = app.get_employee_by_id(employees, "EMP001")
        assert result["name"] == "Alice"


# ===========================================================================
# get_entries_for_employee
# ===========================================================================

class TestGetEntriesForEmployee:

    def test_returns_matching_entries(self):
        entries = [
            make_entry(emp_id="EMP001"),
            make_entry(emp_id="EMP002"),
            make_entry(emp_id="EMP001"),
        ]
        result = app.get_entries_for_employee(entries, "EMP001")
        assert len(result) == 2

    def test_returns_empty_list_for_no_match(self):
        entries = [make_entry(emp_id="EMP002")]
        assert app.get_entries_for_employee(entries, "EMP001") == []

    def test_empty_entries_returns_empty(self):
        assert app.get_entries_for_employee([], "EMP001") == []


# ===========================================================================
# summarise_employee_hours
# ===========================================================================

class TestSummariseEmployeeHours:

    def test_empty_input_returns_dataframe(self):
        result = app.summarise_employee_hours([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_correct_columns(self):
        result = app.summarise_employee_hours([make_entry()])
        for col in ["employee_id", "name", "week", "total_hours", "status", "month"]:
            assert col in result.columns

    def test_row_count_matches_entries(self):
        entries = [
            make_entry(week_label="2026-05-01"),
            make_entry(week_label="2026-05-08"),
            make_entry(week_label="2026-05-15"),
        ]
        result = app.summarise_employee_hours(entries)
        assert len(result) == 3

    def test_total_hours_calculated(self):
        entry = make_entry(hours_dict={"Mon": 8.0, "Tue": 8.0, "Wed": 8.0, "Thu": 8.0, "Fri": 8.0})
        result = app.summarise_employee_hours([entry])
        assert result.iloc[0]["total_hours"] == 40.0

    def test_status_preserved(self):
        entry = make_entry(status="approved")
        result = app.summarise_employee_hours([entry])
        assert result.iloc[0]["status"] == "approved"


# ===========================================================================
# build_chat_response
# ===========================================================================

class TestBuildChatResponse:

    def test_no_employees_returns_guidance(self):
        result = app.build_chat_response("how many employees?", [], [])
        assert "no employees" in result.lower() or "not" in result.lower()

    def test_employee_count_query(self):
        employees = [make_employee("Alice", "EMP001"), make_employee("Bob", "EMP002")]
        result = app.build_chat_response("how many employees are there?", employees, [])
        assert "2" in result

    def test_pending_count_query(self):
        employees = [make_employee()]
        entries = [
            make_entry(status="pending"),
            make_entry(status="pending"),
            make_entry(status="approved"),
        ]
        result = app.build_chat_response("how many pending approvals?", employees, entries)
        assert "2" in result

    def test_approved_count_query(self):
        employees = [make_employee()]
        entries = [make_entry(status="approved"), make_entry(status="pending")]
        result = app.build_chat_response("how many approved?", employees, entries)
        assert "1" in result

    def test_top_performer_query(self):
        employees = [make_employee("Alice", "EMP001"), make_employee("Bob", "EMP002")]
        entries = [
            make_entry(emp_id="EMP001", hours_dict={"Mon": 8.0, "Tue": 8.0}),  # 16h
            make_entry(emp_id="EMP002", hours_dict={f"D{i}": 8.0 for i in range(7)}),  # 56h
        ]
        result = app.build_chat_response("who has the most hours?", employees, entries)
        assert "Bob" in result or "EMP002" in result

    def test_fallback_response_contains_hint(self):
        employees = [make_employee()]
        result = app.build_chat_response("what is the weather like?", employees, [])
        assert "try" in result.lower() or "ask" in result.lower() or "average" in result.lower()

    def test_returns_string(self):
        result = app.build_chat_response("any question", [make_employee()], [])
        assert isinstance(result, str)


# ===========================================================================
# generate_payroll_report
# ===========================================================================

class TestGeneratePayrollReport:

    def test_returns_none_when_reportlab_unavailable(self, monkeypatch):
        monkeypatch.setattr(app, "REPORTLAB_AVAILABLE", False)
        result = app.generate_payroll_report([], [])
        assert result is None

    def test_returns_bytes_with_data(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        employees = [make_employee()]
        entries = [make_entry(month="2026-05")]
        pdf = app.generate_payroll_report(employees, entries, "2026-05")
        assert isinstance(pdf, bytes)
        assert len(pdf) > 100

    def test_pdf_magic_bytes(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        pdf = app.generate_payroll_report(
            [make_employee()], [make_entry(month="2026-05")], "2026-05"
        )
        assert pdf[:4] == b"%PDF"

    def test_empty_inputs_still_generates(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        pdf = app.generate_payroll_report([], [], "2026-05")
        assert pdf is not None
        assert pdf[:4] == b"%PDF"

    def test_multiple_employees(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        employees = [
            make_employee("Alice", "EMP001"),
            make_employee("Bob", "EMP002"),
        ]
        entries = [
            make_entry("EMP001", month="2026-05"),
            make_entry("EMP002", month="2026-05"),
        ]
        pdf = app.generate_payroll_report(employees, entries, "2026-05")
        assert pdf[:4] == b"%PDF"


# ===========================================================================
# generate_pay_stub
# ===========================================================================

class TestGeneratePayStub:

    def test_returns_none_when_reportlab_unavailable(self, monkeypatch):
        monkeypatch.setattr(app, "REPORTLAB_AVAILABLE", False)
        result = app.generate_pay_stub(make_employee(), [], "2026-05")
        assert result is None

    def test_returns_bytes_with_entries(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        emp = make_employee()
        entries = [make_entry(month="2026-05")]
        pdf = app.generate_pay_stub(emp, entries, "2026-05")
        assert isinstance(pdf, bytes)
        assert len(pdf) > 100

    def test_pdf_magic_bytes(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        emp = make_employee()
        entries = [make_entry(month="2026-05")]
        pdf = app.generate_pay_stub(emp, entries, "2026-05")
        assert pdf[:4] == b"%PDF"

    def test_no_entries_for_month_still_generates(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        emp = make_employee()
        # Entries are for a different month
        entries = [make_entry(month="2025-01")]
        pdf = app.generate_pay_stub(emp, entries, "2026-05")
        assert pdf is not None
        assert pdf[:4] == b"%PDF"

    def test_empty_entries_list(self):
        if not app.REPORTLAB_AVAILABLE:
            pytest.skip("reportlab not installed")
        pdf = app.generate_pay_stub(make_employee(), [], "2026-05")
        assert pdf is not None
        assert pdf[:4] == b"%PDF"

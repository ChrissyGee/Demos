"""
Unit tests for ClaudeExcelPlugIn/app.py.

The app is a pure UI/content guide — no anentra_lib usage, no business logic
beyond data constants and two helper functions. Tests therefore focus on:
  1. Data constant structure and correctness (schema + key content assertions)
  2. init_session_state — verifies keys, defaults, and idempotency
  3. render_table helper — DataFrame creation from the constant lists

Streamlit is mocked before import so module-level st calls are safe.
"""

import os
import sys
import importlib.util
import pandas as pd
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock streamlit before importing app.
# Use importlib so the module is loaded as 'excel_app' and does not collide
# with other 'app' modules when the full test suite runs together.
# ---------------------------------------------------------------------------
_st = MagicMock()
_st.session_state = MagicMock()
sys.modules["streamlit"] = _st

_app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
_spec = importlib.util.spec_from_file_location("excel_app", _app_path)
app = importlib.util.module_from_spec(_spec)
sys.modules["excel_app"] = app
_spec.loader.exec_module(app)


# ---------------------------------------------------------------------------
# SUPPORTED_VERSIONS
# ---------------------------------------------------------------------------

class TestSupportedVersions:

    def test_has_five_platforms(self):
        assert len(app.SUPPORTED_VERSIONS) == 5

    def test_all_rows_have_required_keys(self):
        for row in app.SUPPORTED_VERSIONS:
            assert set(row.keys()) == {"Platform", "Minimum Version", "Notes"}

    def test_excel_web_is_present(self):
        platforms = [r["Platform"] for r in app.SUPPORTED_VERSIONS]
        assert any("Web" in p for p in platforms)

    def test_phone_is_marked_not_supported(self):
        phone = next(
            (r for r in app.SUPPORTED_VERSIONS if "phone" in r["Platform"].lower()),
            None,
        )
        assert phone is not None, "Phone platform entry not found"
        assert "not supported" in phone["Minimum Version"].lower()

    def test_windows_entry_references_microsoft_365(self):
        windows = next(
            (r for r in app.SUPPORTED_VERSIONS if "Windows" in r["Platform"]),
            None,
        )
        assert windows is not None
        assert "Microsoft 365" in windows["Minimum Version"]

    def test_all_notes_are_nonempty(self):
        for row in app.SUPPORTED_VERSIONS:
            assert row["Notes"].strip()


# ---------------------------------------------------------------------------
# CLAUDE_PLANS
# ---------------------------------------------------------------------------

class TestClaudePlans:

    def test_has_five_plans(self):
        assert len(app.CLAUDE_PLANS) == 5

    def test_all_rows_have_required_keys(self):
        for row in app.CLAUDE_PLANS:
            assert set(row.keys()) == {"Plan", "Excel Add-in Access", "Best For"}

    def test_free_plan_has_no_add_in_access(self):
        free = next(p for p in app.CLAUDE_PLANS if "Free" in p["Plan"])
        assert free["Excel Add-in Access"] == "No"

    def test_pro_plan_has_add_in_access(self):
        pro = next(p for p in app.CLAUDE_PLANS if p["Plan"] == "Claude Pro")
        assert pro["Excel Add-in Access"] == "Yes"

    def test_enterprise_plan_references_sso(self):
        ent = next(p for p in app.CLAUDE_PLANS if "Enterprise" in p["Plan"])
        assert "SSO" in ent["Excel Add-in Access"] or "SSO" in ent["Best For"]

    def test_all_best_for_are_nonempty(self):
        for row in app.CLAUDE_PLANS:
            assert row["Best For"].strip()

    def test_plan_names_are_unique(self):
        names = [p["Plan"] for p in app.CLAUDE_PLANS]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# MODELS_AVAILABLE
# ---------------------------------------------------------------------------

class TestModelsAvailable:

    def test_all_rows_have_required_keys(self):
        for row in app.MODELS_AVAILABLE:
            assert {"Model", "Best For"}.issubset(set(row.keys()))

    def test_sonnet_is_present(self):
        assert any("Sonnet" in r["Model"] for r in app.MODELS_AVAILABLE)

    def test_opus_is_present(self):
        assert any("Opus" in r["Model"] for r in app.MODELS_AVAILABLE)

    def test_haiku_is_present(self):
        assert any("Haiku" in r["Model"] for r in app.MODELS_AVAILABLE)

    def test_all_best_for_are_nonempty(self):
        for row in app.MODELS_AVAILABLE:
            assert row["Best For"].strip()

    def test_model_names_are_unique(self):
        names = [r["Model"] for r in app.MODELS_AVAILABLE]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# INDIVIDUAL_CHECKLIST & ADMIN_CHECKLIST
# ---------------------------------------------------------------------------

class TestChecklists:

    def test_individual_checklist_has_unique_keys(self):
        keys = [item["key"] for item in app.INDIVIDUAL_CHECKLIST]
        assert len(keys) == len(set(keys))

    def test_admin_checklist_has_unique_keys(self):
        keys = [item["key"] for item in app.ADMIN_CHECKLIST]
        assert len(keys) == len(set(keys))

    def test_no_shared_keys_between_checklists(self):
        ind_keys = {item["key"] for item in app.INDIVIDUAL_CHECKLIST}
        adm_keys = {item["key"] for item in app.ADMIN_CHECKLIST}
        assert ind_keys.isdisjoint(adm_keys)

    def test_individual_items_have_nonempty_labels(self):
        for item in app.INDIVIDUAL_CHECKLIST:
            assert item["label"].strip()

    def test_admin_items_have_nonempty_labels(self):
        for item in app.ADMIN_CHECKLIST:
            assert item["label"].strip()

    def test_individual_keys_prefixed_ind(self):
        for item in app.INDIVIDUAL_CHECKLIST:
            assert item["key"].startswith("ind_")

    def test_admin_keys_prefixed_adm(self):
        for item in app.ADMIN_CHECKLIST:
            assert item["key"].startswith("adm_")

    def test_checklist_items_are_dicts_with_key_and_label(self):
        for item in app.INDIVIDUAL_CHECKLIST + app.ADMIN_CHECKLIST:
            assert isinstance(item, dict)
            assert "key" in item and "label" in item


# ---------------------------------------------------------------------------
# RECOMMENDED_VIDEOS & VIDEO_SUMMARY_POINTS
# ---------------------------------------------------------------------------

class TestVideoContent:

    def test_recommended_videos_have_required_keys(self):
        for v in app.RECOMMENDED_VIDEOS:
            assert {"title", "url", "covers"}.issubset(set(v.keys()))

    def test_recommended_video_urls_are_nonempty(self):
        for v in app.RECOMMENDED_VIDEOS:
            assert v["url"].strip()

    def test_recommended_video_titles_are_nonempty(self):
        for v in app.RECOMMENDED_VIDEOS:
            assert v["title"].strip()

    def test_video_summary_points_nonempty_list(self):
        assert len(app.VIDEO_SUMMARY_POINTS) > 0

    def test_all_video_summary_points_are_strings(self):
        for pt in app.VIDEO_SUMMARY_POINTS:
            assert isinstance(pt, str) and pt.strip()

    def test_video_summary_covers_individual_install(self):
        combined = " ".join(app.VIDEO_SUMMARY_POINTS).lower()
        assert "add-in" in combined or "install" in combined

    def test_video_summary_mentions_admin_path(self):
        combined = " ".join(app.VIDEO_SUMMARY_POINTS).lower()
        assert "admin" in combined or "365" in combined


# ---------------------------------------------------------------------------
# init_session_state
# ---------------------------------------------------------------------------

class TestInitSessionState:

    def test_all_checklist_keys_initialized_to_false(self):
        _st.session_state = {}
        app.init_session_state()
        for item in app.INDIVIDUAL_CHECKLIST + app.ADMIN_CHECKLIST:
            assert item["key"] in _st.session_state
            assert _st.session_state[item["key"]] is False

    def test_checklist_initialised_flag_set_to_true(self):
        _st.session_state = {}
        app.init_session_state()
        assert _st.session_state.get("checklist_initialised") is True

    def test_second_call_is_idempotent(self):
        _st.session_state = {}
        app.init_session_state()
        key = app.INDIVIDUAL_CHECKLIST[0]["key"]
        _st.session_state[key] = True  # simulate user ticking a box
        app.init_session_state()  # should be a no-op
        assert _st.session_state[key] is True

    def test_covers_all_individual_and_admin_keys(self):
        _st.session_state = {}
        app.init_session_state()
        all_keys = {item["key"] for item in app.INDIVIDUAL_CHECKLIST + app.ADMIN_CHECKLIST}
        assert all_keys.issubset(set(_st.session_state.keys()))


# ---------------------------------------------------------------------------
# render_table helper — DataFrame construction logic
# ---------------------------------------------------------------------------

class TestRenderTableLogic:
    """
    render_table() itself calls st.dataframe, which is a Streamlit side-effect.
    These tests verify the pure pd.DataFrame construction that underpins it
    and that each static dataset converts cleanly to a DataFrame.
    """

    def test_simple_rows_to_dataframe(self):
        rows = [{"A": "x", "B": 1}, {"A": "y", "B": 2}]
        df = pd.DataFrame(rows)
        assert list(df.columns) == ["A", "B"]
        assert len(df) == 2

    def test_supported_versions_dataframe_columns(self):
        df = pd.DataFrame(app.SUPPORTED_VERSIONS)
        assert set(df.columns) == {"Platform", "Minimum Version", "Notes"}

    def test_supported_versions_dataframe_row_count(self):
        df = pd.DataFrame(app.SUPPORTED_VERSIONS)
        assert len(df) == 5

    def test_claude_plans_dataframe_columns(self):
        df = pd.DataFrame(app.CLAUDE_PLANS)
        assert set(df.columns) == {"Plan", "Excel Add-in Access", "Best For"}

    def test_models_available_dataframe_columns(self):
        df = pd.DataFrame(app.MODELS_AVAILABLE)
        assert {"Model", "Best For"}.issubset(set(df.columns))

    def test_no_null_values_in_supported_versions(self):
        df = pd.DataFrame(app.SUPPORTED_VERSIONS)
        assert not df.isnull().any().any()

    def test_no_null_values_in_claude_plans(self):
        df = pd.DataFrame(app.CLAUDE_PLANS)
        assert not df.isnull().any().any()

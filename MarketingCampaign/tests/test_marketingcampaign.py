"""
Unit tests for MarketingCampaign/app.py.

All tool functions and step functions are tested through their deterministic
paths (no OpenAI key → llm_complete() always returns None → template fallbacks).

Streamlit is mocked before import. st.session_state.console is a real list so
log() calls succeed without raising AttributeError.
"""

import os
import sys
import pytest
import pandas as pd
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
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_brief(**overrides) -> dict:
    base = {
        "business_name": "Acme Corp",
        "industry": "SaaS / B2B Tech",
        "audience": "Tech founders aged 30-50",
        "goals": ["Drive new customer acquisition", "Increase brand awareness"],
        "budget": 50000.0,
        "timeline_weeks": 12,
        "channels_preferred": ["Meta Ads (FB/IG)", "Google Search Ads", "Email"],
        "extra_notes": "",
        "created_at": "2026-05-23T00:00:00",
    }
    base.update(overrides)
    return base


def _make_campaign(brief: dict) -> dict:
    budget_alloc = app._gen_budget_allocation(brief)
    return {
        "campaign_objective": app.tool_document_filler(
            "Grow {biz} within {weeks} weeks.", {"biz": brief["business_name"], "weeks": str(brief["timeline_weeks"])}
        ),
        "audience_personas": "Persona 1 — Primary buyer content here enough text.",
        "key_messages": "Pillar 1 — outcome Pillar 2 — trust Pillar 3 — urgency",
        "creative_concepts": "Concept A — Functional content. Concept B — Emotional content.",
        "channel_plan": "Channel plan content goes here with enough text.",
        "budget_allocation": budget_alloc,
        "kpis": app._gen_kpis(brief),
        "timeline": app._gen_timeline(brief),
    }


def _make_validation(status="PASS", issues=None) -> dict:
    return {"status": status, "issues": issues or [], "checked_at": "2026-05-23T00:00:00"}


# ---------------------------------------------------------------------------
# tool_web_search
# ---------------------------------------------------------------------------

class TestToolWebSearch:

    def test_returns_list(self):
        result = app.tool_web_search("brand awareness")
        assert isinstance(result, list)

    def test_returns_n_results(self):
        result = app.tool_web_search("customer acquisition", n=2)
        assert len(result) == 2

    def test_each_result_has_required_keys(self):
        for item in app.tool_web_search("saas marketing", n=3):
            assert {"source", "title", "snippet", "url"}.issubset(set(item.keys()))

    def test_deterministic_for_same_query(self):
        r1 = app.tool_web_search("email marketing strategy", n=3)
        r2 = app.tool_web_search("email marketing strategy", n=3)
        assert [i["source"] for i in r1] == [i["source"] for i in r2]

    def test_different_queries_produce_different_source_orders(self):
        r1 = app.tool_web_search("social media ads", n=3)
        r2 = app.tool_web_search("print advertising budget", n=3)
        assert [i["source"] for i in r1] != [i["source"] for i in r2]

    def test_query_appears_in_snippet(self):
        query = "tiktok influencer campaigns"
        results = app.tool_web_search(query, n=1)
        assert query in results[0]["snippet"]

    def test_url_is_nonempty_string(self):
        for item in app.tool_web_search("growth hacking", n=2):
            assert isinstance(item["url"], str) and item["url"].strip()

    def test_n_equals_1_returns_single_item(self):
        result = app.tool_web_search("retention campaigns", n=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# tool_rag_retrieve
# ---------------------------------------------------------------------------

class TestToolRagRetrieve:

    def test_returns_list(self):
        result = app.tool_rag_retrieve("saas b2b marketing")
        assert isinstance(result, list)

    def test_respects_k(self):
        result = app.tool_rag_retrieve("dtc ecommerce fitness", k=2)
        assert len(result) <= 2

    def test_each_hit_has_required_keys(self):
        for doc in app.tool_rag_retrieve("budget allocation", k=3):
            assert {"id", "title", "tags", "text"}.issubset(set(doc.keys()))

    def test_saas_query_retrieves_saas_doc(self):
        result = app.tool_rag_retrieve("saas b2b demand linkedin", k=5)
        ids = [d["id"] for d in result]
        assert "rag-002" in ids

    def test_budget_query_retrieves_budget_doc(self):
        result = app.tool_rag_retrieve("budget allocation spend", k=5)
        ids = [d["id"] for d in result]
        assert "rag-007" in ids

    def test_no_match_falls_back_to_top_corpus(self):
        result = app.tool_rag_retrieve("zzzqqqxyz999unrealterm", k=3)
        assert len(result) == 3
        assert result[0]["id"] == app.RAG_CORPUS[0]["id"]

    def test_dtc_query_retrieves_dtc_doc(self):
        result = app.tool_rag_retrieve("dtc ecommerce channel launch", k=5)
        ids = [d["id"] for d in result]
        assert "rag-001" in ids


# ---------------------------------------------------------------------------
# tool_metrics_calculator
# ---------------------------------------------------------------------------

class TestToolMetricsCalculator:

    def _simple_split(self):
        return {"Meta Ads (FB/IG)": 60.0, "Email": 40.0}

    def test_returns_dataframe(self):
        df = app.tool_metrics_calculator(10000.0, self._simple_split())
        assert isinstance(df, pd.DataFrame)

    def test_has_correct_number_of_rows(self):
        df = app.tool_metrics_calculator(10000.0, self._simple_split())
        assert len(df) == 2

    def test_required_columns_present(self):
        df = app.tool_metrics_calculator(10000.0, self._simple_split())
        expected = {"Channel", "Allocation %", "Spend ($)", "Impressions (est.)",
                    "Clicks (est.)", "Conversions (est.)", "CPA (est. $)"}
        assert expected.issubset(set(df.columns))

    def test_spend_sums_to_budget(self):
        split = {"Meta Ads (FB/IG)": 40.0, "Google Search Ads": 35.0, "Email": 25.0}
        budget = 20000.0
        df = app.tool_metrics_calculator(budget, split)
        assert abs(df["Spend ($)"].sum() - budget) < 0.02

    def test_impressions_are_nonnegative(self):
        df = app.tool_metrics_calculator(5000.0, self._simple_split())
        assert (df["Impressions (est.)"] >= 0).all()

    def test_unknown_channel_uses_default_benchmarks(self):
        split = {"SomeNewChannel XYZ": 100.0}
        df = app.tool_metrics_calculator(10000.0, split)
        assert len(df) == 1
        assert df.iloc[0]["Spend ($)"] == pytest.approx(10000.0)

    def test_zero_percent_allocation_gives_zero_spend(self):
        split = {"Meta Ads (FB/IG)": 0.0, "Email": 100.0}
        df = app.tool_metrics_calculator(10000.0, split)
        meta_row = df[df["Channel"] == "Meta Ads (FB/IG)"].iloc[0]
        assert meta_row["Spend ($)"] == 0.0


# ---------------------------------------------------------------------------
# tool_document_filler
# ---------------------------------------------------------------------------

class TestToolDocumentFiller:

    def test_replaces_single_placeholder(self):
        result = app.tool_document_filler("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_replaces_multiple_placeholders(self):
        result = app.tool_document_filler("{a} and {b}", {"a": "foo", "b": "bar"})
        assert result == "foo and bar"

    def test_unknown_placeholder_left_unchanged(self):
        result = app.tool_document_filler("Hello {unknown}!", {"name": "X"})
        assert "{unknown}" in result

    def test_empty_fields_returns_template_unchanged(self):
        tmpl = "No placeholders here."
        assert app.tool_document_filler(tmpl, {}) == tmpl

    def test_numeric_value_converted_to_string(self):
        result = app.tool_document_filler("Budget: {amount}", {"amount": 50000})
        assert "50000" in result


# ---------------------------------------------------------------------------
# step1_understand_brief
# ---------------------------------------------------------------------------

class TestStep1UnderstandBrief:

    def _make_form(self, **overrides):
        base = {
            "business_name": "  WidgetCo  ",
            "industry": "Retail",
            "audience": "Gen Z shoppers",
            "goals": ["Drive new customer acquisition"],
            "budget": "30000",
            "timeline_weeks": "8",
            "channels_preferred": ["TikTok"],
            "extra_notes": "Focus on mobile.",
        }
        base.update(overrides)
        return base

    def test_returns_dict(self):
        brief = app.step1_understand_brief(self._make_form())
        assert isinstance(brief, dict)

    def test_strips_business_name_whitespace(self):
        brief = app.step1_understand_brief(self._make_form(business_name="  TrimMe  "))
        assert brief["business_name"] == "TrimMe"

    def test_budget_converted_to_float(self):
        brief = app.step1_understand_brief(self._make_form())
        assert isinstance(brief["budget"], float)

    def test_timeline_weeks_converted_to_int(self):
        brief = app.step1_understand_brief(self._make_form())
        assert isinstance(brief["timeline_weeks"], int)

    def test_required_keys_present(self):
        brief = app.step1_understand_brief(self._make_form())
        expected = {"business_name", "industry", "audience", "goals",
                    "budget", "timeline_weeks", "channels_preferred",
                    "extra_notes", "created_at"}
        assert expected.issubset(set(brief.keys()))

    def test_created_at_is_iso_string(self):
        brief = app.step1_understand_brief(self._make_form())
        assert "T" in brief["created_at"]

    def test_goals_preserved_as_list(self):
        goals = ["Drive new customer acquisition", "Increase brand awareness"]
        brief = app.step1_understand_brief(self._make_form(goals=goals))
        assert brief["goals"] == goals

    def test_missing_extra_notes_defaults_to_empty_string(self):
        form = self._make_form()
        del form["extra_notes"]
        brief = app.step1_understand_brief(form)
        assert brief["extra_notes"] == ""


# ---------------------------------------------------------------------------
# step2_research
# ---------------------------------------------------------------------------

class TestStep2Research:

    def test_returns_dict(self):
        brief = _make_brief()
        result = app.step2_research(brief)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        brief = _make_brief()
        result = app.step2_research(brief)
        assert {"web_search_query", "web", "rag_query", "rag"}.issubset(set(result.keys()))

    def test_web_results_is_list_of_three(self):
        brief = _make_brief()
        result = app.step2_research(brief)
        assert len(result["web"]) == 3

    def test_rag_results_is_list(self):
        brief = _make_brief()
        result = app.step2_research(brief)
        assert isinstance(result["rag"], list) and len(result["rag"]) > 0

    def test_web_query_contains_industry(self):
        brief = _make_brief(industry="Health & Wellness")
        result = app.step2_research(brief)
        assert "Health" in result["web_search_query"] or "Wellness" in result["web_search_query"]

    def test_rag_query_contains_industry(self):
        brief = _make_brief(industry="SaaS / B2B Tech")
        result = app.step2_research(brief)
        assert "SaaS" in result["rag_query"] or "B2B" in result["rag_query"]


# ---------------------------------------------------------------------------
# _gen_budget_allocation
# ---------------------------------------------------------------------------

class TestGenBudgetAllocation:

    def test_returns_dict_with_split_and_metrics(self):
        brief = _make_brief()
        result = app._gen_budget_allocation(brief)
        assert {"split", "metrics_df"}.issubset(set(result.keys()))

    def test_split_sums_to_100(self):
        brief = _make_brief()
        split = app._gen_budget_allocation(brief)["split"]
        assert abs(sum(split.values()) - 100.0) <= 1.0

    def test_split_keys_match_preferred_channels(self):
        channels = ["Meta Ads (FB/IG)", "TikTok"]
        brief = _make_brief(channels_preferred=channels)
        split = app._gen_budget_allocation(brief)["split"]
        assert set(split.keys()) == set(channels)

    def test_metrics_df_is_dataframe(self):
        brief = _make_brief()
        result = app._gen_budget_allocation(brief)
        assert isinstance(result["metrics_df"], pd.DataFrame)

    def test_no_preferred_channels_uses_default_three(self):
        brief = _make_brief(channels_preferred=[])
        split = app._gen_budget_allocation(brief)["split"]
        assert len(split) == 3

    def test_single_channel_gets_full_100_percent(self):
        brief = _make_brief(channels_preferred=["LinkedIn"])
        split = app._gen_budget_allocation(brief)["split"]
        assert split["LinkedIn"] == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# _gen_kpis
# ---------------------------------------------------------------------------

class TestGenKpis:

    def test_returns_string(self):
        brief = _make_brief()
        result = app._gen_kpis(brief)
        assert isinstance(result, str)

    def test_acquisition_goal_includes_cac(self):
        brief = _make_brief(goals=["Drive new customer acquisition"])
        result = app._gen_kpis(brief)
        assert "CAC" in result or "customer" in result.lower()

    def test_awareness_goal_includes_reach(self):
        brief = _make_brief(goals=["Increase brand awareness"])
        result = app._gen_kpis(brief)
        assert "Reach" in result

    def test_unknown_goal_falls_back_to_defaults(self):
        brief = _make_brief(goals=["Some completely unknown goal XYZ"])
        result = app._gen_kpis(brief)
        assert "Reach" in result or "Conversion" in result or "Engagement" in result

    def test_multiple_goals_include_kpis_from_each(self):
        brief = _make_brief(goals=["Drive new customer acquisition",
                                   "Grow newsletter / email list"])
        result = app._gen_kpis(brief)
        assert "CAC" in result or "customer" in result.lower()
        assert "subscriber" in result.lower() or "open rate" in result.lower()

    def test_output_contains_primary_kpis_header(self):
        brief = _make_brief()
        result = app._gen_kpis(brief)
        assert "KPI" in result or "kpi" in result.lower() or "-" in result


# ---------------------------------------------------------------------------
# _gen_timeline
# ---------------------------------------------------------------------------

class TestGenTimeline:

    def test_returns_string(self):
        brief = _make_brief()
        result = app._gen_timeline(brief)
        assert isinstance(result, str)

    def test_contains_four_phases(self):
        brief = _make_brief(timeline_weeks=12)
        result = app._gen_timeline(brief)
        assert result.count("**") >= 8  # each phase name is bold → 2 ** markers

    def test_phases_have_date_arrows(self):
        brief = _make_brief(timeline_weeks=8)
        result = app._gen_timeline(brief)
        assert "→" in result

    def test_phase_week_counts_sum_to_timeline_weeks(self):
        brief = _make_brief(timeline_weeks=16)
        result = app._gen_timeline(brief)
        import re
        weeks = [int(m) for m in re.findall(r"(\d+)w ", result)]
        assert sum(weeks) == 16

    def test_short_timeline_still_produces_4_phases(self):
        brief = _make_brief(timeline_weeks=4)
        result = app._gen_timeline(brief)
        assert result.count("- **") == 4


# ---------------------------------------------------------------------------
# step4_validate
# ---------------------------------------------------------------------------

class TestStep4Validate:

    def test_returns_dict_with_required_keys(self):
        brief = _make_brief()
        campaign = _make_campaign(brief)
        research = app.step2_research(brief)
        result = app.step4_validate(campaign, brief, research)
        assert {"status", "issues", "checked_at"}.issubset(set(result.keys()))

    def test_valid_campaign_returns_pass(self):
        brief = _make_brief()
        campaign = _make_campaign(brief)
        research = app.step2_research(brief)
        result = app.step4_validate(campaign, brief, research)
        assert result["status"] == "PASS"

    def test_empty_section_triggers_issue(self):
        brief = _make_brief()
        campaign = _make_campaign(brief)
        campaign["campaign_objective"] = ""
        research = app.step2_research(brief)
        result = app.step4_validate(campaign, brief, research)
        assert result["status"] == "WARN"
        assert any("campaign_objective" in i for i in result["issues"])

    def test_budget_not_summing_to_100_triggers_issue(self):
        brief = _make_brief()
        campaign = _make_campaign(brief)
        campaign["budget_allocation"]["split"] = {"Meta Ads (FB/IG)": 50.0, "Email": 20.0}
        research = app.step2_research(brief)
        result = app.step4_validate(campaign, brief, research)
        assert result["status"] == "WARN"
        assert any("Budget" in i or "100" in i for i in result["issues"])

    def test_issues_is_empty_for_clean_campaign(self):
        brief = _make_brief()
        campaign = _make_campaign(brief)
        research = app.step2_research(brief)
        result = app.step4_validate(campaign, brief, research)
        assert result["issues"] == []

    def test_status_is_warn_when_issues_exist(self):
        brief = _make_brief()
        campaign = _make_campaign(brief)
        campaign["kpis"] = ""
        research = app.step2_research(brief)
        result = app.step4_validate(campaign, brief, research)
        assert result["status"] == "WARN"


# ---------------------------------------------------------------------------
# render_campaign_markdown
# ---------------------------------------------------------------------------

class TestRenderCampaignMarkdown:

    def _build(self):
        brief = _make_brief()
        campaign = _make_campaign(brief)
        validation = _make_validation()
        return brief, campaign, validation

    def test_returns_string(self):
        b, c, v = self._build()
        result = app.render_campaign_markdown(b, c, v)
        assert isinstance(result, str)

    def test_contains_business_name_in_title(self):
        b, c, v = self._build()
        result = app.render_campaign_markdown(b, c, v)
        assert b["business_name"] in result

    def test_contains_all_section_headings(self):
        b, c, v = self._build()
        result = app.render_campaign_markdown(b, c, v)
        for heading in ["Campaign Objective", "Audience Personas", "Key Messages",
                        "Creative Concepts", "Channel Plan", "Budget Allocation",
                        "KPIs", "Timeline"]:
            assert heading in result

    def test_validation_status_shown(self):
        b, c, v = self._build()
        result = app.render_campaign_markdown(b, c, v)
        assert "PASS" in result

    def test_warn_status_shows_validation_notes_section(self):
        b, c, _ = self._build()
        v = _make_validation(status="WARN", issues=["Section 'kpis' is empty."])
        result = app.render_campaign_markdown(b, c, v)
        assert "Validation Notes" in result
        assert "kpis" in result

    def test_no_issues_omits_validation_notes_section(self):
        b, c, v = self._build()
        result = app.render_campaign_markdown(b, c, v)
        assert "Validation Notes" not in result

    def test_budget_figure_appears_in_output(self):
        b, c, v = self._build()
        result = app.render_campaign_markdown(b, c, v)
        # Budget $50,000 should appear somewhere
        assert "50,000" in result or "50000" in result

    def test_markdown_starts_with_heading(self):
        b, c, v = self._build()
        result = app.render_campaign_markdown(b, c, v)
        assert result.strip().startswith("#")

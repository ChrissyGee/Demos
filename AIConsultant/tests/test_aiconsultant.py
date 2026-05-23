"""
Unit tests for AIConsultant/app.py.

All pure-logic functions are tested in isolation. Streamlit is mocked before
import so that log() calls (which write to st.session_state.console) do not
raise errors, and st.set_page_config() at module level is a no-op.
"""

import os
import sys
import pandas as pd
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock streamlit BEFORE importing app so module-level st calls are safe.
# st.session_state.console must be a real list so log() append/len work.
# ---------------------------------------------------------------------------
_st = MagicMock()
_st.session_state = MagicMock()
_st.session_state.console = []
sys.modules["streamlit"] = _st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

SAMPLE_FORM = {
    "business_name": "TestCo",
    "industry": "SaaS / B2B Tech",
    "company_size": "Small (11-50)",
    "timeline_weeks": 12,
    "budget": 50000.0,
    "strategic_goals": ["Reduce operating cost", "Improve customer satisfaction / retention"],
    "ai_tools_in_use": ["General-purpose chatbots (ChatGPT, Claude, Gemini)"],
    "ai_tool_details": "ChatGPT used informally.",
    "ai_effectiveness": "Some time savings.",
    "governance_notes": "No formal policy.",
    "data_sources": "CRM and support tickets.",
    "pain_points": "Manual data entry is slow.\nKnowledge is fragmented.",
    "desired_outcomes": "Automate reports.\nImprove support quality.",
    "bottlenecks": "Approval cycles take too long.",
    "stakeholders": "CEO, CTO",
    "extra_context": "",
    "created_at": "2026-05-23T00:00:00",
}


# ---------------------------------------------------------------------------
# tool_web_search
# ---------------------------------------------------------------------------

class TestToolWebSearch:

    def test_returns_n_results(self):
        results = app.tool_web_search("banking AI", n=3)
        assert len(results) == 3

    def test_returns_fewer_when_n_is_1(self):
        results = app.tool_web_search("retail", n=1)
        assert len(results) == 1

    def test_result_keys(self):
        results = app.tool_web_search("healthcare", n=1)
        assert set(results[0].keys()) == {"source", "title", "snippet", "url"}

    def test_is_deterministic(self):
        r1 = app.tool_web_search("saas support deflection", n=3)
        r2 = app.tool_web_search("saas support deflection", n=3)
        assert r1 == r2

    def test_snippet_references_query(self):
        query = "manufacturing quality inspection"
        results = app.tool_web_search(query, n=2)
        for r in results:
            assert query in r["snippet"]

    def test_different_queries_produce_different_order(self):
        r1 = [r["source"] for r in app.tool_web_search("banking financial", n=7)]
        r2 = [r["source"] for r in app.tool_web_search("healthcare clinical", n=7)]
        assert r1 != r2


# ---------------------------------------------------------------------------
# tool_rag_retrieve
# ---------------------------------------------------------------------------

class TestToolRagRetrieve:

    def test_respects_k(self):
        results = app.tool_rag_retrieve("banking copilot", k=2)
        assert len(results) <= 2

    def test_relevant_doc_retrieved_for_banking(self):
        results = app.tool_rag_retrieve("banking financial copilot", k=5)
        ids = [d["id"] for d in results]
        assert "rag-001" in ids

    def test_relevant_doc_retrieved_for_saas(self):
        results = app.tool_rag_retrieve("saas support deflection chatbot", k=5)
        ids = [d["id"] for d in results]
        assert "rag-002" in ids

    def test_fallback_when_no_keyword_match(self):
        # Gibberish query returns default RAG_CORPUS[:k]
        results = app.tool_rag_retrieve("zzzzxyzqqqq999", k=3)
        assert len(results) == 3
        assert results[0]["id"] == "rag-001"

    def test_result_has_required_keys(self):
        results = app.tool_rag_retrieve("insurance claims", k=2)
        for doc in results:
            assert {"id", "title", "text", "tags"}.issubset(set(doc.keys()))


# ---------------------------------------------------------------------------
# tool_driver_tree
# ---------------------------------------------------------------------------

class TestToolDriverTree:

    def test_returns_nodes_list(self):
        tree = app.tool_driver_tree(["Slow approvals"], ["Reduce operating cost"])
        assert "nodes" in tree
        assert len(tree["nodes"]) == 1

    def test_multiple_pain_points_produce_multiple_nodes(self):
        tree = app.tool_driver_tree(["Pain A", "Pain B", "Pain C"], [])
        assert len(tree["nodes"]) == 3

    def test_empty_pain_points_returns_empty_nodes(self):
        tree = app.tool_driver_tree([], [])
        assert tree["nodes"] == []

    def test_node_keys(self):
        tree = app.tool_driver_tree(["Manual entry"], [])
        node = tree["nodes"][0]
        assert {"symptom", "drivers", "linked_goals"}.issubset(set(node.keys()))

    def test_standard_drivers_present(self):
        tree = app.tool_driver_tree(["Slow cycle"], [])
        drivers = tree["nodes"][0]["drivers"]
        assert "Manual data entry / lookup" in drivers
        assert "Wait time between handoffs" in drivers

    def test_goals_with_cost_keyword_are_linked(self):
        tree = app.tool_driver_tree(["problem"], ["Reduce operating cost"])
        assert "Reduce operating cost" in tree["nodes"][0]["linked_goals"]

    def test_goals_without_keyword_are_not_linked(self):
        tree = app.tool_driver_tree(["problem"], ["Enter a new market"])
        assert "Enter a new market" not in tree["nodes"][0]["linked_goals"]

    def test_satisfaction_keyword_links_goal(self):
        tree = app.tool_driver_tree(["problem"],
                                    ["Improve customer satisfaction / retention"])
        assert "Improve customer satisfaction / retention" in tree["nodes"][0]["linked_goals"]


# ---------------------------------------------------------------------------
# tool_prioritisation
# ---------------------------------------------------------------------------

class TestToolPrioritisation:

    def _opp(self, name, metric="cost", effort=3, data_ready=True, roi_band="Medium (3-12mo)"):
        return {"name": name, "metric": metric, "effort": effort,
                "data_ready": data_ready, "roi_band": roi_band}

    def test_returns_dataframe(self):
        df = app.tool_prioritisation([self._opp("Build chatbot")])
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = app.tool_prioritisation([self._opp("Build chatbot")])
        expected = {
            "Opportunity", "Primary metric", "Impact (1-5)",
            "Feasibility (1-5)", "Effort (1-5)", "ROI band", "Priority score",
        }
        assert expected.issubset(set(df.columns))

    def test_sorted_descending_by_priority_score(self):
        opps = [
            self._opp("High impact", metric="revenue", effort=1),
            self._opp("Low impact", metric="cost", effort=5, data_ready=False),
        ]
        df = app.tool_prioritisation(opps)
        scores = df["Priority score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_revenue_metric_gives_impact_5(self):
        df = app.tool_prioritisation([self._opp("Revenue opp", metric="revenue")])
        assert df.iloc[0]["Impact (1-5)"] == 5

    def test_cost_metric_gives_impact_5(self):
        df = app.tool_prioritisation([self._opp("Cost opp", metric="cost")])
        assert df.iloc[0]["Impact (1-5)"] == 5

    def test_speed_metric_gives_impact_4(self):
        df = app.tool_prioritisation([self._opp("Speed opp", metric="speed")])
        assert df.iloc[0]["Impact (1-5)"] == 4

    def test_unknown_metric_defaults_to_impact_3(self):
        df = app.tool_prioritisation([self._opp("Mystery", metric="unknown")])
        assert df.iloc[0]["Impact (1-5)"] == 3

    def test_data_not_ready_reduces_feasibility(self):
        df_ready = app.tool_prioritisation([self._opp("A", data_ready=True, effort=3)])
        df_not_ready = app.tool_prioritisation([self._opp("A", data_ready=False, effort=3)])
        assert df_ready.iloc[0]["Feasibility (1-5)"] > df_not_ready.iloc[0]["Feasibility (1-5)"]

    def test_index_reset_to_zero(self):
        df = app.tool_prioritisation([self._opp("X"), self._opp("Y", metric="revenue")])
        assert list(df.index) == [0, 1]


# ---------------------------------------------------------------------------
# tool_document_filler
# ---------------------------------------------------------------------------

class TestToolDocumentFiller:

    def test_single_placeholder(self):
        assert app.tool_document_filler("Hello {name}!", {"name": "World"}) == "Hello World!"

    def test_multiple_placeholders(self):
        result = app.tool_document_filler("{a} and {b}", {"a": "foo", "b": "bar"})
        assert result == "foo and bar"

    def test_no_placeholders_unchanged(self):
        t = "No variables here."
        assert app.tool_document_filler(t, {"unused": "value"}) == t

    def test_empty_template(self):
        assert app.tool_document_filler("", {"k": "v"}) == ""

    def test_numeric_value_coerced_to_string(self):
        assert app.tool_document_filler("Score: {score}", {"score": 42}) == "Score: 42"

    def test_unreferenced_field_is_ignored(self):
        result = app.tool_document_filler("{x}", {"x": "hello", "y": "ignored"})
        assert result == "hello"


# ---------------------------------------------------------------------------
# step1_assess_landscape
# ---------------------------------------------------------------------------

class TestStep1AssessLandscape:

    def test_captures_tools_in_use(self):
        landscape = app.step1_assess_landscape(SAMPLE_FORM)
        assert landscape["tools_in_use"] == SAMPLE_FORM["ai_tools_in_use"]

    def test_strips_whitespace_from_fields(self):
        form = {**SAMPLE_FORM, "ai_tool_details": "  some detail  "}
        landscape = app.step1_assess_landscape(form)
        assert landscape["details"] == "some detail"

    def test_all_keys_present(self):
        landscape = app.step1_assess_landscape(SAMPLE_FORM)
        assert set(landscape.keys()) == {
            "tools_in_use", "details", "effectiveness_notes",
            "governance_notes", "data_sources",
        }

    def test_empty_tools_list(self):
        form = {**SAMPLE_FORM, "ai_tools_in_use": []}
        landscape = app.step1_assess_landscape(form)
        assert landscape["tools_in_use"] == []

    def test_empty_details_stripped_to_empty_string(self):
        form = {**SAMPLE_FORM, "ai_tool_details": "   "}
        landscape = app.step1_assess_landscape(form)
        assert landscape["details"] == ""


# ---------------------------------------------------------------------------
# step2_identify_opportunities
# ---------------------------------------------------------------------------

class TestStep2IdentifyOpportunities:

    def test_splits_pain_points_on_newline(self):
        result = app.step2_identify_opportunities(SAMPLE_FORM)
        assert "Manual data entry is slow." in result["explicit_pain_points"]
        assert "Knowledge is fragmented." in result["explicit_pain_points"]

    def test_splits_desired_outcomes_on_newline(self):
        result = app.step2_identify_opportunities(SAMPLE_FORM)
        assert "Automate reports." in result["explicit_opportunities"]

    def test_latent_opp_triggered_by_manual_keyword(self):
        form = {**SAMPLE_FORM, "pain_points": "We do a lot of manual work.",
                "desired_outcomes": "", "bottlenecks": "", "extra_context": ""}
        result = app.step2_identify_opportunities(form)
        assert any("Automate manual" in opp for opp in result["latent_opportunities"])

    def test_latent_opp_triggered_by_knowledge_keyword(self):
        form = {**SAMPLE_FORM, "pain_points": "knowledge base is outdated",
                "desired_outcomes": "", "bottlenecks": "", "extra_context": ""}
        result = app.step2_identify_opportunities(form)
        assert any("RAG" in opp for opp in result["latent_opportunities"])

    def test_latent_opp_triggered_by_report_keyword(self):
        form = {**SAMPLE_FORM, "pain_points": "report writing takes too long",
                "desired_outcomes": "", "bottlenecks": "", "extra_context": ""}
        result = app.step2_identify_opportunities(form)
        assert any("report" in opp.lower() for opp in result["latent_opportunities"])

    def test_required_keys_present(self):
        result = app.step2_identify_opportunities(SAMPLE_FORM)
        assert set(result.keys()) == {
            "explicit_pain_points", "explicit_opportunities",
            "bottlenecks", "latent_opportunities", "stakeholders",
        }

    def test_empty_fields_produce_empty_lists(self):
        form = {**SAMPLE_FORM, "pain_points": "", "desired_outcomes": "",
                "bottlenecks": "", "extra_context": ""}
        result = app.step2_identify_opportunities(form)
        assert result["explicit_pain_points"] == []
        assert result["explicit_opportunities"] == []
        assert result["bottlenecks"] == []

    def test_no_duplicate_latent_opps(self):
        # Same keyword appearing twice should not produce duplicate latent entries.
        form = {**SAMPLE_FORM,
                "pain_points": "manual entry is slow",
                "desired_outcomes": "reduce manual steps",
                "bottlenecks": "", "extra_context": ""}
        result = app.step2_identify_opportunities(form)
        latent = result["latent_opportunities"]
        assert len(latent) == len(set(latent))


# ---------------------------------------------------------------------------
# step3_industry_research
# ---------------------------------------------------------------------------

class TestStep3IndustryResearch:

    def test_always_includes_failure_doc(self):
        result = app.step3_industry_research(SAMPLE_FORM)
        assert "rag-005" in [d["id"] for d in result["rag"]]

    def test_returns_web_results(self):
        result = app.step3_industry_research(SAMPLE_FORM)
        assert len(result["web"]) > 0

    def test_returns_rag_results(self):
        result = app.step3_industry_research(SAMPLE_FORM)
        assert len(result["rag"]) > 0

    def test_required_keys_present(self):
        result = app.step3_industry_research(SAMPLE_FORM)
        assert set(result.keys()) == {"web_search_query", "web", "rag_query", "rag"}

    def test_web_query_contains_industry(self):
        result = app.step3_industry_research(SAMPLE_FORM)
        assert "SaaS" in result["web_search_query"]

    def test_rag_query_contains_industry(self):
        result = app.step3_industry_research(SAMPLE_FORM)
        assert "SaaS" in result["rag_query"]


# ---------------------------------------------------------------------------
# step5_prioritisation
# ---------------------------------------------------------------------------

class TestStep5Prioritisation:

    def _opps(self, explicit=None, latent=None):
        return {
            "explicit_opportunities": explicit or [],
            "latent_opportunities": latent or [],
        }

    def test_seeds_defaults_when_no_candidates(self):
        result = app.step5_prioritisation(self._opps())
        assert not result["priority_df"].empty

    def test_uses_explicit_opportunities(self):
        result = app.step5_prioritisation(
            self._opps(explicit=["Build a support chatbot"])
        )
        opps = result["priority_df"]["Opportunity"].tolist()
        assert any("chatbot" in o.lower() for o in opps)

    def test_uses_latent_opportunities(self):
        result = app.step5_prioritisation(
            self._opps(latent=["Automate manual data entry with an assistant"])
        )
        opps = result["priority_df"]["Opportunity"].tolist()
        assert any("manual" in o.lower() for o in opps)

    def test_returns_required_keys(self):
        result = app.step5_prioritisation(self._opps(explicit=["Something"]))
        assert {"candidates", "priority_df"}.issubset(set(result.keys()))

    def test_priority_df_is_dataframe(self):
        result = app.step5_prioritisation(self._opps(explicit=["Build chatbot"]))
        assert isinstance(result["priority_df"], pd.DataFrame)

    def test_prompt_effort_heuristic_returns_quick_roi(self):
        # Opportunities containing "assistant" map to effort=2 → Quick (<3mo) ROI.
        result = app.step5_prioritisation(
            self._opps(latent=["Deploy an internal RAG assistant"])
        )
        row = result["priority_df"].iloc[0]
        assert row["ROI band"] == "Quick (<3mo)"

    def test_platform_effort_heuristic_returns_long_roi(self):
        # "platform" maps to effort=5 → Long (12mo+) ROI.
        result = app.step5_prioritisation(
            self._opps(explicit=["Build a custom ML platform from scratch"])
        )
        row = result["priority_df"].iloc[0]
        assert row["ROI band"] == "Long (12mo+)"


# ---------------------------------------------------------------------------
# step_validate
# ---------------------------------------------------------------------------

class TestStepValidate:

    def _build(self):
        """Build a minimal but complete set of engagement outputs."""
        landscape = app.step1_assess_landscape(SAMPLE_FORM)
        opportunities = app.step2_identify_opportunities(SAMPLE_FORM)
        research = app.step3_industry_research(SAMPLE_FORM)
        prioritisation = app.step5_prioritisation(opportunities)
        top_name = prioritisation["priority_df"].iloc[0]["Opportunity"]
        driver_tree = {
            "tree": app.tool_driver_tree(["Manual entry"], ["Reduce operating cost"]),
            "narrative": "Root causes are manual processes and fragmented knowledge.",
        }
        recommendations = {
            "headline": f"Focus on {top_name[:30]} for fastest ROI.",
            "supporting_arguments": "Evidence supports this path.",
            "evidence": {},
        }
        fast_plan = {
            "frequency": "Weekly working group.",
            "accountability": "Named owner accountable.",
            "scorecard_df": pd.DataFrame([{
                "Opportunity": top_name,
                "Primary metric": "Cost",
                "Baseline target window": "Weeks 1-4",
                "Pilot target": "+15%",
                "Scale-out target": "+30%",
            }]),
            "timing": "Phase 1: 4 weeks.",
            "risk_change": "Stage-gate every pilot.",
        }
        return landscape, opportunities, research, driver_tree, prioritisation, recommendations, fast_plan

    def test_pass_when_all_sections_present(self):
        l, o, r, dt, p, rec, fp = self._build()
        result = app.step_validate(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec, fast_plan=fp,
        )
        assert result["status"] == "PASS"
        assert result["issues"] == []

    def test_warn_when_landscape_missing(self):
        l, o, r, dt, p, rec, fp = self._build()
        result = app.step_validate(
            intake=SAMPLE_FORM, landscape=None, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec, fast_plan=fp,
        )
        assert result["status"] == "WARN"
        assert any("landscape" in issue for issue in result["issues"])

    def test_warn_when_research_missing(self):
        l, o, r, dt, p, rec, fp = self._build()
        result = app.step_validate(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=None,
            driver_tree=dt, prioritisation=p, recommendations=rec, fast_plan=fp,
        )
        assert result["status"] == "WARN"
        assert any("research" in issue for issue in result["issues"])

    def test_warn_when_headline_does_not_cite_top_opportunity(self):
        l, o, r, dt, p, rec, fp = self._build()
        rec_bad = {**rec, "headline": "XYZZY no overlap with any opportunity name."}
        result = app.step_validate(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec_bad, fast_plan=fp,
        )
        assert result["status"] == "WARN"
        assert any("Headline" in issue for issue in result["issues"])

    def test_result_keys(self):
        l, o, r, dt, p, rec, fp = self._build()
        result = app.step_validate(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec, fast_plan=fp,
        )
        assert set(result.keys()) == {"status", "issues", "checked_at"}

    def test_unverified_dollar_figure_triggers_warn(self):
        l, o, r, dt, p, rec, fp = self._build()
        # $999,999 does not appear in SAMPLE_FORM's intake JSON.
        rec_bad = {**rec, "headline": "This will save $999,999 annually."}
        result = app.step_validate(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec_bad, fast_plan=fp,
        )
        assert result["status"] == "WARN"
        assert any("$999,999" in issue for issue in result["issues"])


# ---------------------------------------------------------------------------
# render_engagement_markdown
# ---------------------------------------------------------------------------

class TestRenderEngagementMarkdown:

    def _build_all(self):
        landscape = app.step1_assess_landscape(SAMPLE_FORM)
        opportunities = app.step2_identify_opportunities(SAMPLE_FORM)
        research = app.step3_industry_research(SAMPLE_FORM)
        prioritisation = app.step5_prioritisation(opportunities)
        top_name = prioritisation["priority_df"].iloc[0]["Opportunity"]
        driver_tree = {
            "tree": app.tool_driver_tree(["Manual entry"], ["Reduce operating cost"]),
            "narrative": "Root causes are manual processes.",
        }
        recommendations = {
            "headline": f"Focus on {top_name[:30]}.",
            "supporting_arguments": "Three supporting points.",
            "evidence": {},
        }
        fast_plan = {
            "frequency": "Weekly reviews.",
            "accountability": "Named owner.",
            "scorecard_df": pd.DataFrame([{
                "Opportunity": top_name,
                "Primary metric": "Cost",
                "Baseline target window": "Weeks 1-4",
                "Pilot target": "+15%",
                "Scale-out target": "+30%",
            }]),
            "timing": "Phase 1: 4w",
            "risk_change": "Stage-gate pilots.",
        }
        validation = {"status": "PASS", "issues": [], "checked_at": "2026-05-23T00:00:00"}
        return (landscape, opportunities, research, driver_tree,
                prioritisation, recommendations, fast_plan, validation)

    def test_starts_with_business_name_heading(self):
        l, o, r, dt, p, rec, fp, v = self._build_all()
        md = app.render_engagement_markdown(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec,
            fast_plan=fp, validation=v,
        )
        assert md.startswith("# AI Consulting Engagement — TestCo")

    def test_contains_all_section_headers(self):
        l, o, r, dt, p, rec, fp, v = self._build_all()
        md = app.render_engagement_markdown(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec,
            fast_plan=fp, validation=v,
        )
        for header in [
            "## 1. Current AI Landscape",
            "## 2. Opportunities & Pain Points",
            "## 3. Industry & Competitor Research",
            "## 4. Driver Tree",
            "## 5. Prioritisation",
            "## 6. Recommendations",
            "## 7. FAST Implementation Plan",
        ]:
            assert header in md

    def test_headline_appears_in_output(self):
        l, o, r, dt, p, rec, fp, v = self._build_all()
        md = app.render_engagement_markdown(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec,
            fast_plan=fp, validation=v,
        )
        assert "Focus on" in md

    def test_validation_warn_issues_shown(self):
        l, o, r, dt, p, rec, fp, _ = self._build_all()
        v = {"status": "WARN", "issues": ["Section X is missing."], "checked_at": ""}
        md = app.render_engagement_markdown(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec,
            fast_plan=fp, validation=v,
        )
        assert "Section X is missing." in md

    def test_validation_pass_has_no_warn_section(self):
        l, o, r, dt, p, rec, fp, v = self._build_all()
        md = app.render_engagement_markdown(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec,
            fast_plan=fp, validation=v,
        )
        assert "## ⚠ Validation Notes" not in md

    def test_tools_in_use_appear_in_landscape_section(self):
        l, o, r, dt, p, rec, fp, v = self._build_all()
        md = app.render_engagement_markdown(
            intake=SAMPLE_FORM, landscape=l, opportunities=o, research=r,
            driver_tree=dt, prioritisation=p, recommendations=rec,
            fast_plan=fp, validation=v,
        )
        assert "General-purpose chatbots" in md

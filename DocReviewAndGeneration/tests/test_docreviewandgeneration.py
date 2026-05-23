"""
Unit tests for DocReviewAndGeneration/app.py.

All tool functions are tested through their deterministic fallback paths
(no OpenAI key in the test environment → llm_complete() always returns None).

Streamlit is mocked before import.  st.session_state.get returns "" so that
get_openai_client() short-circuits cleanly.  st.session_state.console is a
real list so log() calls succeed.
"""

import io
import os
import sys
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock streamlit before importing app
# ---------------------------------------------------------------------------
_st = MagicMock()
_st.session_state = MagicMock()
_st.session_state.console = []
_st.session_state.live_doc = ""
# .get("openai_key") must return "" (falsy) so get_openai_client() → None
_st.session_state.get = MagicMock(return_value="")
sys.modules["streamlit"] = _st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_live_doc(text: str) -> None:
    _st.session_state.live_doc = text


# ---------------------------------------------------------------------------
# tool_create_form
# ---------------------------------------------------------------------------

class TestToolCreateForm:

    def test_known_form_type_returns_new_doc(self):
        result = app.tool_create_form("intake")
        assert "new_doc" in result

    def test_intake_form_has_expected_fields(self):
        result = app.tool_create_form("intake")
        doc = result["new_doc"]
        for field in app.FORM_TEMPLATES["intake"]:
            assert field in doc

    def test_incident_report_form_has_correct_fields(self):
        result = app.tool_create_form("incident_report")
        assert result["fields"] == app.FORM_TEMPLATES["incident_report"]

    def test_unknown_form_type_returns_error_summary(self):
        result = app.tool_create_form("nonexistent_form")
        assert "new_doc" not in result
        assert "Unknown" in result["summary"] or "nonexistent_form" in result["summary"]

    def test_summary_reports_field_count(self):
        result = app.tool_create_form("vendor_onboarding")
        n = len(app.FORM_TEMPLATES["vendor_onboarding"])
        assert str(n) in result["summary"]

    def test_new_doc_is_valid_markdown_heading(self):
        result = app.tool_create_form("intake")
        assert result["new_doc"].startswith("#")

    def test_tool_key_is_correct(self):
        result = app.tool_create_form("intake")
        assert result["tool"] == "create_form"


# ---------------------------------------------------------------------------
# tool_fill_form
# ---------------------------------------------------------------------------

class TestToolFillForm:

    def _blank_intake(self) -> str:
        return app.tool_create_form("intake")["new_doc"]

    def test_fills_a_known_field(self):
        doc = self._blank_intake()
        result = app.tool_fill_form(doc, {"Client name": "Acme Corp"})
        assert "Acme Corp" in result["new_doc"]

    def test_summary_counts_filled_fields(self):
        doc = self._blank_intake()
        result = app.tool_fill_form(doc, {"Client name": "Acme", "Email": "a@b.com"})
        assert "2" in result["summary"]

    def test_empty_value_is_treated_as_missing(self):
        doc = self._blank_intake()
        result = app.tool_fill_form(doc, {"Client name": ""})
        assert "0" in result["summary"] or "missing" in result["summary"].lower()

    def test_unknown_field_does_not_crash(self):
        doc = self._blank_intake()
        result = app.tool_fill_form(doc, {"Nonexistent field xyz": "value"})
        assert "new_doc" in result

    def test_tool_key_is_correct(self):
        result = app.tool_fill_form("any doc", {"Client name": "Test"})
        assert result["tool"] == "fill_form"

    def test_filled_value_replaces_underscore_placeholder(self):
        doc = "# Intake Form\n\n- **Client name:** ______________________\n"
        result = app.tool_fill_form(doc, {"Client name": "TestCo"})
        assert "______" not in result["new_doc"].split("Client name")[1].split("\n")[0]


# ---------------------------------------------------------------------------
# tool_rag_retrieve
# ---------------------------------------------------------------------------

class TestToolRagRetrieve:

    def test_returns_dict_with_required_keys(self):
        result = app.tool_rag_retrieve("compliance gdpr privacy")
        assert {"tool", "hits", "summary"}.issubset(set(result.keys()))

    def test_tool_key_is_correct(self):
        assert app.tool_rag_retrieve("contract")["tool"] == "rag_retrieve"

    def test_respects_k(self):
        result = app.tool_rag_retrieve("contract legal agreement", k=2)
        assert len(result["hits"]) <= 2

    def test_gdpr_query_retrieves_gdpr_doc(self):
        result = app.tool_rag_retrieve("gdpr privacy data compliance", k=5)
        ids = [d["id"] for d in result["hits"]]
        assert "rag-gdpr-001" in ids

    def test_fallback_when_no_match(self):
        result = app.tool_rag_retrieve("zzzxyzqqqabc999", k=3)
        assert len(result["hits"]) == 3
        assert result["hits"][0]["id"] == "rag-gdpr-001"

    def test_summary_mentions_count(self):
        result = app.tool_rag_retrieve("contract legal msa", k=2)
        count = len(result["hits"])
        assert str(count) in result["summary"]


# ---------------------------------------------------------------------------
# tool_extract_key_info
# ---------------------------------------------------------------------------

class TestToolExtractKeyInfo:

    def test_extracts_plain_numbers(self):
        result = app.tool_extract_key_info("We had 312 responses and 41 NPS.")
        assert "312" in result["data"]["numbers"]
        assert "41" in result["data"]["numbers"]

    def test_extracts_percentages(self):
        result = app.tool_extract_key_info("Churn dropped from 14% to 9%.")
        assert "14%" in result["data"]["percentages"]
        assert "9%" in result["data"]["percentages"]

    def test_extracts_iso_dates(self):
        result = app.tool_extract_key_info("Report prepared on 2026-05-23.")
        assert "2026-05-23" in result["data"]["dates"]

    def test_extracts_proper_nouns(self):
        result = app.tool_extract_key_info("Microsoft and Anthropic announced a deal.")
        assert "Microsoft" in result["data"]["proper_nouns"]

    def test_empty_text_returns_empty_lists(self):
        result = app.tool_extract_key_info("")
        for key in ("numbers", "percentages", "dates", "proper_nouns"):
            assert result["data"][key] == []

    def test_tool_key_is_correct(self):
        assert app.tool_extract_key_info("some text")["tool"] == "extract_key_info"

    def test_result_has_summary_key(self):
        result = app.tool_extract_key_info("Revenue grew by 15% in Q1 2026.")
        assert "summary" in result

    def test_limits_numbers_to_25(self):
        # 30 numbers in text; should return at most 25
        text = " ".join(str(i) for i in range(30))
        result = app.tool_extract_key_info(text)
        assert len(result["data"]["numbers"]) <= 25


# ---------------------------------------------------------------------------
# tool_summarize (offline fallback — first N sentences)
# ---------------------------------------------------------------------------

class TestToolSummarize:

    def test_returns_dict_with_tool_and_summary_keys(self):
        result = app.tool_summarize("First. Second. Third.")
        assert {"tool", "summary"}.issubset(set(result.keys()))

    def test_tool_key_is_correct(self):
        assert app.tool_summarize("text")["tool"] == "summarize"

    def test_respects_max_sentences(self):
        text = "One. Two. Three. Four. Five. Six."
        result = app.tool_summarize(text, max_sentences=3)
        # Count full stops in the summary output (should be ≤ max_sentences)
        sents = [s for s in result["summary"].split(".") if s.strip()]
        assert len(sents) <= 3

    def test_empty_text_returns_empty_sentinel(self):
        result = app.tool_summarize("", max_sentences=3)
        assert result["summary"] == "(empty)"

    def test_single_sentence_preserved(self):
        text = "This is the only sentence."
        result = app.tool_summarize(text, max_sentences=5)
        assert "only sentence" in result["summary"]


# ---------------------------------------------------------------------------
# tool_review_document (deterministic checklist)
# ---------------------------------------------------------------------------

class TestToolReviewDocument:

    def test_empty_doc_returns_early_with_message(self):
        result = app.tool_review_document("")
        assert "empty" in result["summary"].lower()
        assert "issues" not in result  # early return has no issues key

    def test_detects_unfilled_placeholder(self):
        doc = "# Title\n\n2026-05-23\n\n- **Field:** ______________________\n"
        result = app.tool_review_document(doc)
        assert any("placeholder" in i.lower() for i in result["issues"])

    def test_detects_todo_marker(self):
        doc = "# Title\n\n2026-05-23\n\nSome text. TODO: finish this."
        result = app.tool_review_document(doc)
        assert any("TODO" in i for i in result["issues"])

    def test_detects_tbd_marker(self):
        doc = "# Title\n\n2026-05-23\n\nFees are TBD."
        result = app.tool_review_document(doc)
        assert any("TBD" in i for i in result["issues"])

    def test_detects_missing_heading(self):
        doc = "2026-05-23\n\nNo heading here."
        result = app.tool_review_document(doc)
        assert any("heading" in i.lower() for i in result["issues"])

    def test_detects_missing_date(self):
        doc = "# Title\n\nNo date mentioned anywhere."
        result = app.tool_review_document(doc)
        assert any("date" in i.lower() for i in result["issues"])

    def test_well_formed_doc_passes_structural_checks(self):
        doc = "# Quarterly Report\n\nPrepared 2026-05-23.\n\nAll content is final.\n"
        result = app.tool_review_document(doc)
        # Structural issues should be empty (no placeholders, TODO, no missing heading/date)
        structural = [i for i in result["issues"]
                      if "placeholder" in i.lower()
                      or "TODO" in i or "TBD" in i
                      or "heading" in i.lower()
                      or "date" in i.lower()]
        assert structural == []

    def test_result_has_required_keys(self):
        doc = "# Report\n\n2026-01-01\n\nContent here."
        result = app.tool_review_document(doc)
        assert {"tool", "focus", "issues", "summary"}.issubset(set(result.keys()))

    def test_tool_key_is_correct(self):
        doc = "# Report\n\n2026-01-01\n\nContent."
        assert app.tool_review_document(doc)["tool"] == "review_document"


# ---------------------------------------------------------------------------
# tool_generate_live_doc (offline fallback — template fill)
# ---------------------------------------------------------------------------

class TestToolGenerateLiveDoc:

    def test_returns_required_keys(self):
        result = app.tool_generate_live_doc("report", "Q1 Analysis")
        assert {"tool", "new_doc", "summary", "doc_type"}.issubset(set(result.keys()))

    def test_tool_key_is_correct(self):
        assert app.tool_generate_live_doc("report", "Test")["tool"] == "generate_live_doc"

    def test_offline_summary_mentions_template(self):
        result = app.tool_generate_live_doc("report", "Q1 Analysis")
        assert "offline" in result["summary"].lower() or "template" in result["summary"].lower()

    def test_topic_appears_in_report_title(self):
        result = app.tool_generate_live_doc("report", "Annual Review 2026")
        assert "Annual Review 2026" in result["new_doc"]

    def test_unknown_doc_type_falls_back_to_report(self):
        result = app.tool_generate_live_doc("unknown_type", "Test topic")
        # Should not raise; new_doc should be populated
        assert result["new_doc"].strip()

    def test_contract_template_includes_signatures_section(self):
        result = app.tool_generate_live_doc("contract", "Service deal")
        assert "Signatures" in result["new_doc"] or "Signature" in result["new_doc"]

    def test_memo_template_includes_memo_header(self):
        result = app.tool_generate_live_doc("memo", "IT policy update")
        assert "MEMORANDUM" in result["new_doc"] or "Memo" in result["new_doc"]


# ---------------------------------------------------------------------------
# route_intent
# ---------------------------------------------------------------------------

class TestRouteIntent:

    def setup_method(self):
        # Default: empty live doc so fallback → "generate_report"
        _set_live_doc("")

    def test_generate_report_intent(self):
        assert app.route_intent("generate a report about Q1 sales") == "generate_report"

    def test_generate_contract_intent(self):
        assert app.route_intent("draft a services contract for Acme") == "generate_contract"

    def test_generate_memo_intent(self):
        assert app.route_intent("write a memo about the new policy") == "generate_memo"

    def test_review_compliance_intent(self):
        assert app.route_intent("review for GDPR compliance") == "review_compliance"

    def test_review_general_intent(self):
        assert app.route_intent("review the document for quality") == "review_general"

    def test_create_form_intent(self):
        assert app.route_intent("create an intake form") == "create_form"

    def test_fill_form_intent(self):
        assert app.route_intent("fill the form with my details") == "fill_form"

    def test_summarize_intent(self):
        assert app.route_intent("summarize the document") == "summarize"

    def test_extract_intent(self):
        assert app.route_intent("extract key numbers and dates") == "extract"

    def test_rag_intent(self):
        assert app.route_intent("show me the knowledge reference for contracts") == "rag"

    def test_edit_intent(self):
        assert app.route_intent("edit the title of the document") == "edit"

    def test_fallback_with_empty_doc_returns_generate_report(self):
        _set_live_doc("")
        assert app.route_intent("xyzzy unknown phrase") == "generate_report"

    def test_fallback_with_nonempty_doc_returns_edit(self):
        _set_live_doc("# Existing document")
        assert app.route_intent("xyzzy unknown phrase") == "edit"


# ---------------------------------------------------------------------------
# parse_uploaded_file (file-like mock — txt/md/pdf paths)
# ---------------------------------------------------------------------------

class MockFile:
    """Minimal file-like object for testing parse_uploaded_file."""

    def __init__(self, name: str, content):
        self.name = name
        self._data = content if isinstance(content, bytes) else content.encode("utf-8")

    def read(self):
        return self._data


class TestParseUploadedFile:

    def test_txt_file_returns_text_kind(self):
        f = MockFile("note.txt", "Hello world")
        result = app.parse_uploaded_file(f)
        assert result is not None
        assert result["kind"] == "text"
        assert result["content"] == "Hello world"

    def test_md_file_returns_text_kind(self):
        f = MockFile("readme.md", "# Title\n\nContent here.")
        result = app.parse_uploaded_file(f)
        assert result is not None
        assert result["kind"] == "text"

    def test_txt_preserves_filename(self):
        f = MockFile("my_report.txt", "content")
        result = app.parse_uploaded_file(f)
        assert result["name"] == "my_report.txt"

    def test_pdf_file_extracts_printable_ascii(self):
        # A PDF with printable ASCII content should return something.
        pdf_like = b"%PDF-1.4 Hello from a fake PDF."
        f = MockFile("doc.pdf", pdf_like)
        result = app.parse_uploaded_file(f)
        assert result is not None
        assert result["kind"] == "pdf-extracted"
        assert "Hello" in result["content"]

    def test_unsupported_extension_returns_none(self):
        f = MockFile("data.xlsx", b"\x00\x01\x02")
        result = app.parse_uploaded_file(f)
        assert result is None

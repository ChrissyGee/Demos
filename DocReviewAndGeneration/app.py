"""
================================================================================
DOCUMENT REVIEW & GENERATION AGENT — STREAMLIT MVP
================================================================================

A self-contained demo of a Document Agent that takes natural-language
instructions ("review for compliance", "fill this form", "generate report
from this data") and produces or edits a live document the user can see
and tweak in real time.

ARCHITECTURE (matches the whiteboard sketch):

    +---------+         +-----------------+         +------------------------+
    |  Chat   |  ---->  |  Document Agent | ----->  |        Tools           |
    |  (User) |         |  (router +      |         |  - generate_live_doc   |
    +---------+         |   reasoner)     |         |  - edit_live_doc       |
                        +--------+--------+         |  - create_form         |
                                 |                  |  - rag_retrieve        |
                                 v                  |  - fill_form           |
                        +-----------------+         |  - review_document     |
                        |  Live Document  | <-------+  - summarize / extract |
                        |  (editable)     |         +------------------------+
                        +-----------------+

THE AGENTIC LOOP
    Every chat turn runs:
        1. PLAN  — pick which tool(s) to call, based on the user's intent.
        2. ACT   — execute the tool(s) against the current live document
                   or uploaded sources.
        3. WRITE — mutate the live document if the tool produced new content.
        4. REPLY — give the user a short natural-language summary of what
                   the agent did + which tools it used.
    Every step is logged to the Agent Console so the reasoning is auditable.

DETERMINISTIC CORE + OPTIONAL LLM
    The router and every tool work offline with deterministic logic so the
    demo always runs. If `OPENAI_API_KEY` is configured, generation and
    review tools call the LLM with the current document as grounding.

USAGE
    pip install streamlit pandas openai
    streamlit run app.py
================================================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import re
import io
import json
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import pandas as pd
import streamlit as st

# OpenAI is optional — the app works without it.
try:
    from openai import OpenAI
    _OPENAI_IMPORTED = True
except Exception:
    _OPENAI_IMPORTED = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Document Review & Generation Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# TEMPLATES — used by generate_live_doc & create_form
# ============================================================

DOC_TEMPLATES: Dict[str, str] = {
    "report": (
        "# {title}\n\n"
        "_Prepared {date}_\n\n"
        "## Executive Summary\n{summary}\n\n"
        "## Key Findings\n{findings}\n\n"
        "## Recommendations\n{recommendations}\n\n"
        "## Appendix — Source Data\n{data}\n"
    ),
    "contract": (
        "# Services Agreement\n\n"
        "**Effective Date:** {date}\n\n"
        "This Agreement is entered into between **{party_a}** (\"Client\") "
        "and **{party_b}** (\"Provider\").\n\n"
        "## 1. Scope of Services\n{scope}\n\n"
        "## 2. Term\n{term}\n\n"
        "## 3. Fees\n{fees}\n\n"
        "## 4. Confidentiality\nEach party shall protect the other's "
        "Confidential Information using reasonable measures.\n\n"
        "## 5. Termination\nEither party may terminate this Agreement with "
        "thirty (30) days' written notice.\n\n"
        "## 6. Signatures\n\n"
        "Client: ______________________   Date: __________\n\n"
        "Provider: ____________________   Date: __________\n"
    ),
    "summary": (
        "# {title} — Summary\n\n"
        "_Generated {date}_\n\n"
        "## TL;DR\n{tldr}\n\n"
        "## Key Points\n{points}\n\n"
        "## Open Questions\n{questions}\n"
    ),
    "memo": (
        "**MEMORANDUM**\n\n"
        "**To:** {to}\n"
        "**From:** {sender}\n"
        "**Date:** {date}\n"
        "**Re:** {subject}\n\n"
        "---\n\n"
        "{body}\n"
    ),
}

FORM_TEMPLATES: Dict[str, List[str]] = {
    "intake": [
        "Client name", "Company", "Email", "Phone",
        "Project description", "Timeline", "Budget",
    ],
    "incident_report": [
        "Reporter name", "Date of incident", "Location",
        "What happened", "Who was involved", "Immediate action taken",
        "Follow-up required",
    ],
    "vendor_onboarding": [
        "Vendor legal name", "Primary contact", "Email", "Phone",
        "Tax ID", "Address", "Banking details", "Insurance certificate",
    ],
}


# ============================================================
# RAG CORPUS — small pre-loaded knowledge base
# ============================================================
# Used by the `rag_retrieve` tool and the compliance reviewer.

RAG_CORPUS: List[Dict] = [
    {
        "id": "rag-gdpr-001",
        "title": "GDPR — data subject rights checklist",
        "tags": ["gdpr", "privacy", "compliance", "data", "rights"],
        "text": (
            "Documents handling EU personal data must reference: lawful basis "
            "for processing, data minimisation, data subject rights (access, "
            "rectification, erasure), retention period, and DPO contact."
        ),
    },
    {
        "id": "rag-msa-001",
        "title": "Standard MSA clauses",
        "tags": ["contract", "msa", "agreement", "legal"],
        "text": (
            "Every services agreement should include: scope, term, fees, "
            "IP ownership, confidentiality, liability cap, indemnification, "
            "termination for convenience and for cause, and governing law."
        ),
    },
    {
        "id": "rag-soc2-001",
        "title": "SOC 2 — vendor security expectations",
        "tags": ["soc2", "vendor", "security", "compliance"],
        "text": (
            "Vendor agreements should require: SOC 2 Type II report on "
            "request, encryption in transit and at rest, breach notification "
            "within 72 hours, sub-processor disclosure, and audit rights."
        ),
    },
    {
        "id": "rag-style-001",
        "title": "Plain-language writing guide",
        "tags": ["style", "writing", "plain language"],
        "text": (
            "Prefer short sentences (under 25 words). Avoid passive voice. "
            "Define acronyms on first use. Use active verbs. Numbers under "
            "10 spelled out unless technical."
        ),
    },
    {
        "id": "rag-report-001",
        "title": "Executive report structure",
        "tags": ["report", "executive", "structure"],
        "text": (
            "Executive reports lead with a 3-sentence TL;DR, followed by "
            "findings (numbered, evidence-backed), recommendations (one per "
            "finding), and an appendix with source data."
        ),
    },
]


# ============================================================
# DEMO DATA — used when the user clicks 'Load demo doc'
# ============================================================

DEMO_DOC = (
    "# Q1 Customer Feedback Snapshot\n\n"
    "Prepared 2026-04-02.\n\n"
    "We received 312 survey responses from active customers between Jan and "
    "Mar 2026. Net Promoter Score landed at 41, up from 32 in Q4 2025. The "
    "three most common positive themes were onboarding speed, support "
    "responsiveness, and pricing clarity. The two most common complaints "
    "were mobile app crashes (mentioned by 64 respondents) and confusing "
    "billing for annual plans (47 respondents). Churn intent dropped from "
    "14% to 9% quarter over quarter.\n"
)

DEMO_CSV = pd.DataFrame({
    "month": ["Jan", "Feb", "Mar"],
    "responses": [98, 112, 102],
    "nps": [37, 41, 44],
    "complaints_mobile": [22, 19, 23],
    "complaints_billing": [18, 14, 15],
})


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state() -> None:
    """Bootstrap every persistent key the app needs."""
    if "initialised" in st.session_state:
        return

    st.session_state.live_doc: str = ""
    st.session_state.uploaded_sources: List[Dict] = []   # [{name, kind, content}]
    st.session_state.chat: List[Dict] = []               # [{role, content}]
    st.session_state.console: List[Dict] = []            # [{ts, step, message}]
    st.session_state.outputs: List[Dict] = []            # generated artefacts
    st.session_state.form_state: Dict = {}
    st.session_state.openai_key: str = ""

    st.session_state.initialised = True


# ============================================================
# CONSOLE LOGGING
# ============================================================

def log(step: str, message: str) -> None:
    """Append one reasoning step to the agent's console log."""
    st.session_state.console.append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "step": step,
        "message": message,
    })
    if len(st.session_state.console) > 250:
        st.session_state.console = st.session_state.console[-250:]


# ============================================================
# OPENAI HELPERS
# ============================================================

def get_openai_client() -> Optional["OpenAI"]:
    """Return an OpenAI client if a key is set, else None."""
    if not _OPENAI_IMPORTED:
        return None
    key = os.environ.get("OPENAI_API_KEY") or st.session_state.get("openai_key")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


def llm_complete(system: str, user: str, max_tokens: int = 600) -> Optional[str]:
    """Call the LLM; return text on success or None on any failure."""
    client = get_openai_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log("LLM", f"OpenAI call failed: {e}")
        return None


# ============================================================
# SOURCE-PARSING HELPERS
# ============================================================

def parse_uploaded_file(uploaded_file) -> Optional[Dict]:
    """
    Convert a Streamlit uploaded file into a source dict.

    Supported:
        * .txt / .md — read as UTF-8 text
        * .csv       — read as a pandas DataFrame, stringified to markdown
        * .pdf       — naive byte-level text extraction (skipped if it fails)

    PDF parsing here is intentionally minimal so the demo has no heavy
    dependencies. For production use, swap in `pypdf` or `pdfplumber` —
    see the Integration Steps tab.
    """
    name = uploaded_file.name
    lower = name.lower()
    try:
        if lower.endswith((".txt", ".md")):
            content = uploaded_file.read().decode("utf-8", errors="replace")
            return {"name": name, "kind": "text", "content": content}
        if lower.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return {"name": name, "kind": "csv",
                    "content": df.to_markdown(index=False),
                    "df": df}
        if lower.endswith(".pdf"):
            raw = uploaded_file.read()
            # Naive: extract anything that looks like printable ASCII.
            text = re.sub(rb"[^\x20-\x7e\n\r]+", b" ", raw).decode(errors="replace")
            text = re.sub(r"\s+", " ", text).strip()
            return {"name": name, "kind": "pdf-extracted", "content": text[:8000]}
    except Exception as e:
        log("Upload", f"Failed to parse {name}: {e}")
    return None


def combined_sources_text() -> str:
    """Concatenate all uploaded sources into one grounding context string."""
    if not st.session_state.uploaded_sources:
        return ""
    parts = []
    for src in st.session_state.uploaded_sources:
        parts.append(f"--- SOURCE: {src['name']} ({src['kind']}) ---\n{src['content']}\n")
    return "\n".join(parts)


# ============================================================
# TOOLS — the agent's toolbelt
# ============================================================
# Each tool is a pure function that returns a dict with at least:
#   {"tool": "<name>", "summary": "...", ...tool-specific keys}
# Tools that mutate the live document set `new_doc` to the full new text.

def tool_generate_live_doc(doc_type: str, topic: str, sources: str = "") -> Dict:
    """Generate a new document from a template + LLM (or deterministic fallback)."""
    log("Tool", f"generate_live_doc(doc_type={doc_type!r}, topic={topic!r})")
    template = DOC_TEMPLATES.get(doc_type, DOC_TEMPLATES["report"])
    today = datetime.now().strftime("%Y-%m-%d")

    # Try LLM first.
    llm_out = llm_complete(
        system=(
            "You are a document drafter. Produce CLEAN MARKDOWN only. "
            "Use ONLY facts from the supplied sources — do not invent "
            "figures, names, or claims. If sources are empty, write a "
            "well-structured placeholder document that a human can fill in."
        ),
        user=(
            f"Draft a `{doc_type}` document about: {topic}\n\n"
            f"Sources:\n{sources or '(none)'}\n\n"
            f"Follow this skeleton:\n{template}"
        ),
        max_tokens=800,
    )
    if llm_out:
        return {"tool": "generate_live_doc", "doc_type": doc_type,
                "new_doc": llm_out,
                "summary": f"Drafted a new {doc_type} document via LLM."}

    # Deterministic fallback fills the template with safe placeholder text.
    filled = template.format(
        title=topic or "Untitled Report",
        date=today,
        summary="(LLM unavailable — please add an executive summary.)",
        findings="1. _Finding placeholder._\n2. _Finding placeholder._",
        recommendations="- _Recommendation placeholder._",
        data=sources or "_No source data attached._",
        party_a="Client Name", party_b="Provider Name",
        scope="_Describe the services to be provided._",
        term="_Specify the term._", fees="_Specify the fees._",
        tldr="_One-paragraph TL;DR._",
        points="- _Point 1_\n- _Point 2_",
        questions="- _Open question 1_",
        to="_Recipient_", sender="Document Agent",
        subject=topic or "Untitled Memo",
        body="_Memo body goes here._",
    )
    return {"tool": "generate_live_doc", "doc_type": doc_type,
            "new_doc": filled,
            "summary": f"Drafted a new {doc_type} document from a template (offline mode)."}


def tool_edit_live_doc(current_doc: str, instruction: str) -> Dict:
    """Edit the live document according to a natural-language instruction."""
    log("Tool", f"edit_live_doc(instruction={instruction!r})")
    if not current_doc.strip():
        return {"tool": "edit_live_doc", "new_doc": current_doc,
                "summary": "Nothing to edit — the live document is empty."}

    llm_out = llm_complete(
        system=(
            "You are a careful document editor. Apply ONLY the change the "
            "user requested. Return the FULL revised document as markdown, "
            "preserving every section that wasn't asked to change. Do not "
            "invent new facts."
        ),
        user=(
            f"Current document:\n```\n{current_doc}\n```\n\n"
            f"Edit instruction: {instruction}\n\n"
            f"Return the revised full document."
        ),
        max_tokens=1200,
    )
    if llm_out:
        return {"tool": "edit_live_doc", "new_doc": llm_out,
                "summary": f"Edited the document per: {instruction!r}"}

    # Offline fallback: append a revision note so nothing is lost silently.
    revised = current_doc + f"\n\n> _Revision note ({datetime.now():%H:%M}): {instruction}_"
    return {"tool": "edit_live_doc", "new_doc": revised,
            "summary": "Appended your instruction as a revision note (offline mode)."}


def tool_create_form(form_type: str) -> Dict:
    """Create a blank form (markdown checklist) from a template."""
    log("Tool", f"create_form(form_type={form_type!r})")
    fields = FORM_TEMPLATES.get(form_type)
    if fields is None:
        return {"tool": "create_form",
                "summary": f"Unknown form type {form_type!r}. "
                           f"Try: {', '.join(FORM_TEMPLATES.keys())}."}
    md = [f"# {form_type.replace('_', ' ').title()} Form", ""]
    for f in fields:
        md.append(f"- **{f}:** ______________________")
    new_doc = "\n".join(md)
    return {"tool": "create_form", "form_type": form_type, "fields": fields,
            "new_doc": new_doc,
            "summary": f"Created a blank {form_type} form with {len(fields)} fields."}


def tool_fill_form(current_doc: str, values: Dict[str, str]) -> Dict:
    """
    Fill a previously-created form with user-supplied values.

    Operates on the markdown produced by `tool_create_form` — replaces the
    underscore placeholder for each known field.
    """
    log("Tool", f"fill_form(fields_filled={len(values)})")
    new_doc = current_doc
    filled, missing = 0, []
    for field, value in values.items():
        if not value:
            missing.append(field)
            continue
        # Replace the placeholder line for this exact field label.
        pattern = re.compile(
            rf"(- \*\*{re.escape(field)}:\*\*) +_+",
        )
        new_doc, n = pattern.subn(rf"\1 {value}", new_doc)
        if n > 0:
            filled += 1
    return {"tool": "fill_form", "new_doc": new_doc,
            "summary": f"Filled {filled} field(s); {len(missing)} left blank."}


def tool_rag_retrieve(query: str, k: int = 3) -> Dict:
    """Retrieve the top-k documents from the in-memory RAG corpus."""
    log("Tool", f"rag_retrieve(query={query!r})")
    tokens = {t.lower() for t in re.findall(r"\w+", query)}
    scored = []
    for doc in RAG_CORPUS:
        haystack = " ".join(doc["tags"]) + " " + doc["title"].lower()
        score = sum(1 for t in tokens if t in haystack)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    hits = [d for _, d in scored[:k]] or RAG_CORPUS[:k]
    bullets = "\n".join(f"- **{d['title']}** ({d['id']}): {d['text']}" for d in hits)
    return {"tool": "rag_retrieve", "hits": hits,
            "summary": f"Retrieved {len(hits)} doc(s):\n{bullets}"}


def tool_review_document(current_doc: str, focus: str = "compliance") -> Dict:
    """
    Review the live document. Runs a deterministic checklist + (optionally)
    an LLM second pass for prose-level feedback.
    """
    log("Tool", f"review_document(focus={focus!r})")
    if not current_doc.strip():
        return {"tool": "review_document",
                "summary": "Nothing to review — the live document is empty."}

    issues: List[str] = []

    # --- Deterministic checks --------------------------------------------
    # Empty placeholders left behind.
    if re.search(r"_+\s*$", current_doc, re.MULTILINE):
        issues.append("Document contains unfilled underscore placeholders.")
    if "TODO" in current_doc or "TBD" in current_doc:
        issues.append("Document still contains TODO/TBD markers.")
    # Headings.
    if not re.search(r"^#\s+", current_doc, re.MULTILINE):
        issues.append("Document has no top-level heading.")
    # Date present?
    if not re.search(r"\b20\d{2}-\d{2}-\d{2}\b", current_doc) and \
       not re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
                     current_doc):
        issues.append("Document does not mention a date.")

    # --- Focus-specific RAG-backed checks --------------------------------
    rag_hits: List[Dict] = []
    if focus.lower() in {"compliance", "gdpr", "soc2", "legal", "contract"}:
        rag_hits = tool_rag_retrieve(focus)["hits"]
        for hit in rag_hits:
            # Crude check: does the doc mention any of the keywords?
            hit_keywords = [w for w in re.findall(r"\w+", hit["text"]) if len(w) > 6]
            mentions = sum(1 for w in hit_keywords[:8] if w.lower() in current_doc.lower())
            if mentions < 2:
                issues.append(
                    f"Document weakly aligned with '{hit['title']}' "
                    f"({hit['id']}) — fewer than 2 expected terms appear."
                )

    # --- Optional LLM second pass ---------------------------------------
    llm_notes = llm_complete(
        system=(
            f"You are a careful document reviewer focusing on: {focus}. "
            "Give a SHORT review (max 5 bullets). Only cite issues actually "
            "in the document. No invented quotes."
        ),
        user=f"Document:\n```\n{current_doc}\n```",
        max_tokens=350,
    )

    summary_lines = []
    if issues:
        summary_lines.append("**Checklist findings:**")
        summary_lines.extend(f"- {i}" for i in issues)
    else:
        summary_lines.append("**Checklist findings:** none.")
    if llm_notes:
        summary_lines.append("\n**LLM review notes:**")
        summary_lines.append(llm_notes)

    return {"tool": "review_document", "focus": focus,
            "issues": issues, "rag_hits": rag_hits, "llm_notes": llm_notes,
            "summary": "\n".join(summary_lines)}


def tool_summarize(text: str, max_sentences: int = 5) -> Dict:
    """Summarize a chunk of text (LLM-first, naive fallback)."""
    log("Tool", f"summarize(len_chars={len(text)})")
    llm_out = llm_complete(
        system=("Summarize the supplied text in plain English. Stay under "
                f"{max_sentences} sentences. Do not invent facts."),
        user=text,
        max_tokens=250,
    )
    if llm_out:
        return {"tool": "summarize", "summary": llm_out}
    # Fallback: first-N-sentences heuristic.
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    short = " ".join(sents[:max_sentences])
    return {"tool": "summarize", "summary": short or "(empty)"}


def tool_extract_key_info(text: str) -> Dict:
    """Pull out numbers, percentages, dates, and capitalised names."""
    log("Tool", "extract_key_info()")
    out = {
        "numbers": re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text)[:25],
        "percentages": re.findall(r"\b\d+(?:\.\d+)?%", text)[:25],
        "dates": re.findall(r"\b(?:20\d{2}-\d{2}-\d{2}|Q[1-4] 20\d{2})\b", text)[:25],
        "proper_nouns": list(dict.fromkeys(
            re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?\b", text)
        ))[:25],
    }
    bullets = []
    for k, v in out.items():
        bullets.append(f"- **{k}**: {', '.join(v) if v else '_(none)_'}")
    return {"tool": "extract_key_info", "data": out,
            "summary": "Extracted:\n" + "\n".join(bullets)}


# ============================================================
# AGENT — router + reasoning loop
# ============================================================

# Order matters: the first matching intent wins.
INTENT_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(create|make|start|new)\s+(an?\s+)?(intake|incident_report|vendor_onboarding)\s+form\b",
     "create_form"),
    (r"\b(fill|complete|populate)\s+(the\s+)?form\b", "fill_form"),
    (r"\b(generate|draft|write|create)\b.*(contract|agreement|msa)", "generate_contract"),
    (r"\b(generate|draft|write|create)\b.*(report)", "generate_report"),
    (r"\b(generate|draft|write|create)\b.*(memo)", "generate_memo"),
    (r"\b(generate|draft|write|create)\b.*(summary)", "generate_summary"),
    (r"\b(review|check|audit)\b.*(compliance|gdpr|soc2|legal|contract)", "review_compliance"),
    (r"\b(review|check|audit)\b", "review_general"),
    (r"\b(summari[sz]e|tl;dr)\b", "summarize"),
    (r"\b(extract|pull|find).*(key|info|numbers|dates|names)", "extract"),
    (r"\b(rag|knowledge|reference|standard|template)\b", "rag"),
    (r"\b(edit|revise|update|change|tweak|rewrite)\b", "edit"),
]


def route_intent(message: str) -> str:
    """Map a free-form user message to an intent key."""
    for pattern, intent in INTENT_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            return intent
    return "edit" if st.session_state.live_doc.strip() else "generate_report"


def run_agent(user_message: str) -> str:
    """
    One agent turn — plan, act, write, reply.

    Returns a short natural-language reply for the chat panel; mutates the
    live document in session state as a side effect.
    """
    log("Plan", f"Routing user message: {user_message!r}")
    intent = route_intent(user_message)
    log("Plan", f"Intent: {intent}")

    sources = combined_sources_text()
    doc = st.session_state.live_doc
    result: Dict = {}

    # --- ACT — pick the matching tool(s) -----------------------------
    if intent == "create_form":
        m = re.search(r"(intake|incident_report|vendor_onboarding)", user_message, re.IGNORECASE)
        form_type = (m.group(1).lower() if m else "intake")
        result = tool_create_form(form_type)

    elif intent == "fill_form":
        # Pull "field: value" pairs out of the message.
        pairs = dict(re.findall(r"([A-Za-z][A-Za-z ]+?)\s*[:=]\s*([^,;\n]+)", user_message))
        cleaned = {k.strip().rstrip(":"): v.strip() for k, v in pairs.items() if v.strip()}
        if not cleaned:
            return ("I can fill the form when you give me `field: value` pairs, "
                    "e.g. `fill the form Client name: Acme, Email: a@b.com`. "
                    "You can also use the Live Document tab's form helper.")
        result = tool_fill_form(doc, cleaned)

    elif intent == "generate_contract":
        result = tool_generate_live_doc("contract", user_message, sources)

    elif intent == "generate_report":
        result = tool_generate_live_doc("report", user_message, sources)

    elif intent == "generate_memo":
        result = tool_generate_live_doc("memo", user_message, sources)

    elif intent == "generate_summary":
        result = tool_generate_live_doc("summary", user_message, sources)

    elif intent == "review_compliance":
        focus_match = re.search(r"\b(gdpr|soc2|legal|contract|compliance)\b",
                                user_message, re.IGNORECASE)
        focus = focus_match.group(1) if focus_match else "compliance"
        result = tool_review_document(doc, focus=focus)

    elif intent == "review_general":
        result = tool_review_document(doc, focus="quality")

    elif intent == "summarize":
        target = sources or doc
        result = tool_summarize(target)
        # Summaries are stored as outputs rather than overwriting the doc.

    elif intent == "extract":
        target = sources or doc
        result = tool_extract_key_info(target)

    elif intent == "rag":
        result = tool_rag_retrieve(user_message)

    elif intent == "edit":
        result = tool_edit_live_doc(doc, user_message)

    else:
        return "I'm not sure how to help with that. Try 'generate a report', 'review for compliance', or 'fill the form'."

    # --- WRITE — update the live document or save as output -----------
    if "new_doc" in result:
        st.session_state.live_doc = result["new_doc"]
        log("Write", "Updated the live document.")

    if intent in {"summarize", "extract", "rag", "review_compliance",
                  "review_general"}:
        st.session_state.outputs.append({
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": f"{intent} · {user_message[:60]}",
            "tool": result["tool"],
            "content": result["summary"],
        })
        log("Write", f"Saved {intent} output to Generated Outputs.")

    # --- REPLY --------------------------------------------------------
    return result.get("summary", "Done.")


# ============================================================
# UI — TABS
# ============================================================

def render_upload_and_chat() -> None:
    """Tab 1 — upload sources and chat with the agent."""
    st.subheader("📥 Upload Sources & Chat")

    col_up, col_demo = st.columns([2, 1])
    with col_up:
        uploads = st.file_uploader(
            "Upload PDF / text / markdown / CSV files",
            type=["pdf", "txt", "md", "csv"],
            accept_multiple_files=True,
        )
        if uploads:
            for u in uploads:
                parsed = parse_uploaded_file(u)
                if parsed:
                    # De-dupe by filename.
                    st.session_state.uploaded_sources = [
                        s for s in st.session_state.uploaded_sources
                        if s["name"] != parsed["name"]
                    ] + [parsed]
                    log("Upload", f"Loaded {parsed['name']} ({parsed['kind']}).")
            st.success(f"Loaded {len(uploads)} file(s).")

    with col_demo:
        st.markdown("**Quick start**")
        if st.button("Load demo doc + CSV", use_container_width=True):
            st.session_state.live_doc = DEMO_DOC
            st.session_state.uploaded_sources.append({
                "name": "demo_metrics.csv",
                "kind": "csv",
                "content": DEMO_CSV.to_markdown(index=False),
                "df": DEMO_CSV,
            })
            log("Upload", "Loaded demo doc and demo CSV.")
            st.success("Demo loaded — try 'review for compliance' in chat.")
            st.rerun()

    if st.session_state.uploaded_sources:
        st.divider()
        st.markdown("#### Sources loaded")
        for s in st.session_state.uploaded_sources:
            with st.expander(f"📎 {s['name']} ({s['kind']})"):
                if s["kind"] == "csv" and "df" in s:
                    st.dataframe(s["df"], use_container_width=True, hide_index=True)
                else:
                    st.text_area("Content", value=s["content"], height=180,
                                 key=f"src_{s['name']}", disabled=True)

    st.divider()
    st.markdown("#### 💬 Chat with the Document Agent")
    st.caption("Try: *generate a contract for Acme Corp* · *review for compliance* · "
               "*create an intake form* · *fill the form Client name: Acme, Email: a@b.com* · "
               "*summarize the sources* · *extract key info*")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Tell the agent what to do...")
    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        reply = run_agent(user_input)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


def render_live_document() -> None:
    """Tab 2 — editable live preview of the working document."""
    st.subheader("📝 Live Document")
    st.caption("Edit directly — your changes become the new baseline the agent edits next.")

    c1, c2 = st.columns([2, 3])

    with c1:
        st.markdown("**Source (markdown)**")
        new_text = st.text_area(
            "Document source",
            value=st.session_state.live_doc,
            height=520,
            label_visibility="collapsed",
        )
        if new_text != st.session_state.live_doc:
            st.session_state.live_doc = new_text
            log("Edit", "User edited the live document directly.")

        c1a, c1b, c1c = st.columns(3)
        if c1a.button("Clear", use_container_width=True):
            st.session_state.live_doc = ""
            st.rerun()
        if c1b.button("Save as output", use_container_width=True):
            if st.session_state.live_doc.strip():
                st.session_state.outputs.append({
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "title": "Manual snapshot",
                    "tool": "manual_save",
                    "content": st.session_state.live_doc,
                })
                st.success("Saved to Generated Outputs.")
        c1c.download_button(
            "⬇ Download .md",
            data=st.session_state.live_doc or "(empty)",
            file_name=f"document_{datetime.now():%Y%m%d_%H%M%S}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with c2:
        st.markdown("**Rendered preview**")
        if st.session_state.live_doc.strip():
            st.markdown(st.session_state.live_doc)
        else:
            st.info("Empty document. Generate one from the Chat tab or paste content here.")

    # ---- Quick form filler -------------------------------------------
    st.divider()
    st.markdown("#### 🧾 Quick Form Filler")
    st.caption("If the live document is a form, fill its fields here without typing them in chat.")

    detected_fields = re.findall(r"-\s+\*\*([^*]+):\*\*", st.session_state.live_doc)
    if not detected_fields:
        st.caption("No form fields detected in the current document.")
    else:
        with st.form("form_filler"):
            values = {}
            cols = st.columns(2)
            for i, field in enumerate(detected_fields):
                values[field] = cols[i % 2].text_input(field, key=f"ff_{field}")
            submitted = st.form_submit_button("Fill form", type="primary",
                                              use_container_width=True)
        if submitted:
            cleaned = {k: v for k, v in values.items() if v.strip()}
            result = tool_fill_form(st.session_state.live_doc, cleaned)
            st.session_state.live_doc = result["new_doc"]
            st.success(result["summary"])
            st.rerun()


def render_agent_console() -> None:
    """Tab 3 — full reasoning trace of every agent turn."""
    st.subheader("🧠 Agent Console")
    st.caption("Every plan → tool call → write step the agent takes is logged here.")

    c1, c2 = st.columns([1, 1])
    if c1.button("Clear console"):
        st.session_state.console = []
        st.rerun()
    c2.metric("Log entries", len(st.session_state.console))

    if not st.session_state.console:
        st.info("Console is empty. Talk to the agent in the Upload & Chat tab.")
        return

    # Group by step type for at-a-glance counts.
    counts: Dict[str, int] = {}
    for e in st.session_state.console:
        counts[e["step"]] = counts.get(e["step"], 0) + 1
    cols = st.columns(min(6, max(1, len(counts))))
    for i, (k, v) in enumerate(counts.items()):
        cols[i % len(cols)].metric(k, v)

    st.divider()
    for entry in st.session_state.console[::-1]:
        prefix = {
            "Plan": "🧭", "Tool": "🔧", "Write": "✍",
            "Upload": "📥", "Edit": "📝", "LLM": "🤖",
        }.get(entry["step"], "•")
        st.markdown(
            f"`{entry['ts']}` {prefix} **{entry['step']}** — {entry['message']}"
        )


def render_generated_outputs() -> None:
    """Tab 4 — list of saved artefacts (summaries, reviews, manual saves)."""
    st.subheader("📦 Generated Outputs")
    st.caption("Summaries, reviews, extractions, and manual snapshots end up here. "
               "Download any of them for sharing.")

    if not st.session_state.outputs:
        st.info("No outputs yet. Ask the agent to summarize, review, or extract — "
                "or click 'Save as output' from the Live Document tab.")
        return

    if st.button("Clear all outputs"):
        st.session_state.outputs = []
        st.rerun()

    for i, out in enumerate(st.session_state.outputs[::-1]):
        with st.expander(f"📄 {out['title']}  ·  {out['ts']}  ·  {out['tool']}",
                         expanded=(i == 0)):
            st.markdown(out["content"])
            st.download_button(
                "⬇ Download",
                data=out["content"],
                file_name=f"{out['tool']}_{out['ts'].replace(' ', '_').replace(':', '')}.md",
                mime="text/markdown",
                key=f"dl_{i}",
            )


def render_integration_steps() -> None:
    """Tab 5 — how to swap simulated parts for real systems."""
    st.subheader("🔌 Integration Steps")
    st.markdown(
        """
        This MVP simulates the external bits so it always runs. To productionise:

        ### 1. Real PDF parsing
        Replace the byte-level extractor in `parse_uploaded_file()`:
        ```python
        from pypdf import PdfReader
        text = "\\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
        # For scanned PDFs, add OCR via `pytesseract` or AWS Textract.
        ```

        ### 2. Real RAG
        Swap `tool_rag_retrieve()` for a vector search:
        ```python
        emb = client.embeddings.create(model="text-embedding-3-small", input=query)
        hits = vector_db.query(emb.data[0].embedding, k=3)
        ```
        Stores: pgvector, Pinecone, Weaviate, Qdrant.

        ### 3. Real document storage
        Persist `st.session_state.live_doc` and `outputs` so reloading the
        page doesn't lose state. Options:
        - S3 / GCS for files
        - Postgres + a `documents` table for metadata + versions
        - Git for diff history

        ### 4. Format-specific generation
        Markdown is fine for previews but not always for the final artefact:
        - DOCX → `python-docx`
        - PDF → `weasyprint` (HTML→PDF) or `reportlab`
        - DocuSign / Adobe Sign for execution

        ### 5. Stronger reviewer
        The current reviewer combines a deterministic checklist + LLM notes.
        Add:
        - A clause-classifier (e.g. fine-tuned spaCy model) for contracts
        - A PII detector (Presidio) before any download
        - A diff-against-template tool for standard agreements

        ### 6. Tool calling via the LLM
        For more flexible routing, replace the regex router with OpenAI's
        function-calling API. The current router is intentionally explicit
        so the demo works offline — but production should let the LLM
        choose tools given a structured schema.

        ### 7. Audit trail
        The Agent Console is a great UX touch but not durable. Push every
        entry to your observability stack (Datadog, Honeycomb, OTel) so
        you can replay any agent decision after the fact.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📄 Document Agent")
        st.caption("Review & generation MVP")

        st.divider()
        st.markdown("### 🔑 OpenAI API key (optional)")
        st.text_input(
            "Key (session-only)",
            type="password",
            key="openai_key",
            label_visibility="collapsed",
        )
        if not _OPENAI_IMPORTED:
            st.info("`openai` package missing — using deterministic fallbacks.")
        elif not (os.environ.get("OPENAI_API_KEY") or st.session_state.openai_key):
            st.warning("No key — LLM tools fall back to templates.")
        else:
            st.success("OpenAI key detected.")

        st.divider()
        st.markdown("### 🛠 Tool palette")
        for t in [
            "generate_live_doc", "edit_live_doc", "create_form",
            "fill_form", "rag_retrieve", "review_document",
            "summarize", "extract_key_info",
        ]:
            st.markdown(f"- `{t}`")

        st.divider()
        st.markdown("### 🧠 Console (latest)")
        if st.button("Clear console", use_container_width=True,
                     key="sidebar_clear_console"):
            st.session_state.console = []
            st.rerun()
        with st.container(height=300):
            if not st.session_state.console:
                st.caption("No reasoning steps yet.")
            for entry in st.session_state.console[-30:][::-1]:
                st.markdown(
                    f"`{entry['ts']}` **{entry['step']}** — {entry['message']}"
                )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("📄 Document Review & Generation Agent")
    st.caption("Chat tells the Document Agent what to do · the agent calls tools · "
               "the live document updates · everything is auditable in the console.")

    tab_chat, tab_doc, tab_console, tab_out, tab_int = st.tabs([
        "📥 Upload & Chat",
        "📝 Live Document",
        "🧠 Agent Console",
        "📦 Generated Outputs",
        "🔌 Integration Steps",
    ])

    with tab_chat:    render_upload_and_chat()
    with tab_doc:     render_live_document()
    with tab_console: render_agent_console()
    with tab_out:     render_generated_outputs()
    with tab_int:     render_integration_steps()


if __name__ == "__main__":
    main()

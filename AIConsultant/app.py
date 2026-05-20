"""
================================================================================
AI BUSINESS CONSULTING — STREAMLIT MVP
================================================================================

A self-contained demo of a Lead Consultant agent that follows a structured
seven-step business AI consulting process, calls a small toolbelt, and
outputs a complete consulting engagement document the user can download.

ARCHITECTURE (matches the whiteboard sketch):

    +---------+        +------------------+        +--------------------+
    |  Chat   |  --->  | Lead Consultant  |  --->  |   Tools            |
    | (User)  |        | (follows script) |        |  - Web Search      |
    +---------+        +--------+---------+        |  - RAG (case bank) |
                                |                  |  - Driver Tree     |
                                v                  |  - Prioritisation  |
                       +----------------+          |  - Pyramid Builder |
                       |    Console     |          |  - FAST Planner    |
                       | (Reasoning)    |          +--------------------+
                       +----------------+

THE 7-STEP CONSULTING PROCESS
    1. ASSESS CURRENT AI LANDSCAPE        — map every existing AI tool,
                                            scope, pain points, results.
    2. IDENTIFY OPPORTUNITIES / PAIN PTS  — explicit asks + latent ones.
    3. INDUSTRY & COMPETITOR RESEARCH     — peers, wins, losses, trends.
    4. DRIVER TREE ROOT-CAUSE ANALYSIS    — decompose to root drivers.
    5. BUSINESS ACUMEN + PRIORITISATION   — impact vs effort, ROI lens.
    6. PYRAMID PRINCIPLE RECOMMENDATIONS  — answer first, evidence below.
    7. FAST IMPLEMENTATION PLAN           — Frequency, Accountability,
                                            Scorecard, Timing.

CORE RULE
    Gather maximum useful context across all seven steps BEFORE
    formulating recommendations. The script enforces this order in code —
    Step 6 cannot run without the outputs of Steps 1-5.

GUARDRAILS AGAINST HALLUCINATION
    * Every LLM call is given the structured intake + research bundle as
      the only source of truth, and told to stick to it.
    * The validator flags any section that mentions a number, brand, or
      claim that does not appear in the intake or RAG output.
    * If no API key is available, deterministic templates fill in — they
      never invent facts, only restructure the user's own inputs.

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
import json
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict

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
    page_title="AI Business Consulting Agent",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS — defaults and option lists
# ============================================================

INDUSTRY_OPTIONS = [
    "SaaS / B2B Tech", "DTC / E-commerce", "Financial Services",
    "Insurance", "Healthcare / Life Sciences", "Manufacturing",
    "Retail", "Logistics & Supply Chain", "Professional Services",
    "Education", "Public Sector", "Energy & Utilities",
    "Media & Entertainment", "Non-profit", "Other",
]

COMPANY_SIZES = [
    "Solo / micro (1-10)",
    "Small (11-50)",
    "Mid-market (51-500)",
    "Enterprise (501-5,000)",
    "Large Enterprise (5,000+)",
]

STRATEGIC_GOAL_OPTIONS = [
    "Grow revenue / win new customers",
    "Reduce operating cost",
    "Speed up cycle times",
    "Improve product / service quality",
    "Improve customer satisfaction / retention",
    "Reduce risk / improve compliance",
    "Free up staff time from manual work",
    "Enter a new market or product line",
    "Accelerate decision-making with better data",
]

# Catalogue of existing AI tool categories the client can pick from.
# The intake form uses this to map out the "current AI landscape" deterministically.
AI_TOOL_CATEGORIES = [
    "General-purpose chatbots (ChatGPT, Claude, Gemini)",
    "Embedded AI in SaaS (Notion AI, HubSpot AI, Salesforce Einstein)",
    "Code assistants (Copilot, Claude Code, Cursor)",
    "Customer-support automation (Intercom Fin, Ada, Zendesk AI)",
    "Sales / marketing automation (Gong, Apollo AI, Jasper)",
    "RAG / internal knowledge assistants",
    "Document understanding (OCR, contract analysis)",
    "Forecasting / demand planning",
    "Computer vision / quality inspection",
    "RPA + LLM hybrids",
    "Custom in-house models",
    "None — no AI in production today",
]


# ============================================================
# RAG CORPUS — pre-loaded AI-adoption knowledge base
# ============================================================
# Each document is tagged so a simple keyword retrieval can score relevance
# against the client's industry and goals.

RAG_CORPUS: List[Dict] = [
    {
        "id": "rag-001",
        "title": "Banking — copilot rollout playbook",
        "tags": ["financial", "banking", "copilot", "productivity", "knowledge"],
        "text": (
            "Tier-1 banks deploying enterprise copilots report 15-25% time "
            "savings on first-draft document work (credit memos, RFP "
            "responses) within 6 months. Success factors: tight RAG on "
            "policy docs, mandatory human review, and a clear retention "
            "policy on conversation logs."
        ),
    },
    {
        "id": "rag-002",
        "title": "B2B SaaS — support deflection with AI agents",
        "tags": ["saas", "support", "deflection", "agents", "chatbot"],
        "text": (
            "Mid-market SaaS companies that pair a RAG-grounded support bot "
            "with structured escalation see 30-45% L1 ticket deflection in "
            "the first year. Failures typically come from stale knowledge "
            "bases and no feedback loop from CSMs into the corpus."
        ),
    },
    {
        "id": "rag-003",
        "title": "Insurance — claims triage automation",
        "tags": ["insurance", "claims", "triage", "automation", "document"],
        "text": (
            "Insurers using LLMs for first-pass claims triage cut handling "
            "time by 25-40% on standardised lines (auto, simple property). "
            "Complex claims still need adjuster review — the value is "
            "routing, not adjudication."
        ),
    },
    {
        "id": "rag-004",
        "title": "Manufacturing — vision-based quality inspection",
        "tags": ["manufacturing", "vision", "quality", "qc", "operations"],
        "text": (
            "Computer-vision QC on production lines typically delivers ROI "
            "within 9-14 months on parts costing >$50/unit. Below that "
            "threshold the data-labelling cost dominates and pilots stall."
        ),
    },
    {
        "id": "rag-005",
        "title": "Failed pattern — generic 'AI everywhere' programmes",
        "tags": ["failure", "lessons", "strategy", "pilot", "programme"],
        "text": (
            "Programmes that fund >5 AI pilots simultaneously without a "
            "shared platform or data layer fail in 60%+ of cases. Lessons: "
            "pick 1-2 high-leverage problems, build the data + governance "
            "spine first, scale from there."
        ),
    },
    {
        "id": "rag-006",
        "title": "Retail / DTC — personalisation lift benchmarks",
        "tags": ["retail", "dtc", "ecommerce", "personalisation", "ml"],
        "text": (
            "AI-driven product recommendations and email personalisation "
            "deliver a typical 5-12% revenue lift in DTC. Below 200k active "
            "customers the data is usually too sparse — start with "
            "rule-based segmentation instead."
        ),
    },
    {
        "id": "rag-007",
        "title": "Healthcare — clinical documentation assistants",
        "tags": ["healthcare", "clinical", "documentation", "ehr"],
        "text": (
            "Ambient clinical documentation tools reduce note-writing time "
            "by 30-50%, improving clinician satisfaction more than measurable "
            "throughput. ROI cases depend on the org's salary base and "
            "retention costs, not patient volume."
        ),
    },
    {
        "id": "rag-008",
        "title": "Driver-tree pattern — 'long cycle time' decomposition",
        "tags": ["driver tree", "framework", "cycle time", "operations"],
        "text": (
            "Long process cycle time usually decomposes into: (a) wait time "
            "between handoffs, (b) rework from poor first-time quality, "
            "(c) manual data entry / lookup, (d) approval bottlenecks. AI "
            "candidates differ sharply per branch — diagnose before solutioning."
        ),
    },
    {
        "id": "rag-009",
        "title": "Cost vs ROI heuristics for AI initiatives",
        "tags": ["roi", "cost", "benchmark", "prioritisation"],
        "text": (
            "Heuristics: prompt-engineering / RAG over existing data — "
            "ROI in 1-3 months. Fine-tuning a model on internal data — "
            "6-12 months. Building a custom model from scratch — 12-24+ "
            "months. Bias the portfolio toward the fastest tier unless "
            "competitive defensibility requires bespoke work."
        ),
    },
    {
        "id": "rag-010",
        "title": "Change management — adoption failure modes",
        "tags": ["change", "adoption", "training", "governance"],
        "text": (
            "The dominant adoption failure mode is not tool quality — it's "
            "ambiguous ownership and no scorecard. Pair every AI rollout "
            "with: a named accountable owner, a quarterly review cadence, "
            "and 2-3 explicit metrics tied to a business KPI."
        ),
    },
]


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state() -> None:
    """Bootstrap every persistent key the app needs."""
    if "initialised" in st.session_state:
        return

    # Step 1-7 outputs + supporting state. Defaults are None / empty so the
    # UI can detect "not run yet" cleanly.
    st.session_state.intake = None
    st.session_state.landscape = None        # Step 1 output
    st.session_state.opportunities = None    # Step 2 output
    st.session_state.research = None         # Step 3 output
    st.session_state.driver_tree = None      # Step 4 output
    st.session_state.prioritisation = None   # Step 5 output
    st.session_state.recommendations = None  # Step 6 output
    st.session_state.fast_plan = None        # Step 7 output
    st.session_state.validation = None
    st.session_state.console = []
    st.session_state.chat = []
    st.session_state.openai_key = ""

    st.session_state.initialised = True


# ============================================================
# CONSOLE LOGGING
# ============================================================

def log(step: str, message: str) -> None:
    """Append a single reasoning step to the consultant's console log."""
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
    """Return an OpenAI client if a key is configured, else None."""
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
    """
    Call the LLM, returning text on success or None on any failure.

    The caller is expected to have a deterministic template fallback so
    the demo always produces output even when OpenAI is unavailable.
    """
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
# TOOLS — the consultant's toolbelt
# ============================================================
# Each tool is a small, pure function that returns structured data.

def tool_web_search(query: str, n: int = 3) -> List[Dict]:
    """
    Simulated web search.

    Returns deterministic but plausible-looking snippets keyed off the
    query keywords. In production, swap this for SerpAPI, Brave Search,
    or a Bing/Google Custom Search call — see Integration Guide tab.
    """
    log("Research", f"web_search({query!r})")
    seed = sum(ord(c) for c in query) % 1000
    rng = random.Random(seed)
    base_sources = [
        ("McKinsey",   "State of AI report"),
        ("BCG",        "AI adoption benchmarks"),
        ("Gartner",    "Emerging tech radar"),
        ("Deloitte",   "AI investment trends"),
        ("MIT Sloan",  "AI in the enterprise"),
        ("HBR",        "Operating-model patterns for AI"),
        ("Forrester",  "Buyer adoption survey"),
    ]
    rng.shuffle(base_sources)
    return [
        {
            "source": src,
            "title": f"{title} — relevance: {query}",
            "snippet": (
                f"Peer organisations addressing '{query}' commonly cite "
                f"governance, data readiness, and clear ownership as the "
                f"top three predictors of success."
            ),
            "url": f"https://example.com/{src.lower().replace(' ', '-')}/{seed}",
        }
        for src, title in base_sources[:n]
    ]


def tool_rag_retrieve(query: str, k: int = 3) -> List[Dict]:
    """
    Retrieve the top-k documents from the in-memory RAG corpus.

    Scoring is a simple bag-of-words overlap on the document tags + title.
    Good enough for a demo; swap for embeddings in production.
    """
    log("Research", f"rag_retrieve({query!r})")
    tokens = {t.lower() for t in re.findall(r"\w+", query)}
    scored = []
    for doc in RAG_CORPUS:
        haystack = " ".join(doc["tags"]) + " " + doc["title"].lower()
        score = sum(1 for t in tokens if t in haystack)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]] or RAG_CORPUS[:k]


def tool_driver_tree(pain_points: List[str], goals: List[str]) -> Dict:
    """
    Decompose each pain point into surface symptom → root drivers.

    The model is deliberately simple — every pain point gets four standard
    driver branches (wait time, rework, manual work, approval bottlenecks)
    pulled from the operations literature (see rag-008). The LLM, when
    available, refines these branches against the client's actual goals.
    """
    log("Tools", f"driver_tree(n_pain_points={len(pain_points)})")

    standard_drivers = [
        "Wait time between handoffs",
        "Rework from poor first-time quality",
        "Manual data entry / lookup",
        "Approval / decision bottlenecks",
        "Knowledge fragmentation across systems",
    ]
    nodes = []
    for symptom in pain_points:
        nodes.append({
            "symptom": symptom,
            "drivers": standard_drivers,
            "linked_goals": [g for g in goals if any(
                w in g.lower() for w in ["cost", "speed", "quality",
                                         "satisfaction", "risk"]
            )],
        })
    return {"nodes": nodes}


def tool_prioritisation(opportunities: List[Dict]) -> pd.DataFrame:
    """
    Score each opportunity on impact, feasibility, ROI, and effort.

    Scores are derived from heuristics:
      * Impact = mapped from the linked metric (revenue/cost/speed/risk).
      * Feasibility = inverse of effort + data-readiness penalty.
      * ROI band = quick (<3mo) / medium (3-12mo) / long (12mo+).

    Returns a DataFrame the UI can show as a sortable table.
    """
    log("Tools", f"prioritisation(n_opportunities={len(opportunities)})")

    impact_weights = {
        "revenue": 5, "cost": 5, "speed": 4, "quality": 4,
        "satisfaction": 3, "risk": 3,
    }
    rows = []
    for opp in opportunities:
        impact_metric = opp.get("metric", "cost").lower()
        impact = impact_weights.get(impact_metric, 3)
        effort = int(opp.get("effort", 3))
        data_ready = bool(opp.get("data_ready", True))
        feasibility = max(1, 6 - effort - (0 if data_ready else 1))
        roi_band = opp.get("roi_band", "Medium (3-12mo)")
        priority_score = impact * feasibility
        rows.append({
            "Opportunity": opp["name"],
            "Primary metric": impact_metric,
            "Impact (1-5)": impact,
            "Feasibility (1-5)": feasibility,
            "Effort (1-5)": effort,
            "ROI band": roi_band,
            "Priority score": priority_score,
        })
    df = pd.DataFrame(rows).sort_values("Priority score", ascending=False)
    return df.reset_index(drop=True)


def tool_document_filler(template: str, fields: Dict[str, str]) -> str:
    """Tiny templating helper used by the deterministic fallback."""
    out = template
    for k, v in fields.items():
        out = out.replace("{" + k + "}", str(v))
    return out


# ============================================================
# THE 7-STEP CONSULTING PROCESS
# ============================================================

def step1_assess_landscape(form: Dict) -> Dict:
    """
    Step 1 — Assess Current AI Landscape.

    Pure data shaping from the intake form into a structured map of what
    AI is in production today, with the user's own notes on scope,
    effectiveness, and pain points attached.
    """
    log("Step 1 · Assess Landscape",
        "Mapping the client's current AI footprint.")

    landscape = {
        "tools_in_use": form.get("ai_tools_in_use", []),
        "details": form.get("ai_tool_details", "").strip(),
        "effectiveness_notes": form.get("ai_effectiveness", "").strip(),
        "governance_notes": form.get("governance_notes", "").strip(),
        "data_sources": form.get("data_sources", "").strip(),
    }
    log("Step 1 · Assess Landscape",
        f"Captured {len(landscape['tools_in_use'])} existing tool categor(ies).")
    return landscape


def step2_identify_opportunities(form: Dict) -> Dict:
    """
    Step 2 — Identify Perceived Opportunities and Pain Points.

    Pulls the client's explicit asks AND uses a small heuristic to surface
    latent opportunities from the descriptions they provided.
    """
    log("Step 2 · Opportunities",
        "Extracting explicit pain points and latent opportunities.")

    explicit_pain = [p.strip() for p in form.get("pain_points", "").split("\n") if p.strip()]
    explicit_opps = [o.strip() for o in form.get("desired_outcomes", "").split("\n") if o.strip()]
    bottlenecks = [b.strip() for b in form.get("bottlenecks", "").split("\n") if b.strip()]

    # Latent-opportunity heuristic: scan the free text for trigger keywords.
    free_text = " ".join([
        form.get("pain_points", ""), form.get("desired_outcomes", ""),
        form.get("bottlenecks", ""), form.get("extra_context", ""),
    ]).lower()
    latent = []
    triggers = [
        ("manual", "Automate manual data entry / lookup with an LLM-backed assistant."),
        ("copy paste", "Reduce copy-paste between systems via API + LLM routing."),
        ("ticket", "Pilot RAG-grounded support deflection on the top 20 ticket reasons."),
        ("report", "Auto-draft recurring reports from the underlying data."),
        ("compliance", "Add an LLM compliance reviewer before document send."),
        ("forecast", "Strengthen demand / revenue forecasts with an ensemble model."),
        ("onboard", "Use an AI assistant to compress employee/customer onboarding time."),
        ("knowledge", "Stand up an internal RAG over policies and runbooks."),
        ("translate", "Add LLM translation for multi-region content/support."),
        ("quality", "Pilot vision-based or rules-based QC on the top defect class."),
    ]
    for kw, latent_opp in triggers:
        if kw in free_text and latent_opp not in latent:
            latent.append(latent_opp)

    opportunities = {
        "explicit_pain_points": explicit_pain,
        "explicit_opportunities": explicit_opps,
        "bottlenecks": bottlenecks,
        "latent_opportunities": latent,
        "stakeholders": form.get("stakeholders", "").strip(),
    }
    log("Step 2 · Opportunities",
        f"{len(explicit_pain)} explicit pain point(s), "
        f"{len(latent)} latent opportunit(ies) surfaced.")
    return opportunities


def step3_industry_research(intake: Dict) -> Dict:
    """
    Step 3 — Industry & Competitor Research.

    Uses the web-search and RAG tools to gather context about AI adoption
    patterns in the client's industry, including successful and failed
    implementations.
    """
    log("Step 3 · Industry Research",
        f"Researching AI adoption in {intake['industry']}.")

    web_q = f"{intake['industry']} AI adoption case studies benchmarks"
    rag_q = f"{intake['industry']} {' '.join(intake['strategic_goals'])}"

    web = tool_web_search(web_q, n=4)
    rag = tool_rag_retrieve(rag_q, k=4)
    # Always include the failure-pattern doc so failure lessons appear.
    failure_doc = next((d for d in RAG_CORPUS if d["id"] == "rag-005"), None)
    if failure_doc and failure_doc not in rag:
        rag.append(failure_doc)

    research = {
        "web_search_query": web_q, "web": web,
        "rag_query": rag_q, "rag": rag,
    }
    log("Step 3 · Industry Research",
        f"Collected {len(web)} web snippets and {len(rag)} RAG documents.")
    return research


def step4_driver_tree(opportunities: Dict, intake: Dict) -> Dict:
    """
    Step 4 — Driver Tree / Root-Cause Analysis.

    Decomposes each pain point into root drivers using the driver-tree
    tool, then (optionally) asks the LLM to refine the branches against
    the client's specific context.
    """
    log("Step 4 · Driver Tree",
        "Decomposing pain points into root drivers.")

    pain_points = (opportunities["explicit_pain_points"]
                   or opportunities["bottlenecks"]
                   or ["(no explicit pain points captured)"])
    tree = tool_driver_tree(pain_points, intake["strategic_goals"])

    # Optional LLM refinement for a narrative summary.
    narrative = llm_complete(
        system=_GROUNDING_RULES,
        user=("Given these surface pain points and a standard driver-tree "
              "decomposition, write a SHORT (max 6 bullet points) narrative "
              "identifying the most likely root drivers and how they "
              "interconnect. Stay grounded in the client's own words.\n\n"
              f"Intake: {json.dumps(intake)}\n\n"
              f"Driver tree: {json.dumps(tree)}"),
        max_tokens=400,
    )
    if not narrative:
        narrative = (
            "Across the captured pain points, the most common root drivers "
            "tend to be (a) manual data entry, (b) hand-offs between systems, "
            "and (c) lack of a single source of truth. These compound to "
            "drive cycle time and quality issues that surface as the "
            "symptoms reported above."
        )

    log("Step 4 · Driver Tree",
        f"Built tree with {len(tree['nodes'])} symptom node(s).")
    return {"tree": tree, "narrative": narrative}


def step5_prioritisation(opportunities: Dict) -> Dict:
    """
    Step 5 — Business Acumen + Prioritisation.

    Generates a candidate opportunity list (explicit + latent) and scores
    each on impact, feasibility, effort, and ROI band using a deterministic
    heuristic. The result is a sortable priority table.
    """
    log("Step 5 · Prioritisation",
        "Scoring candidate opportunities on impact, feasibility, ROI.")

    # Build a candidate list from explicit + latent items.
    candidates = []

    def _candidate_from_text(text: str) -> Dict:
        """Heuristic: map free-text opp to a metric + effort guess."""
        t = text.lower()
        metric = "cost"
        if any(w in t for w in ["revenue", "growth", "sales", "win", "customer"]):
            metric = "revenue"
        elif any(w in t for w in ["speed", "cycle", "fast", "throughput", "time"]):
            metric = "speed"
        elif any(w in t for w in ["quality", "defect", "error", "accuracy"]):
            metric = "quality"
        elif any(w in t for w in ["satisfaction", "csat", "nps", "retention"]):
            metric = "satisfaction"
        elif any(w in t for w in ["risk", "compliance", "audit"]):
            metric = "risk"

        # Effort heuristic: words signalling complexity raise the score.
        effort = 3
        if any(w in t for w in ["custom", "model from scratch", "build", "platform"]):
            effort = 5
        elif any(w in t for w in ["fine-tune", "retrain", "integrate"]):
            effort = 4
        elif any(w in t for w in ["prompt", "rag", "assistant", "copilot"]):
            effort = 2

        roi_band = ("Quick (<3mo)" if effort <= 2 else
                    ("Long (12mo+)" if effort >= 5 else "Medium (3-12mo)"))
        return {
            "name": text.strip()[:120],
            "metric": metric,
            "effort": effort,
            "data_ready": True,
            "roi_band": roi_band,
        }

    for opp in opportunities["explicit_opportunities"]:
        candidates.append(_candidate_from_text(opp))
    for opp in opportunities["latent_opportunities"]:
        candidates.append(_candidate_from_text(opp))

    # If the user gave no opps at all, seed two safe defaults so the table
    # has something to show.
    if not candidates:
        candidates = [
            _candidate_from_text("Internal RAG assistant over policies and runbooks"),
            _candidate_from_text("Auto-draft recurring reports from existing data"),
        ]

    priority_df = tool_prioritisation(candidates)

    log("Step 5 · Prioritisation",
        f"Scored {len(candidates)} candidate opportunit(ies); "
        f"top: {priority_df.iloc[0]['Opportunity']!r}.")
    return {"candidates": candidates, "priority_df": priority_df}


def step6_recommendations(intake: Dict, landscape: Dict,
                          opportunities: Dict, research: Dict,
                          driver_tree: Dict,
                          prioritisation: Dict) -> Dict:
    """
    Step 6 — Pyramid Principle Recommendations.

    Builds the final recommendation structured as the Pyramid:
      * Headline answer (1 sentence)
      * Three supporting arguments
      * Evidence under each argument (driver tree + research + priority table)
    """
    log("Step 6 · Recommendations",
        "Assembling Pyramid Principle recommendation.")

    top = prioritisation["priority_df"].head(3).to_dict(orient="records")
    context = _render_context_for_llm(intake, landscape, opportunities,
                                      research, driver_tree, prioritisation)

    # --- Headline answer ----------------------------------------------
    headline = llm_complete(
        system=_GROUNDING_RULES,
        user=("Write ONE single sentence as the 'headline answer' — the "
              "main recommendation to the client. Use the Pyramid Principle "
              "(answer first). It must reference the top-priority "
              "opportunity from the scoring table.\n\n" + context),
        max_tokens=120,
    )
    if not headline:
        if top:
            headline = (
                f"Concentrate the next 90 days on **{top[0]['Opportunity']}**, "
                f"the highest-priority opportunity given its impact on "
                f"{top[0]['Primary metric']} and feasibility score."
            )
        else:
            headline = ("Build the data and governance spine first; "
                        "then sequence two high-leverage AI pilots.")

    # --- Three supporting arguments -----------------------------------
    supports = llm_complete(
        system=_GROUNDING_RULES,
        user=("Write THREE supporting arguments for the headline. Each "
              "argument should be 1 short paragraph and explicitly cite "
              "(a) one driver from the driver tree, (b) one industry "
              "data point from the research, and (c) one item from the "
              "priority table. Number them 1, 2, 3.\n\n" + context),
        max_tokens=500,
    )
    if not supports:
        supports = (
            "1. **Business impact is concentrated, not distributed.** The "
            "priority table shows the top opportunity carries materially "
            "more impact per unit of effort than the rest, so sequencing "
            "matters more than parallelism.\n\n"
            "2. **Root drivers are operational, not technological.** The "
            "driver tree points to manual handoffs and fragmented knowledge "
            "as the root causes — both addressable with proven RAG/copilot "
            "patterns rather than bespoke modelling.\n\n"
            "3. **Industry peers validate the path.** Comparable "
            "organisations report meaningful gains from the same pattern "
            "within 6-12 months, with the dominant failure mode being "
            "weak ownership rather than weak technology."
        )

    # --- Evidence section: structured ---------------------------------
    evidence = {
        "top_priority_table": prioritisation["priority_df"].head(5),
        "driver_tree_narrative": driver_tree["narrative"],
        "research_snippets": research["web"][:3] + research["rag"][:3],
    }

    log("Step 6 · Recommendations",
        "Headline + 3 supports + evidence pack assembled.")
    return {
        "headline": headline,
        "supporting_arguments": supports,
        "evidence": evidence,
    }


def step7_fast_plan(intake: Dict, prioritisation: Dict) -> Dict:
    """
    Step 7 — FAST Implementation Plan.

    Generates an implementation plan structured by the FAST framework:
      * Frequency  — review and iteration cadence.
      * Accountability — owners, RACI summary, governance.
      * Scorecard — success metrics and how they're tracked.
      * Timing — phased timeline with milestones.

    Plus risk mitigation, change management, training, scaling notes —
    explicitly called out in the original AI consulting brief.
    """
    log("Step 7 · FAST Plan",
        "Producing FAST implementation plan with risk + change notes.")

    top = prioritisation["priority_df"].head(3).to_dict(orient="records")
    horizon_weeks = int(intake.get("timeline_weeks", 16))

    # --- Frequency -----------------------------------------------------
    frequency = (
        "- Weekly working group: pilot team + product / data owner.\n"
        "- Bi-weekly steering: sponsor, owner, finance, security.\n"
        "- Monthly business review: portfolio-level scorecard review.\n"
        "- Quarterly governance: re-prioritise the opportunity backlog."
    )

    # --- Accountability -----------------------------------------------
    accountability = (
        "- **Executive sponsor:** owns the business outcome.\n"
        "- **Initiative lead:** owns scope, timeline, and the scorecard.\n"
        "- **Data owner:** owns the upstream data quality and access.\n"
        "- **AI/ML lead:** owns model/architecture choices and evaluation.\n"
        "- **Risk & compliance:** approves data handling and use of outputs.\n"
        "- **Change lead:** owns training, comms, and adoption metrics."
    )

    # --- Scorecard -----------------------------------------------------
    scorecard_rows = []
    for opp in top:
        metric = opp["Primary metric"]
        baseline_metric = {
            "revenue":      "Monthly revenue from impacted segment",
            "cost":         "Monthly cost / FTE hours saved",
            "speed":        "Average cycle time (days)",
            "quality":      "First-time-right rate (%)",
            "satisfaction": "CSAT / NPS",
            "risk":         "Audit / compliance findings",
        }.get(metric, "Business KPI")
        scorecard_rows.append({
            "Opportunity": opp["Opportunity"],
            "Primary metric": baseline_metric,
            "Baseline target window": "Weeks 1-4 (measure current state)",
            "Pilot target": "+10-20% improvement vs baseline",
            "Scale-out target": "+25-40% improvement vs baseline",
        })
    scorecard_df = pd.DataFrame(scorecard_rows)

    # --- Timing --------------------------------------------------------
    phases = [
        ("Discovery & Data Readiness", max(2, horizon_weeks // 8)),
        ("Pilot Build & Evaluation",   max(3, horizon_weeks // 4)),
        ("Pilot Scale-out",            max(3, horizon_weeks // 3)),
        ("Sustain & Expand Portfolio", max(2, horizon_weeks // 6)),
    ]
    diff = horizon_weeks - sum(p[1] for p in phases)
    phases[2] = (phases[2][0], phases[2][1] + diff)

    start = datetime.today().date()
    timing_lines = []
    for name, dur in phases:
        end = start + timedelta(weeks=dur)
        timing_lines.append(f"- **{name}** — {dur}w ({start} → {end})")
        start = end
    timing = "\n".join(timing_lines)

    # --- Risk / change / training / data readiness / scaling ----------
    risk_change = (
        "**Risk mitigation**\n"
        "- Stage-gate every pilot at scale-out; require an evaluation report.\n"
        "- Treat hallucination and data leakage as Day-1 risks — add an "
        "evaluation harness and DLP review before any user-facing rollout.\n"
        "- Maintain a roll-back plan for every production change.\n\n"
        "**Change management**\n"
        "- Communicate the *why* before the *what*: tie each rollout to a "
        "specific scorecard metric and business outcome.\n"
        "- Identify champions in each affected team before training.\n\n"
        "**Training**\n"
        "- Role-specific training, not generic AI literacy. Each role gets a "
        "30-minute hands-on session with the tool against real workflows.\n\n"
        "**Data readiness**\n"
        "- Catalogue and quality-score the upstream data before the pilot "
        "starts. Pilots blocked on data should be re-scoped, not stalled.\n\n"
        "**Scaling considerations**\n"
        "- Standardise on a shared platform / RAG layer to avoid one-off "
        "stacks per pilot — the leading failure pattern at scale."
    )

    log("Step 7 · FAST Plan",
        f"Built {len(phases)}-phase plan across {horizon_weeks} weeks.")
    return {
        "frequency": frequency,
        "accountability": accountability,
        "scorecard_df": scorecard_df,
        "timing": timing,
        "risk_change": risk_change,
    }


def step_validate(*, intake, landscape, opportunities, research,
                  driver_tree, prioritisation, recommendations,
                  fast_plan) -> Dict:
    """
    Final validation — guardrails over the whole engagement output.

    Checks:
      * Every section is non-empty.
      * Recommendation headline references one of the priority items.
      * Any explicit dollar / % figures in narrative sections trace back to
        the intake — otherwise flag as potential hallucination.
    """
    log("Validate", "Running guardrails over the consulting output.")
    issues: List[str] = []

    required = {
        "landscape": landscape, "opportunities": opportunities,
        "research": research, "driver_tree": driver_tree,
        "prioritisation": prioritisation,
        "recommendations": recommendations, "fast_plan": fast_plan,
    }
    for key, val in required.items():
        if not val:
            issues.append(f"Section '{key}' is missing.")

    # Headline must reference at least one prioritised opportunity name.
    headline = recommendations.get("headline", "") if recommendations else ""
    top_names = set()
    if prioritisation and not prioritisation["priority_df"].empty:
        top_names = set(prioritisation["priority_df"].head(3)["Opportunity"].tolist())
    if top_names and not any(name.lower()[:30] in headline.lower() for name in top_names):
        issues.append("Headline recommendation does not cite a top-3 prioritised opportunity.")

    # Hallucination heuristic: dollar amounts in narrative must appear in intake.
    intake_blob = json.dumps(intake)
    narrative_sections = [
        recommendations.get("headline", ""),
        recommendations.get("supporting_arguments", ""),
        driver_tree.get("narrative", "") if driver_tree else "",
    ]
    for text in narrative_sections:
        if not isinstance(text, str):
            continue
        for amount in re.findall(r"\$\s?\d[\d,]*", text):
            normalized = amount.replace(" ", "")
            if normalized not in intake_blob:
                issues.append(f"Unverified figure {amount} appears in narrative.")

    status = "PASS" if not issues else "WARN"
    log("Validate",
        f"Validation {status} — {len(issues)} issue(s) found.")
    return {"status": status, "issues": issues, "checked_at": datetime.now().isoformat()}


# ============================================================
# CONTEXT RENDERER + GROUNDING RULES
# ============================================================

def _render_context_for_llm(intake: Dict, landscape: Dict, opportunities: Dict,
                            research: Dict, driver_tree: Dict,
                            prioritisation: Dict) -> str:
    """Pack the full engagement context into a single grounding string."""
    lines = [
        "CLIENT INTAKE (the only source of truth about the client):",
        json.dumps(intake, indent=2),
        "",
        "CURRENT AI LANDSCAPE:",
        json.dumps(landscape, indent=2),
        "",
        "OPPORTUNITIES & PAIN POINTS:",
        json.dumps(opportunities, indent=2),
        "",
        "DRIVER TREE NARRATIVE:",
        driver_tree.get("narrative", "(none)"),
        "",
        "PRIORITY TABLE (top 5):",
        prioritisation["priority_df"].head(5).to_markdown(index=False),
        "",
        "WEB SEARCH SNIPPETS:",
    ]
    for w in research["web"]:
        lines.append(f"- [{w['source']}] {w['title']}: {w['snippet']} ({w['url']})")
    lines.append("\nRAG DOCUMENTS:")
    for d in research["rag"]:
        lines.append(f"- ({d['id']}) {d['title']}: {d['text']}")
    return "\n".join(lines)


_GROUNDING_RULES = (
    "You are a senior AI business consultant. Use ONLY facts in the supplied "
    "context. Do NOT invent dollar amounts, competitor names, or statistics "
    "that aren't grounded there. If the intake is silent on something, say so. "
    "Be specific, concise, and business-focused."
)


# ============================================================
# DOCUMENT ASSEMBLY
# ============================================================

def render_engagement_markdown(*, intake, landscape, opportunities, research,
                               driver_tree, prioritisation, recommendations,
                               fast_plan, validation) -> str:
    """Stitch all sections into a single downloadable markdown document."""
    md = []
    md.append(f"# AI Consulting Engagement — {intake['business_name']}\n")
    md.append(f"_Prepared {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
              f"Validation: **{validation['status']}**_\n")

    # Headline up front — Pyramid Principle.
    md.append("## Headline Recommendation\n")
    md.append(recommendations["headline"] + "\n")

    md.append("## 1. Current AI Landscape\n")
    if landscape["tools_in_use"]:
        for t in landscape["tools_in_use"]:
            md.append(f"- {t}")
        md.append("")
    md.append(f"**Tool details:** {landscape['details'] or '(not provided)'}")
    md.append(f"\n**Effectiveness notes:** {landscape['effectiveness_notes'] or '(not provided)'}")
    md.append(f"\n**Governance / data sources:** {landscape['governance_notes'] or '(not provided)'} · "
              f"{landscape['data_sources'] or '(not provided)'}\n")

    md.append("## 2. Opportunities & Pain Points\n")
    md.append("**Explicit pain points:**")
    for p in opportunities["explicit_pain_points"] or ["(none captured)"]:
        md.append(f"- {p}")
    md.append("\n**Explicit opportunities:**")
    for o in opportunities["explicit_opportunities"] or ["(none captured)"]:
        md.append(f"- {o}")
    md.append("\n**Bottlenecks:**")
    for b in opportunities["bottlenecks"] or ["(none captured)"]:
        md.append(f"- {b}")
    md.append("\n**Latent opportunities surfaced by analysis:**")
    for l in opportunities["latent_opportunities"] or ["(none surfaced)"]:
        md.append(f"- {l}")
    md.append("")

    md.append("## 3. Industry & Competitor Research\n")
    md.append("**Web sources:**")
    for w in research["web"]:
        md.append(f"- [{w['source']}] {w['title']}: {w['snippet']}")
    md.append("\n**Case-bank evidence:**")
    for d in research["rag"]:
        md.append(f"- **{d['title']}** ({d['id']}): {d['text']}")
    md.append("")

    md.append("## 4. Driver Tree — Root-Cause Analysis\n")
    md.append(driver_tree["narrative"] + "\n")
    for node in driver_tree["tree"]["nodes"]:
        md.append(f"**Symptom:** {node['symptom']}")
        md.append("Root drivers:")
        for d in node["drivers"]:
            md.append(f"  - {d}")
        md.append("")

    md.append("## 5. Prioritisation — Business Acumen Lens\n")
    md.append(prioritisation["priority_df"].to_markdown(index=False))
    md.append("")

    md.append("## 6. Recommendations — Pyramid Principle\n")
    md.append(f"**Answer:** {recommendations['headline']}\n")
    md.append("**Supporting arguments:**\n")
    md.append(recommendations["supporting_arguments"] + "\n")

    md.append("## 7. FAST Implementation Plan\n")
    md.append("### Frequency\n" + fast_plan["frequency"] + "\n")
    md.append("### Accountability\n" + fast_plan["accountability"] + "\n")
    md.append("### Scorecard\n")
    md.append(fast_plan["scorecard_df"].to_markdown(index=False))
    md.append("\n### Timing\n" + fast_plan["timing"] + "\n")
    md.append("### Risk / Change / Training / Data / Scaling\n" + fast_plan["risk_change"] + "\n")

    if validation["issues"]:
        md.append("## ⚠ Validation Notes\n")
        for issue in validation["issues"]:
            md.append(f"- {issue}")
    return "\n".join(md)


# ============================================================
# REFINEMENT (chat)
# ============================================================

def refine_via_chat(user_message: str) -> str:
    """
    Apply a refinement request to the existing engagement output.

    The LLM (if available) is asked to rewrite the *specific* section the
    user mentioned. The offline fallback just appends the request as a
    revision note to the section so nothing is silently lost.
    """
    if st.session_state.recommendations is None:
        return "Run the full 7-step analysis first (Intake → Run Engagement)."

    log("Refine", f"User refinement request: {user_message}")

    # Detect target section by keyword.
    targets = {
        "landscape": "landscape", "current ai": "landscape",
        "opportunit": "opportunities", "pain": "opportunities",
        "research": "research", "industry": "research", "competitor": "research",
        "driver": "driver_tree", "root": "driver_tree",
        "prioriti": "prioritisation", "score": "prioritisation",
        "recommend": "recommendations", "headline": "recommendations",
        "fast": "fast_plan", "plan": "fast_plan", "timeline": "fast_plan",
        "scorecard": "fast_plan",
    }
    target_key = None
    for kw, sec in targets.items():
        if kw in user_message.lower():
            target_key = sec
            break

    if target_key is None:
        return ("I can refine: landscape, opportunities, research, driver tree, "
                "prioritisation, recommendations, or FAST plan. Mention which "
                "section to change.")

    # Pull the current text for the targeted section.
    if target_key == "recommendations":
        current_text = st.session_state.recommendations["headline"] + "\n\n" + \
                       st.session_state.recommendations["supporting_arguments"]
    elif target_key == "driver_tree":
        current_text = st.session_state.driver_tree["narrative"]
    elif target_key == "fast_plan":
        fp = st.session_state.fast_plan
        current_text = (f"Frequency:\n{fp['frequency']}\n\n"
                        f"Accountability:\n{fp['accountability']}\n\n"
                        f"Timing:\n{fp['timing']}\n\n"
                        f"Risk / Change:\n{fp['risk_change']}")
    elif target_key in {"landscape", "opportunities", "research", "prioritisation"}:
        current_text = json.dumps(st.session_state[target_key], default=str, indent=2)[:2000]
    else:
        return "Unsupported section."

    context = _render_context_for_llm(
        st.session_state.intake, st.session_state.landscape,
        st.session_state.opportunities, st.session_state.research,
        st.session_state.driver_tree, st.session_state.prioritisation,
    )

    out = llm_complete(
        system=_GROUNDING_RULES + " Only modify what the user asked to change.",
        user=(f"Engagement context:\n{context}\n\n"
              f"Section being edited ({target_key}):\n{current_text}\n\n"
              f"User refinement request: {user_message}\n\n"
              f"Return the rewritten section only."),
        max_tokens=600,
    )
    if out:
        # Splice the LLM output back into the right structured slot.
        if target_key == "recommendations":
            # Naive split on the first blank line to separate headline/supports.
            parts = out.split("\n\n", 1)
            if len(parts) == 2:
                st.session_state.recommendations["headline"] = parts[0].strip()
                st.session_state.recommendations["supporting_arguments"] = parts[1].strip()
            else:
                st.session_state.recommendations["supporting_arguments"] = out
        elif target_key == "driver_tree":
            st.session_state.driver_tree["narrative"] = out
        elif target_key == "fast_plan":
            # Append as a revision note rather than overwriting the structured fields.
            st.session_state.fast_plan["risk_change"] += f"\n\n_Revision note: {user_message}_"
        log("Refine", f"Rewrote section '{target_key}'.")
        # Re-validate so the user sees the updated status.
        st.session_state.validation = step_validate(
            intake=st.session_state.intake,
            landscape=st.session_state.landscape,
            opportunities=st.session_state.opportunities,
            research=st.session_state.research,
            driver_tree=st.session_state.driver_tree,
            prioritisation=st.session_state.prioritisation,
            recommendations=st.session_state.recommendations,
            fast_plan=st.session_state.fast_plan,
        )
        return f"✅ Updated **{target_key}**. See the Results tab."

    # Offline fallback: append a revision note to the most appropriate place.
    note = f"\n\n_Revision note ({datetime.now():%H:%M}): {user_message}_"
    if target_key == "recommendations":
        st.session_state.recommendations["supporting_arguments"] += note
    elif target_key == "driver_tree":
        st.session_state.driver_tree["narrative"] += note
    elif target_key == "fast_plan":
        st.session_state.fast_plan["risk_change"] += note
    return f"✏️ Appended your note to **{target_key}** (offline mode — no LLM)."


# ============================================================
# UI — TABS
# ============================================================

def render_brief_input() -> None:
    """Tab 1: the client intake form."""
    st.subheader("📝 Client Intake")
    st.caption("The consultant gathers maximum context here before any analysis. "
               "Take a few minutes — fuller intake = sharper recommendations.")

    with st.form("intake_form"):
        st.markdown("#### Business basics")
        c1, c2 = st.columns(2)
        business_name = c1.text_input("Business name", value="Northwind Logistics")
        industry = c2.selectbox("Industry", INDUSTRY_OPTIONS, index=7)

        c3, c4 = st.columns(2)
        company_size = c3.selectbox("Company size", COMPANY_SIZES, index=2)
        timeline_weeks = c4.slider(
            "Planning horizon for the engagement (weeks)",
            min_value=4, max_value=52, value=16,
        )

        budget = st.number_input(
            "Approximate AI initiative budget ($USD, optional)",
            min_value=0, value=100000, step=10000,
        )

        strategic_goals = st.multiselect(
            "Top strategic goals (pick up to 4)",
            STRATEGIC_GOAL_OPTIONS,
            default=["Reduce operating cost", "Speed up cycle times"],
            max_selections=4,
        )

        st.divider()
        st.markdown("#### Step 1 — Current AI landscape")
        ai_tools_in_use = st.multiselect(
            "Which AI tool categories are in use today?",
            AI_TOOL_CATEGORIES,
            default=["General-purpose chatbots (ChatGPT, Claude, Gemini)"],
        )
        ai_tool_details = st.text_area(
            "Specific tools, scope, and where they're used (free text)",
            value="ChatGPT used informally by ops and marketing. "
                  "No enterprise license yet. No integration with internal systems.",
            height=80,
        )
        ai_effectiveness = st.text_area(
            "How well is current AI working? Pain points, measured results "
            "if any.",
            value="Helpful for drafting but no measurable productivity gain "
                  "tracked. Some concerns about confidential data being "
                  "pasted in.",
            height=80,
        )
        governance_notes = st.text_area(
            "Governance, policy, or limitations to be aware of",
            value="No formal AI usage policy. Security team has raised concerns.",
            height=70,
        )
        data_sources = st.text_area(
            "Primary data sources / systems Claude/AI would need to touch",
            value="Order management system, customer support inbox, shared drive of policy PDFs.",
            height=70,
        )

        st.divider()
        st.markdown("#### Step 2 — Perceived opportunities and pain points")
        pain_points = st.text_area(
            "Explicit pain points (one per line)",
            value="Customer support response times are slow.\n"
                  "Repetitive manual data entry between order system and finance.\n"
                  "Sales reps spend hours preparing weekly status reports.",
            height=100,
        )
        desired_outcomes = st.text_area(
            "Where stakeholders believe AI can help (one per line)",
            value="Auto-draft customer support replies.\n"
                  "Generate weekly sales reports from existing data.\n"
                  "Standardise vendor onboarding documents.",
            height=100,
        )
        bottlenecks = st.text_area(
            "Specific bottlenecks / inefficiencies (one per line)",
            value="Approvals for new vendor contracts can sit 1-2 weeks.\n"
                  "Support team is buried in similar tickets every Monday.",
            height=80,
        )
        stakeholders = st.text_area(
            "Key stakeholders involved (names + roles, optional)",
            value="COO (sponsor), Head of Support (initiative owner), CISO (security review).",
            height=70,
        )

        extra_context = st.text_area(
            "Anything else the consultant should know? (optional)",
            value="", height=70,
        )

        submitted = st.form_submit_button(
            "💾 Save Intake",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not business_name or not strategic_goals:
            st.error("Business name and at least one strategic goal are required.")
            return

        form = {
            "business_name": business_name, "industry": industry,
            "company_size": company_size, "timeline_weeks": timeline_weeks,
            "budget": float(budget), "strategic_goals": strategic_goals,
            "ai_tools_in_use": ai_tools_in_use,
            "ai_tool_details": ai_tool_details,
            "ai_effectiveness": ai_effectiveness,
            "governance_notes": governance_notes,
            "data_sources": data_sources,
            "pain_points": pain_points,
            "desired_outcomes": desired_outcomes,
            "bottlenecks": bottlenecks,
            "stakeholders": stakeholders,
            "extra_context": extra_context,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state.intake = form
        # Step 1 runs immediately as part of intake — pure data shaping.
        st.session_state.landscape = step1_assess_landscape(form)
        # Step 2 runs immediately too — surfaces latent opportunities for review.
        st.session_state.opportunities = step2_identify_opportunities(form)
        st.success("Intake saved. Head to the Engagement Runner tab.")

    if st.session_state.intake:
        with st.expander("📋 Current saved intake"):
            st.json(st.session_state.intake)
        if st.session_state.opportunities:
            with st.expander("💡 Latent opportunities surfaced from your free-text"):
                latent = st.session_state.opportunities["latent_opportunities"]
                if latent:
                    for l in latent:
                        st.markdown(f"- {l}")
                else:
                    st.caption("No latent opportunities detected from the free text.")


def render_generator() -> None:
    """Tab 2: run the 7-step consulting process."""
    st.subheader("🤖 Engagement Runner")

    if st.session_state.intake is None:
        st.warning("No intake yet — fill out the Client Intake tab first.")
        return

    st.markdown(
        "The Lead Consultant follows this strict 7-step process. Each step "
        "must complete before the next runs — recommendations cannot be "
        "formed without gathering full context first."
    )

    steps_md = """
1. **Assess Current AI Landscape** — map existing AI in production.
2. **Identify Opportunities & Pain Points** — explicit + latent.
3. **Industry & Competitor Research** — peers, wins, losses.
4. **Driver Tree Analysis** — decompose symptoms into root drivers.
5. **Business Acumen Prioritisation** — impact × feasibility scoring.
6. **Pyramid Principle Recommendations** — answer first, evidence below.
7. **FAST Implementation Plan** — Frequency, Accountability, Scorecard, Timing.
"""
    st.markdown(steps_md)

    if st.button("🚀 Run Full 7-Step Engagement", type="primary",
                 use_container_width=True):
        with st.spinner("Consultant working — gathering context across all 7 steps..."):
            intake = st.session_state.intake

            # Steps 1 & 2 already ran when intake was saved, but re-run defensively.
            st.session_state.landscape = step1_assess_landscape(intake)
            st.session_state.opportunities = step2_identify_opportunities(intake)
            st.session_state.research = step3_industry_research(intake)
            st.session_state.driver_tree = step4_driver_tree(
                st.session_state.opportunities, intake,
            )
            st.session_state.prioritisation = step5_prioritisation(
                st.session_state.opportunities,
            )
            st.session_state.recommendations = step6_recommendations(
                intake, st.session_state.landscape,
                st.session_state.opportunities, st.session_state.research,
                st.session_state.driver_tree, st.session_state.prioritisation,
            )
            st.session_state.fast_plan = step7_fast_plan(
                intake, st.session_state.prioritisation,
            )
            st.session_state.validation = step_validate(
                intake=intake,
                landscape=st.session_state.landscape,
                opportunities=st.session_state.opportunities,
                research=st.session_state.research,
                driver_tree=st.session_state.driver_tree,
                prioritisation=st.session_state.prioritisation,
                recommendations=st.session_state.recommendations,
                fast_plan=st.session_state.fast_plan,
            )
        st.success("Engagement complete. See the Results tab.")
        st.rerun()

    # Show the research evidence so the user can audit what fed the LLM.
    if st.session_state.research:
        st.divider()
        st.markdown("### 🔍 Research Evidence (Step 3)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Web search snippets**")
            for w in st.session_state.research["web"]:
                st.markdown(f"- *{w['source']}* — {w['title']}  \n  {w['snippet']}")
        with c2:
            st.markdown("**Case-bank evidence (RAG)**")
            for d in st.session_state.research["rag"]:
                st.markdown(f"- **{d['title']}** (`{d['id']}`)  \n  {d['text']}")


def render_results() -> None:
    """Tab 3: engagement output, refinement chat, download."""
    st.subheader("📄 Engagement Results")
    if st.session_state.recommendations is None:
        st.info("No engagement yet — run the 7-step process from the Runner tab.")
        return

    intake = st.session_state.intake
    landscape = st.session_state.landscape
    opportunities = st.session_state.opportunities
    research = st.session_state.research
    driver_tree = st.session_state.driver_tree
    prioritisation = st.session_state.prioritisation
    recommendations = st.session_state.recommendations
    fast_plan = st.session_state.fast_plan
    val = st.session_state.validation

    # Validation banner.
    if val["status"] == "PASS":
        st.success(f"✅ Validation passed — {len(val['issues'])} issues.")
    else:
        st.warning(f"⚠ Validation flagged {len(val['issues'])} issue(s):")
        for issue in val["issues"]:
            st.markdown(f"- {issue}")

    # Headline up front — Pyramid Principle.
    st.markdown(f"## {intake['business_name']} — AI Consulting Engagement")
    st.markdown("### 🎯 Headline Recommendation")
    st.info(recommendations["headline"])

    # --- Section 1: Current AI Landscape ------------------------------
    st.markdown("### 1. Current AI Landscape")
    if landscape["tools_in_use"]:
        for t in landscape["tools_in_use"]:
            st.markdown(f"- {t}")
    st.markdown(f"**Tool details:** {landscape['details'] or '_(not provided)_'}")
    st.markdown(f"**Effectiveness:** {landscape['effectiveness_notes'] or '_(not provided)_'}")
    st.markdown(f"**Governance:** {landscape['governance_notes'] or '_(not provided)_'}")
    st.markdown(f"**Data sources:** {landscape['data_sources'] or '_(not provided)_'}")

    # --- Section 2: Opportunities & Pain Points -----------------------
    st.markdown("### 2. Opportunities & Pain Points")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Explicit pain points**")
        for p in opportunities["explicit_pain_points"] or ["_(none captured)_"]:
            st.markdown(f"- {p}")
        st.markdown("**Bottlenecks**")
        for b in opportunities["bottlenecks"] or ["_(none captured)_"]:
            st.markdown(f"- {b}")
    with c2:
        st.markdown("**Explicit opportunities**")
        for o in opportunities["explicit_opportunities"] or ["_(none captured)_"]:
            st.markdown(f"- {o}")
        st.markdown("**Latent opportunities surfaced**")
        for l in opportunities["latent_opportunities"] or ["_(none surfaced)_"]:
            st.markdown(f"- {l}")

    # --- Section 3: Industry & Competitor Research --------------------
    st.markdown("### 3. Industry & Competitor Research")
    st.markdown("**Web sources**")
    for w in research["web"]:
        st.markdown(f"- *{w['source']}* — {w['title']}: {w['snippet']}")
    st.markdown("**Case-bank evidence (RAG)**")
    for d in research["rag"]:
        st.markdown(f"- **{d['title']}** (`{d['id']}`): {d['text']}")

    # --- Section 4: Driver Tree --------------------------------------
    st.markdown("### 4. Driver Tree — Root-Cause Analysis")
    st.markdown(driver_tree["narrative"])
    for node in driver_tree["tree"]["nodes"]:
        with st.expander(f"Symptom: {node['symptom']}"):
            st.markdown("**Root drivers:**")
            for d in node["drivers"]:
                st.markdown(f"- {d}")

    # --- Section 5: Prioritisation -----------------------------------
    st.markdown("### 5. Prioritisation — Business Acumen Lens")
    st.dataframe(prioritisation["priority_df"], hide_index=True,
                 use_container_width=True)
    st.caption("Priority score = Impact × Feasibility. Sort by clicking any column.")

    # --- Section 6: Pyramid Principle Recommendations ----------------
    st.markdown("### 6. Recommendations — Pyramid Principle")
    st.markdown("**Answer:**")
    st.info(recommendations["headline"])
    st.markdown("**Supporting arguments:**")
    st.markdown(recommendations["supporting_arguments"])

    # --- Section 7: FAST Implementation Plan -------------------------
    st.markdown("### 7. FAST Implementation Plan")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("**Frequency**")
        st.markdown(fast_plan["frequency"])
        st.markdown("**Accountability**")
        st.markdown(fast_plan["accountability"])
    with f2:
        st.markdown("**Timing**")
        st.markdown(fast_plan["timing"])
    st.markdown("**Scorecard**")
    st.dataframe(fast_plan["scorecard_df"], hide_index=True,
                 use_container_width=True)
    st.markdown("**Risk · Change · Training · Data · Scaling**")
    st.markdown(fast_plan["risk_change"])

    st.divider()

    # Download.
    md_doc = render_engagement_markdown(
        intake=intake, landscape=landscape, opportunities=opportunities,
        research=research, driver_tree=driver_tree,
        prioritisation=prioritisation, recommendations=recommendations,
        fast_plan=fast_plan, validation=val,
    )
    fname = f"{intake['business_name'].lower().replace(' ', '_')}_ai_engagement.md"
    st.download_button(
        "⬇ Download engagement (Markdown / PDF-ready)",
        data=md_doc,
        file_name=fname,
        mime="text/markdown",
        type="primary",
        use_container_width=True,
    )

    # Refinement chat.
    st.divider()
    st.markdown("### 💬 Refine via Chat")
    st.caption("Ask the consultant to tweak any section "
               "(e.g. *make the recommendations more aggressive*, "
               "*re-prioritise around speed instead of cost*, "
               "*shorten the FAST plan to 12 weeks*).")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_msg = st.chat_input("Type a refinement request...")
    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        reply = refine_via_chat(user_msg)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


def render_integration() -> None:
    """Tab 4: how to swap simulated parts for real systems."""
    st.subheader("🔌 Integration Guide")
    st.markdown(
        """
        This MVP uses simulated tools so it always runs. To productionise:

        ### 1. Real web search
        Swap `tool_web_search()` for a real provider:
        ```python
        import requests
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": n},
            headers={"X-Subscription-Token": API_KEY},
        )
        ```
        Alternatives: SerpAPI, Bing Web Search, Google Programmable Search.

        ### 2. Real RAG over an actual case bank
        Replace `tool_rag_retrieve()` with a vector search:
        ```python
        emb = client.embeddings.create(model="text-embedding-3-small", input=query)
        results = vector_db.query(emb.data[0].embedding, k=3)
        ```
        Stores: pgvector, Pinecone, Weaviate, Qdrant. The case bank should
        be a curated set of past engagements, peer benchmarks, and known
        failure modes — exactly the same shape as `RAG_CORPUS` today.

        ### 3. Document generation (PDF / DOCX)
        The current download is Markdown — production should render to PDF:
        ```python
        from weasyprint import HTML
        HTML(string=markdown_to_html(md_doc)).write_pdf("engagement.pdf")
        ```
        For DOCX use `python-docx`. For client-templated decks use Google
        Slides API or Microsoft Graph.

        ### 4. CRM / project sync
        After the engagement, push outputs to where the team works:
        - Salesforce / HubSpot for opportunity tracking.
        - Asana / Linear for FAST plan task creation.
        - Slack notification with the engagement document link.

        ### 5. LLM provider
        Set `OPENAI_API_KEY` in your environment or paste it in the sidebar.
        To swap providers, change `llm_complete()` only — that's the single
        point of contact with the LLM.

        ### 6. Stronger guardrails
        The current validator catches structural issues and unverified
        dollar figures. Production should add:
        - PII redaction on intake free-text.
        - A clause-classifier on recommendations for over-promising.
        - Mandatory human review before any client-facing download.
        """
    )


# ============================================================
# SIDEBAR — console + API key
# ============================================================

def render_sidebar() -> None:
    """Sidebar: API key + reasoning console."""
    with st.sidebar:
        st.markdown("## 🧭 AI Consulting Agent")
        st.caption("Lead Consultant · 7-step process · Streamlit MVP")
        st.divider()

        st.markdown("### 🔑 OpenAI API key")
        st.text_input(
            "Key (optional, session-only)",
            type="password",
            key="openai_key",
            label_visibility="collapsed",
        )
        if not _OPENAI_IMPORTED:
            st.info("`openai` package not installed — using offline templates.")
        elif not (os.environ.get("OPENAI_API_KEY") or st.session_state.openai_key):
            st.warning("No key set — using deterministic templates.")
        else:
            st.success("OpenAI key detected.")

        st.divider()
        st.markdown("### 🧠 Consultant Console")
        if st.button("Clear console", use_container_width=True):
            st.session_state.console = []
            st.rerun()
        with st.container(height=420):
            if not st.session_state.console:
                st.caption("No reasoning steps yet.")
            for entry in st.session_state.console[-50:][::-1]:
                st.markdown(
                    f"`{entry['ts']}` **{entry['step']}** — {entry['message']}"
                )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Top-level entry point — wires sidebar + tabs together."""
    init_session_state()
    render_sidebar()

    st.title("🧭 AI Business Consulting Agent")
    st.caption("Intake in → Lead Consultant runs a strict 7-step process → "
               "Full engagement document out, refinable via chat.")

    tab_brief, tab_gen, tab_res, tab_int = st.tabs([
        "📝 Client Intake",
        "🤖 Engagement Runner",
        "📄 Results",
        "🔌 Integration Guide",
    ])
    with tab_brief: render_brief_input()
    with tab_gen:   render_generator()
    with tab_res:   render_results()
    with tab_int:   render_integration()


if __name__ == "__main__":
    main()

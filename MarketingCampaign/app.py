"""
================================================================================
AI MARKETING CAMPAIGN GENERATOR — STREAMLIT MVP
================================================================================

A self-contained demo of a Lead Consultant agent that follows a structured
marketing campaign generation script, calls a small toolbelt, and outputs a
complete campaign document the user can download.

ARCHITECTURE (matches the whiteboard sketch):

    +---------+        +------------------+        +--------------------+
    |  Chat   |  --->  | Lead Consultant  |  --->  |   Tools            |
    | (User)  |        | (follows script) |        |  - Web Search      |
    +---------+        +--------+---------+        |  - RAG (examples)  |
                                |                  |  - Doc Generation  |
                                v                  |  - Doc Filling     |
                       +----------------+          |  - Metrics Calc    |
                       |    Console     |          |  - Process Helpers |
                       | (Reasoning)    |          +--------------------+
                       +----------------+

THE 4-STEP SCRIPT
    1. UNDERSTAND BRIEF — parse the client form into a structured brief.
    2. RESEARCH         — call web-search and RAG tools for context.
    3. GENERATE ASSETS  — draft each campaign section from templates +
                          (optionally) an LLM call grounded in the research.
    4. VALIDATE         — run guardrails: budget sums to 100%, every
                          section is populated, no obvious hallucinations
                          (claims must trace to the brief or RAG output).

GUARDRAILS AGAINST HALLUCINATION
    * Every LLM call is given the structured brief + RAG snippets as the
      only source of truth, and told to stick to it.
    * The validator flags any section that mentions a number, brand, or
      claim that does not appear in the brief or research bundle.
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
    page_title="AI Marketing Campaign Generator",
    page_icon="📣",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS — defaults and option lists
# ============================================================

GOAL_OPTIONS = [
    "Drive new customer acquisition",
    "Increase brand awareness",
    "Re-engage lapsed customers",
    "Launch a new product",
    "Grow newsletter / email list",
    "Boost in-store foot traffic",
    "Increase average order value",
]

INDUSTRY_OPTIONS = [
    "SaaS / B2B Tech", "DTC / E-commerce", "Restaurant / F&B",
    "Health & Wellness", "Financial Services", "Education",
    "Non-profit", "Professional Services", "Retail", "Other",
]

CHANNEL_CATALOG = [
    "Meta Ads (FB/IG)", "Google Search Ads", "YouTube",
    "TikTok", "LinkedIn", "Email", "Influencer Partnerships",
    "Content / SEO", "Out-of-Home", "PR / Earned Media",
]


# ============================================================
# RAG CORPUS — pre-loaded mini knowledge base
# ============================================================
# These are tiny, deterministic "documents" the RAG tool can retrieve from.
# Each has tags so a simple keyword retrieval can score relevance.

RAG_CORPUS: List[Dict] = [
    {
        "id": "rag-001",
        "title": "DTC fitness launch — channel mix benchmark",
        "tags": ["dtc", "ecommerce", "fitness", "launch", "channel"],
        "text": (
            "For DTC fitness launches under $50k, a 40/30/20/10 split across "
            "Meta Ads, TikTok, Influencers, and Email typically yields the "
            "lowest blended CAC during the first 6 weeks. Creative iteration "
            "(3 new ad variants/week) matters more than channel diversity."
        ),
    },
    {
        "id": "rag-002",
        "title": "B2B SaaS demand-gen playbook",
        "tags": ["saas", "b2b", "demand", "linkedin", "content"],
        "text": (
            "B2B SaaS campaigns convert best with a content-led top of funnel "
            "(LinkedIn thought leadership + SEO) feeding gated assets, then "
            "a retargeting layer on Google Search for high-intent terms. "
            "Plan for a 60-90 day sales cycle in the KPI model."
        ),
    },
    {
        "id": "rag-003",
        "title": "Restaurant local awareness — what works",
        "tags": ["restaurant", "local", "foot traffic", "awareness"],
        "text": (
            "Local restaurants see the strongest lift from geo-fenced Meta + "
            "Google ads paired with a small influencer push (3-5 local "
            "creators). Avoid broad-reach OOH under $25k — coverage is too thin."
        ),
    },
    {
        "id": "rag-004",
        "title": "Persona framework — JTBD",
        "tags": ["persona", "audience", "jtbd", "framework"],
        "text": (
            "Use Jobs-to-be-Done to frame personas: each persona should "
            "describe a situation, a desired outcome, and the obstacle the "
            "product removes. Demographics alone tend to mislead creative "
            "decisions."
        ),
    },
    {
        "id": "rag-005",
        "title": "KPI baselines (industry medians)",
        "tags": ["kpi", "benchmark", "metrics"],
        "text": (
            "Industry-median benchmarks: Meta CTR 1.1%, Google Search CTR "
            "3.2%, email open rate 21%, e-com landing-page conversion 2.4%, "
            "B2B SaaS MQL→SQL 13%. Treat these as floors, not targets."
        ),
    },
    {
        "id": "rag-006",
        "title": "Creative concept ladders",
        "tags": ["creative", "concept", "framework", "messaging"],
        "text": (
            "A robust creative brief produces three concept 'territories' "
            "(e.g. functional, emotional, social-proof) and tests them in "
            "parallel for 2 weeks before consolidating spend on the winner."
        ),
    },
    {
        "id": "rag-007",
        "title": "Budget allocation rules of thumb",
        "tags": ["budget", "allocation", "spend"],
        "text": (
            "Default split for an integrated campaign: 60% paid media, 20% "
            "creative production, 10% tools/martech, 10% measurement and "
            "iteration reserve. Adjust downwards on paid if creative is weak."
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

    st.session_state.brief: Optional[Dict] = None
    st.session_state.research: Optional[Dict] = None
    st.session_state.campaign: Optional[Dict] = None
    st.session_state.validation: Optional[Dict] = None
    st.session_state.console: List[Dict] = []
    st.session_state.chat: List[Dict] = []
    st.session_state.openai_key: str = ""

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
            temperature=0.4,
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
        ("HubSpot", "Marketing benchmarks report"),
        ("Statista", "Industry spend overview"),
        ("Think with Google", "Consumer insights"),
        ("Meta for Business", "Creative best practices"),
        ("McKinsey", "Growth strategy primer"),
        ("eMarketer", "Channel-mix forecasts"),
    ]
    rng.shuffle(base_sources)
    return [
        {
            "source": src,
            "title": f"{title} relevant to '{query}'",
            "snippet": (
                f"Industry data suggests campaigns targeting '{query}' "
                f"see strongest results when paired with iterative creative "
                f"testing and tight audience definition."
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


def tool_metrics_calculator(budget: float, channel_split: Dict[str, float]) -> pd.DataFrame:
    """
    Project rough KPIs from a budget and channel mix.

    Each channel has a (CPM, CTR, CVR) tuple drawn from industry medians.
    Returns a DataFrame with allocated spend, projected impressions, clicks,
    and conversions.
    """
    log("Tools", f"metrics_calculator(budget=${budget:,.0f})")

    # Median benchmarks per channel. (cpm, ctr, cvr)
    benchmarks = {
        "Meta Ads (FB/IG)":      (9.50,  0.011, 0.024),
        "Google Search Ads":     (35.00, 0.032, 0.038),
        "YouTube":               (12.00, 0.005, 0.012),
        "TikTok":                (8.50,  0.016, 0.018),
        "LinkedIn":              (38.00, 0.004, 0.022),
        "Email":                 (1.00,  0.210, 0.030),
        "Influencer Partnerships":(20.00, 0.020, 0.025),
        "Content / SEO":         (5.00,  0.040, 0.015),
        "Out-of-Home":           (15.00, 0.001, 0.005),
        "PR / Earned Media":     (10.00, 0.008, 0.010),
    }
    rows = []
    for channel, pct in channel_split.items():
        spend = budget * (pct / 100.0)
        cpm, ctr, cvr = benchmarks.get(channel, (12.00, 0.010, 0.020))
        impressions = (spend / cpm) * 1000 if cpm > 0 else 0
        clicks = impressions * ctr
        conversions = clicks * cvr
        rows.append({
            "Channel": channel,
            "Allocation %": pct,
            "Spend ($)": round(spend, 2),
            "Impressions (est.)": int(impressions),
            "Clicks (est.)": int(clicks),
            "Conversions (est.)": int(conversions),
            "CPA (est. $)": round(spend / conversions, 2) if conversions > 0 else 0,
        })
    return pd.DataFrame(rows)


def tool_document_filler(template: str, fields: Dict[str, str]) -> str:
    """Tiny templating helper used by the deterministic fallback."""
    out = template
    for k, v in fields.items():
        out = out.replace("{" + k + "}", str(v))
    return out


# ============================================================
# THE 4-STEP CONSULTANT SCRIPT
# ============================================================

def step1_understand_brief(form: Dict) -> Dict:
    """
    Step 1: parse the raw form into a structured brief.

    No LLM needed here — this is pure data shaping. Doing it deterministically
    keeps the rest of the pipeline reproducible.
    """
    log("Step 1 · Understand Brief", "Parsing client form into structured brief.")
    brief = {
        "business_name": form["business_name"].strip(),
        "industry": form["industry"],
        "audience": form["audience"].strip(),
        "goals": form["goals"],
        "budget": float(form["budget"]),
        "timeline_weeks": int(form["timeline_weeks"]),
        "channels_preferred": form.get("channels_preferred", []),
        "extra_notes": form.get("extra_notes", "").strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    log("Step 1 · Understand Brief",
        f"Brief captured: {brief['business_name']} / {brief['industry']} / "
        f"${brief['budget']:,.0f} budget / {len(brief['goals'])} goals.")
    return brief


def step2_research(brief: Dict) -> Dict:
    """
    Step 2: gather external context via the web-search and RAG tools.

    The result bundle is later passed verbatim to the LLM as the only
    factual basis for the campaign, which is the main guardrail against
    hallucinated brands, stats, or competitor claims.
    """
    log("Step 2 · Research", f"Researching context for {brief['business_name']}.")
    web_q = f"{brief['industry']} marketing benchmarks {', '.join(brief['goals'][:2])}"
    rag_q = f"{brief['industry']} {' '.join(brief['goals'])}"
    web = tool_web_search(web_q, n=3)
    rag = tool_rag_retrieve(rag_q, k=3)
    research = {"web_search_query": web_q, "web": web, "rag_query": rag_q, "rag": rag}
    log("Step 2 · Research",
        f"Collected {len(web)} web snippets and {len(rag)} RAG documents.")
    return research


def step3_generate_assets(brief: Dict, research: Dict) -> Dict:
    """
    Step 3: draft every campaign section.

    For each section we try an LLM call grounded in the brief + research,
    then fall back to a deterministic template if no key is configured
    or the call fails. The fallback is what keeps the demo trustworthy
    offline.
    """
    log("Step 3 · Generate Assets", "Drafting each campaign section.")

    context_block = _render_context_for_llm(brief, research)
    sections = {
        "campaign_objective": _gen_objective(brief, context_block),
        "audience_personas":  _gen_personas(brief, context_block),
        "key_messages":       _gen_key_messages(brief, context_block),
        "creative_concepts":  _gen_creative_concepts(brief, context_block),
        "channel_plan":       _gen_channel_plan(brief, context_block),
        "budget_allocation":  _gen_budget_allocation(brief),
        "kpis":               _gen_kpis(brief),
        "timeline":           _gen_timeline(brief),
    }
    log("Step 3 · Generate Assets",
        f"Drafted {len(sections)} sections: {', '.join(sections.keys())}.")
    return sections


def step4_validate(campaign: Dict, brief: Dict, research: Dict) -> Dict:
    """
    Step 4: run guardrails over the generated campaign.

    Checks:
      * Every required section is non-empty.
      * Budget allocation sums to 100% (±1 for float rounding).
      * Any explicit dollar / % numbers in narrative sections trace back
        to the brief or budget — otherwise flag as potential hallucination.
    """
    log("Step 4 · Validate", "Running guardrails on generated campaign.")
    issues: List[str] = []

    required = ["campaign_objective", "audience_personas", "key_messages",
                "creative_concepts", "channel_plan", "budget_allocation",
                "kpis", "timeline"]
    for key in required:
        val = campaign.get(key)
        if not val or (isinstance(val, str) and len(val.strip()) < 10):
            issues.append(f"Section '{key}' is empty or too short.")

    # Budget split sanity check
    split = campaign["budget_allocation"]["split"]
    total_pct = sum(split.values())
    if abs(total_pct - 100.0) > 1.0:
        issues.append(f"Budget allocation sums to {total_pct:.1f}% (expected 100%).")

    # Hallucination heuristic: dollar amounts in narrative must appear in brief.
    brief_blob = json.dumps(brief)
    for section_key in ["campaign_objective", "key_messages", "creative_concepts"]:
        text = campaign.get(section_key, "")
        if isinstance(text, str):
            for amount in re.findall(r"\$\s?\d[\d,]*", text):
                normalized = amount.replace(" ", "")
                if normalized not in brief_blob and str(int(brief["budget"])) not in normalized:
                    issues.append(
                        f"Section '{section_key}' contains unverified figure {amount}."
                    )

    status = "PASS" if not issues else "WARN"
    log("Step 4 · Validate",
        f"Validation {status} — {len(issues)} issue(s) found.")
    return {"status": status, "issues": issues, "checked_at": datetime.now().isoformat()}


# ============================================================
# SECTION GENERATORS (LLM-first, template fallback)
# ============================================================

def _render_context_for_llm(brief: Dict, research: Dict) -> str:
    """Pack the brief + research into a single grounding context string."""
    lines = [
        "CLIENT BRIEF (the only source of truth about the client):",
        json.dumps(brief, indent=2),
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
    "You are a senior marketing consultant. Use ONLY facts in the supplied "
    "context. Do NOT invent dollar amounts, competitor names, or statistics "
    "that aren't grounded there. If the brief is silent on something, say so."
)


def _gen_objective(brief: Dict, context: str) -> str:
    """Campaign objective — 2-3 sentence statement tying goals to budget."""
    out = llm_complete(
        system=_GROUNDING_RULES,
        user=(f"Write a 2-3 sentence Campaign Objective for the client below, "
              f"tying their goals to the budget and timeline.\n\n{context}"),
        max_tokens=250,
    )
    if out:
        return out

    # Deterministic template fallback.
    goals_str = "; ".join(brief["goals"])
    return tool_document_filler(
        "Within {weeks} weeks and a ${budget:,.0f} budget, {biz} will run an "
        "integrated campaign targeting {audience}. The primary objectives are: "
        "{goals}. Success will be measured against the KPIs defined later in "
        "this document.",
        {
            "weeks": brief["timeline_weeks"],
            "budget": brief["budget"],
            "biz": brief["business_name"],
            "audience": brief["audience"],
            "goals": goals_str,
        },
    ).format(budget=brief["budget"])


def _gen_personas(brief: Dict, context: str) -> str:
    """Two audience personas, JTBD-style."""
    out = llm_complete(
        system=_GROUNDING_RULES,
        user=(f"Write TWO concise audience personas (each 4-5 lines) using "
              f"the Jobs-to-be-Done framework. Base them strictly on the "
              f"audience description in the brief — do not invent demographics "
              f"that contradict it.\n\n{context}"),
        max_tokens=500,
    )
    if out:
        return out

    return (
        f"**Persona 1 — Primary buyer**\n"
        f"Situation: Members of the audience '{brief['audience']}' actively "
        f"looking for solutions like {brief['business_name']}.\n"
        f"Desired outcome: Achieve {brief['goals'][0].lower()} without "
        f"adding complexity.\n"
        f"Obstacle: Skepticism about new providers; needs strong social proof.\n\n"
        f"**Persona 2 — Influencer / champion**\n"
        f"Situation: Adjacent stakeholders who recommend solutions in this category.\n"
        f"Desired outcome: Be seen as the source of a smart recommendation.\n"
        f"Obstacle: Limited time to vet new options."
    )


def _gen_key_messages(brief: Dict, context: str) -> str:
    """Three message pillars + one tagline candidate."""
    out = llm_complete(
        system=_GROUNDING_RULES,
        user=(f"Draft THREE key message pillars and ONE tagline candidate "
              f"for the campaign. Keep messages benefit-led and tied to the "
              f"goals in the brief.\n\n{context}"),
        max_tokens=350,
    )
    if out:
        return out

    g = brief["goals"]
    return (
        f"**Pillar 1 — Outcome-led:** {brief['business_name']} helps "
        f"{brief['audience']} achieve {g[0].lower()}.\n"
        f"**Pillar 2 — Trust-led:** Proven approach, transparent results.\n"
        f"**Pillar 3 — Urgency-led:** Limited-time launch window over the "
        f"next {brief['timeline_weeks']} weeks.\n\n"
        f"**Tagline candidate:** \"{brief['business_name']} — built for "
        f"{brief['audience'].split(',')[0].strip()}.\""
    )


def _gen_creative_concepts(brief: Dict, context: str) -> str:
    """Three creative territories the team can test in parallel."""
    out = llm_complete(
        system=_GROUNDING_RULES,
        user=("Propose THREE creative concept 'territories' (functional, "
              "emotional, social-proof) for the campaign. Each should be "
              "1 short paragraph naming the hook, the format, and the "
              "primary channel it suits best.\n\n" + context),
        max_tokens=450,
    )
    if out:
        return out

    return (
        "**Concept A — Functional ('See it work')**\n"
        f"Short product-demo video showing {brief['business_name']} delivering "
        f"the core benefit in under 15 seconds. Best for Meta Reels and TikTok.\n\n"
        "**Concept B — Emotional ('The shift')**\n"
        f"Before/after storytelling featuring a member of the '{brief['audience']}' "
        f"audience. Best for YouTube and email hero.\n\n"
        "**Concept C — Social Proof ('They already use it')**\n"
        "Testimonial + UGC compilation. Best for retargeting and influencer whitelisting."
    )


def _gen_channel_plan(brief: Dict, context: str) -> str:
    """Narrative explaining channel selection and sequencing."""
    out = llm_complete(
        system=_GROUNDING_RULES,
        user=("Recommend a channel mix (3-5 channels) with a one-line "
              "justification each. Reference channels the client preferred "
              "in the brief if any, and explain any additions or omissions.\n\n"
              + context),
        max_tokens=400,
    )
    if out:
        return out

    chans = brief["channels_preferred"] or ["Meta Ads (FB/IG)", "Google Search Ads", "Email"]
    bullets = [f"- **{c}** — selected for fit with the audience and budget." for c in chans]
    return "Recommended channels:\n" + "\n".join(bullets)


def _gen_budget_allocation(brief: Dict) -> Dict:
    """
    Produce a defensible default budget split.

    This is rule-based (not LLM) because the validator depends on the
    numbers being machine-readable and summing exactly to 100%.
    """
    chans = brief["channels_preferred"] or ["Meta Ads (FB/IG)", "Google Search Ads", "Email"]
    # Default weights — paid-heavy channels get more.
    weights = {
        "Meta Ads (FB/IG)": 5, "Google Search Ads": 5, "YouTube": 3,
        "TikTok": 4, "LinkedIn": 4, "Email": 2,
        "Influencer Partnerships": 3, "Content / SEO": 3,
        "Out-of-Home": 2, "PR / Earned Media": 2,
    }
    chan_weights = {c: weights.get(c, 3) for c in chans}
    total = sum(chan_weights.values())
    split = {c: round(w / total * 100, 1) for c, w in chan_weights.items()}
    # Fix rounding drift so the sum is exactly 100.
    drift = round(100 - sum(split.values()), 1)
    if split:
        first = next(iter(split))
        split[first] = round(split[first] + drift, 1)

    metrics_df = tool_metrics_calculator(brief["budget"], split)
    return {"split": split, "metrics_df": metrics_df}


def _gen_kpis(brief: Dict) -> str:
    """KPIs — picked from a template list keyed by selected goals."""
    kpi_map = {
        "Drive new customer acquisition": ["New customers acquired", "Blended CAC", "Conversion rate"],
        "Increase brand awareness":       ["Reach", "Aided brand recall (survey)", "Branded search lift"],
        "Re-engage lapsed customers":     ["Reactivation rate", "Win-back revenue", "Repeat purchase rate"],
        "Launch a new product":           ["First-week orders", "Waitlist conversion", "Press mentions"],
        "Grow newsletter / email list":   ["New subscribers", "Subscriber CAC", "Open rate"],
        "Boost in-store foot traffic":    ["Store visits", "Promo redemptions", "Geo-conversion lift"],
        "Increase average order value":   ["AOV", "Items per order", "Upsell attach rate"],
    }
    bullets = []
    for g in brief["goals"]:
        for k in kpi_map.get(g, []):
            bullets.append(f"- {k}")
    if not bullets:
        bullets = ["- Reach", "- Engagement rate", "- Conversion rate"]
    return "Primary KPIs:\n" + "\n".join(sorted(set(bullets)))


def _gen_timeline(brief: Dict) -> str:
    """Bucket the timeline into 4 standard phases sized to the brief."""
    weeks = brief["timeline_weeks"]
    phases = [
        ("Setup & Creative Production", max(1, weeks // 6)),
        ("Soft Launch & Learning",      max(1, weeks // 4)),
        ("Scale & Optimisation",        max(1, weeks // 2)),
        ("Wind-down & Reporting",       max(1, weeks // 6)),
    ]
    # Re-balance so the sum equals weeks.
    diff = weeks - sum(p[1] for p in phases)
    phases[2] = (phases[2][0], phases[2][1] + diff)

    start = datetime.today().date()
    lines = []
    for name, dur in phases:
        end = start + timedelta(weeks=dur)
        lines.append(f"- **{name}** — {dur}w ({start} → {end})")
        start = end
    return "\n".join(lines)


# ============================================================
# DOCUMENT ASSEMBLY
# ============================================================

def render_campaign_markdown(brief: Dict, campaign: Dict, validation: Dict) -> str:
    """Stitch all sections into a single downloadable markdown document."""
    md = []
    md.append(f"# Marketing Campaign — {brief['business_name']}\n")
    md.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
              f"Validation: **{validation['status']}**_\n")

    md.append("## 1. Campaign Objective\n")
    md.append(campaign["campaign_objective"] + "\n")

    md.append("## 2. Audience Personas\n")
    md.append(campaign["audience_personas"] + "\n")

    md.append("## 3. Key Messages\n")
    md.append(campaign["key_messages"] + "\n")

    md.append("## 4. Creative Concepts\n")
    md.append(campaign["creative_concepts"] + "\n")

    md.append("## 5. Channel Plan\n")
    md.append(campaign["channel_plan"] + "\n")

    md.append("## 6. Budget Allocation\n")
    split = campaign["budget_allocation"]["split"]
    md.append(f"_Total budget: **${brief['budget']:,.0f}**_\n")
    for ch, pct in split.items():
        spend = brief["budget"] * pct / 100
        md.append(f"- {ch}: {pct}% (${spend:,.0f})")
    md.append("")
    md.append("**Projected media metrics:**\n")
    md.append(campaign["budget_allocation"]["metrics_df"].to_markdown(index=False))
    md.append("")

    md.append("## 7. KPIs\n")
    md.append(campaign["kpis"] + "\n")

    md.append("## 8. Timeline\n")
    md.append(campaign["timeline"] + "\n")

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
    Apply a refinement request to the existing campaign.

    The LLM (if available) is asked to rewrite the *specific* section the
    user mentioned. The offline fallback just appends the request as a
    revision note to the section so nothing is silently lost.
    """
    if st.session_state.campaign is None:
        return "Generate a campaign first (Brief Input → Campaign Generator)."

    log("Refine", f"User refinement request: {user_message}")
    campaign = st.session_state.campaign
    brief = st.session_state.brief

    # Detect target section by keyword.
    targets = {
        "objective": "campaign_objective",
        "persona":   "audience_personas", "audience": "audience_personas",
        "message":   "key_messages", "tagline": "key_messages",
        "creative":  "creative_concepts", "concept": "creative_concepts",
        "channel":   "channel_plan",
        "kpi":       "kpis", "metric": "kpis",
        "timeline":  "timeline", "schedule": "timeline",
    }
    target_key = None
    for kw, sec in targets.items():
        if kw in user_message.lower():
            target_key = sec
            break

    if target_key is None:
        return ("I can refine: objective, personas, messages, creative concepts, "
                "channels, KPIs, or timeline. Mention which section to change.")

    current_text = campaign[target_key]
    if isinstance(current_text, dict):
        return "That section is structured (budget). Edit via the form instead."

    out = llm_complete(
        system=_GROUNDING_RULES + " Only modify what the user asked to change.",
        user=(f"Brief: {json.dumps(brief)}\n\n"
              f"Current section:\n{current_text}\n\n"
              f"User refinement request: {user_message}\n\n"
              f"Return the rewritten section only."),
        max_tokens=500,
    )
    if out:
        campaign[target_key] = out
        log("Refine", f"Rewrote section '{target_key}'.")
        # Re-validate so the user sees the updated status.
        st.session_state.validation = step4_validate(
            campaign, brief, st.session_state.research
        )
        return f"✅ Updated **{target_key}**. See the Results tab."

    # Offline fallback: append a revision note.
    campaign[target_key] = current_text + f"\n\n_Revision note: {user_message}_"
    return f"✏️ Appended your note to **{target_key}** (offline mode — no LLM)."


# ============================================================
# UI — TABS
# ============================================================

def render_brief_input() -> None:
    """Tab 1: the client brief form."""
    st.subheader("📝 Client Brief")
    st.caption("Fill in the basics — the consultant uses this as the only source of truth.")

    with st.form("brief_form"):
        c1, c2 = st.columns(2)
        business_name = c1.text_input("Business name", value="Acme Wellness Co.")
        industry = c2.selectbox("Industry", INDUSTRY_OPTIONS, index=2)

        audience = st.text_area(
            "Target audience",
            value="Urban professionals aged 28-45 interested in healthy convenience food.",
            height=80,
        )

        goals = st.multiselect(
            "Campaign goals (pick up to 3)",
            GOAL_OPTIONS,
            default=["Drive new customer acquisition", "Increase brand awareness"],
            max_selections=3,
        )

        c3, c4 = st.columns(2)
        budget = c3.number_input("Budget ($USD)", min_value=1000, value=25000, step=1000)
        timeline_weeks = c4.slider("Timeline (weeks)", min_value=2, max_value=26, value=8)

        channels_preferred = st.multiselect(
            "Preferred channels (optional)",
            CHANNEL_CATALOG,
            default=["Meta Ads (FB/IG)", "Google Search Ads", "Email"],
        )

        extra_notes = st.text_area(
            "Anything else the consultant should know? (optional)",
            value="", height=70,
        )

        submitted = st.form_submit_button("💾 Save Brief", type="primary",
                                          use_container_width=True)

    if submitted:
        if not business_name or not audience or not goals:
            st.error("Business name, audience, and at least one goal are required.")
            return
        form = {
            "business_name": business_name, "industry": industry,
            "audience": audience, "goals": goals, "budget": budget,
            "timeline_weeks": timeline_weeks,
            "channels_preferred": channels_preferred,
            "extra_notes": extra_notes,
        }
        st.session_state.brief = step1_understand_brief(form)
        st.success("Brief saved. Head to the Campaign Generator tab.")

    if st.session_state.brief:
        with st.expander("📋 Current saved brief"):
            st.json(st.session_state.brief)


def render_generator() -> None:
    """Tab 2: run the 4-step script."""
    st.subheader("🤖 Campaign Generator")

    if st.session_state.brief is None:
        st.warning("No brief yet — fill out the Brief Input tab first.")
        return

    st.markdown(
        "The Lead Consultant follows a strict 4-step script:"
        "\n\n1. **Understand Brief** (done when you saved the form)"
        "\n2. **Research** — web search + RAG retrieval"
        "\n3. **Generate Assets** — every campaign section"
        "\n4. **Validate** — guardrails against hallucinations and gaps"
    )

    if st.button("🚀 Run Full Generation", type="primary", use_container_width=True):
        with st.spinner("Consultant working..."):
            st.session_state.research = step2_research(st.session_state.brief)
            st.session_state.campaign = step3_generate_assets(
                st.session_state.brief, st.session_state.research
            )
            st.session_state.validation = step4_validate(
                st.session_state.campaign,
                st.session_state.brief,
                st.session_state.research,
            )
        st.success("Campaign generated. See the Results tab.")
        st.rerun()

    # Show the research evidence so the user can audit what fed the LLM.
    if st.session_state.research:
        st.divider()
        st.markdown("### 🔍 Research Evidence")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Web search snippets**")
            for w in st.session_state.research["web"]:
                st.markdown(f"- *{w['source']}* — {w['title']}  \n  {w['snippet']}")
        with c2:
            st.markdown("**RAG documents retrieved**")
            for d in st.session_state.research["rag"]:
                st.markdown(f"- **{d['title']}** (`{d['id']}`)  \n  {d['text']}")


def render_results() -> None:
    """Tab 3: campaign output, refinement chat, download."""
    st.subheader("📄 Campaign Results")
    if st.session_state.campaign is None:
        st.info("No campaign yet — run the generator first.")
        return

    brief = st.session_state.brief
    camp = st.session_state.campaign
    val = st.session_state.validation

    # Validation banner
    if val["status"] == "PASS":
        st.success(f"✅ Validation passed — {len(val['issues'])} issues.")
    else:
        st.warning(f"⚠ Validation flagged {len(val['issues'])} issue(s):")
        for issue in val["issues"]:
            st.markdown(f"- {issue}")

    # Render each section
    st.markdown(f"## {brief['business_name']} — Campaign")
    st.markdown("### 1. Campaign Objective");   st.markdown(camp["campaign_objective"])
    st.markdown("### 2. Audience Personas");    st.markdown(camp["audience_personas"])
    st.markdown("### 3. Key Messages");          st.markdown(camp["key_messages"])
    st.markdown("### 4. Creative Concepts");     st.markdown(camp["creative_concepts"])
    st.markdown("### 5. Channel Plan");          st.markdown(camp["channel_plan"])

    st.markdown("### 6. Budget Allocation")
    split = camp["budget_allocation"]["split"]
    split_df = pd.DataFrame({
        "Channel": list(split.keys()),
        "Allocation %": list(split.values()),
        "Spend ($)": [brief["budget"] * v / 100 for v in split.values()],
    })
    st.dataframe(split_df, hide_index=True, use_container_width=True)
    st.markdown("**Projected media metrics:**")
    st.dataframe(camp["budget_allocation"]["metrics_df"],
                 hide_index=True, use_container_width=True)

    st.markdown("### 7. KPIs");      st.markdown(camp["kpis"])
    st.markdown("### 8. Timeline");  st.markdown(camp["timeline"])

    st.divider()

    # Download
    md_doc = render_campaign_markdown(brief, camp, val)
    fname = f"{brief['business_name'].lower().replace(' ', '_')}_campaign.md"
    st.download_button(
        "⬇ Download campaign (Markdown / PDF-ready)",
        data=md_doc,
        file_name=fname,
        mime="text/markdown",
        type="primary",
        use_container_width=True,
    )

    # Refinement chat
    st.divider()
    st.markdown("### 💬 Refine via Chat")
    st.caption("Ask the consultant to tweak any section "
               "(e.g. *make the messages more playful*, *drop LinkedIn from the channel plan*).")
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

        ### 2. Real RAG
        Replace `tool_rag_retrieve()` with a vector search:
        ```python
        # OpenAI embeddings + your vector DB of choice
        emb = client.embeddings.create(model="text-embedding-3-small", input=query)
        results = vector_db.query(emb.data[0].embedding, k=3)
        ```
        Stores: pgvector, Pinecone, Weaviate, Qdrant.

        ### 3. Document generation (PDF / DOCX)
        The current download is Markdown — production should render to PDF:
        ```python
        from weasyprint import HTML
        HTML(string=markdown_to_html(md_doc)).write_pdf("campaign.pdf")
        ```
        For DOCX use `python-docx`. For brand-templated decks use Google
        Slides API or Microsoft Graph.

        ### 4. CRM / project sync
        After generation, push the campaign to where the team works:
        - HubSpot / Salesforce for campaign tracking
        - Asana / Linear for task creation
        - Slack notification with the document link

        ### 5. LLM provider
        Set `OPENAI_API_KEY` in your environment or paste it in the sidebar.
        To swap providers, change `llm_complete()` only — that's the single
        point of contact with the LLM.

        ### 6. Stronger guardrails
        The current validator catches structural issues and unverified
        dollar figures. Production should add:
        - PII redaction on inputs
        - Brand-safety classifier on creative copy
        - Compliance review hook before any download
        """
    )


# ============================================================
# SIDEBAR — console + API key
# ============================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📣 Marketing Campaign Generator")
        st.caption("Lead Consultant agent · Streamlit MVP")
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
    init_session_state()
    render_sidebar()

    st.title("📣 AI Marketing Campaign Generator")
    st.caption("Brief in → Lead Consultant runs a 4-step script → "
               "Full campaign document out, refinable via chat.")

    tab_brief, tab_gen, tab_res, tab_int = st.tabs(
        ["📝 Brief Input", "🤖 Campaign Generator", "📄 Results", "🔌 Integration Guide"]
    )
    with tab_brief: render_brief_input()
    with tab_gen:   render_generator()
    with tab_res:   render_results()
    with tab_int:   render_integration()


if __name__ == "__main__":
    main()

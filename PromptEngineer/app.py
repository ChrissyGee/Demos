"""
===============================================================================
Prompt Engineering Assistant — Streamlit MVP
===============================================================================

An interactive guide for building high-quality AI prompts using a proven
3-step framework. Features a live prompt preview, a Stripe subscription
paywall (test mode), and AI-powered prompt polishing via a local LLM.

Available Modes (sidebar):
    Step-by-Step Guide          – Interactive 3-step framework + live preview
    Video Walkthrough           – Curated video resources + summary table
    Exercise with Example       – Worked example + user practice area

Dependencies:
    pip install streamlit pandas requests stripe

Configuration (st.secrets or environment variables):
    STRIPE_SECRET_KEY  – Stripe secret key  (test: sk_test_...)
    STRIPE_PRICE_ID    – Stripe subscription price ID  (price_...)
    APP_URL            – Your deployed app URL for Stripe redirects
                         (default: http://localhost:8501)

Usage:
    streamlit run app.py

Running tests:
    pytest app.py -v          # executes all test_* functions at the bottom

Author  : Anentra AI Consulting
Version : 1.0.0
===============================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import stripe

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prompt_engineer")

# ---------------------------------------------------------------------------
# App-level constants
# ---------------------------------------------------------------------------
APP_TITLE = "Prompt Engineering Assistant"
APP_SUBTITLE = "Build smarter AI prompts with a proven 3-step framework"

LLM_ENDPOINT = (
    "http://ec2-18-220-213-64.us-east-2.compute.amazonaws.com:8080/v1/chat/completions"
)
LLM_MODEL = "qwen2.5:7b"
LLM_TIMEOUT = 60  # seconds

# 3-step framework definition — drives both the guide UI and the meta-prompt
FRAMEWORK_STEPS: List[Dict] = [
    {
        "key": "step1",
        "number": 1,
        "title": "Tell the AI exactly what you want it to do",
        "desc": "Define the task or role clearly. Be direct and specific about the action required.",
        "placeholder": (
            "e.g., 'Write a compelling LinkedIn post that positions me as a "
            "thought leader in renewable energy for a mid-career audience…'"
        ),
    },
    {
        "key": "step2",
        "number": 2,
        "title": "Give a one-sentence summary of what you want",
        "desc": "Distil your goal into one clear sentence. This anchors the AI and keeps output focused.",
        "placeholder": (
            "e.g., 'I want to create content that drives engagement among "
            "remote workers aged 25–40 and positions our brand as a thought leader.'"
        ),
    },
    {
        "key": "step3",
        "number": 3,
        "title": "Give headings with bullet points explaining each heading",
        "desc": "Structure the expected output. List the sections the AI should cover.",
        "placeholder": (
            "**Introduction**\n"
            "- Hook the reader with a statistic or bold claim\n"
            "- State the core problem being solved\n\n"
            "**Main Body**\n"
            "- Key argument 1 with supporting evidence\n"
            "- Key argument 2 with a real-world example\n\n"
            "**Conclusion**\n"
            "- Summarise the main points\n"
            "- Clear call-to-action for the reader"
        ),
    },
]

# Worked example used in Exercise mode
EXAMPLE: Dict[str, str] = {
    "step1": "Write a compelling blog post about the benefits of remote work for modern professionals.",
    "step2": (
        "I want to inspire knowledge workers to embrace remote work by highlighting "
        "its productivity, wellness, and career benefits."
    ),
    "step3": (
        "**Introduction**\n"
        "- Open with a surprising remote work statistic\n"
        "- Briefly preview the three main benefits\n\n"
        "**Productivity Benefits**\n"
        "- No commute equals more deep-focus time\n"
        "- Flexible hours aligned to personal energy peaks\n"
        "- Fewer office interruptions and context switches\n\n"
        "**Wellness & Work-Life Balance**\n"
        "- More time for exercise, family, and recovery\n"
        "- Reduced commute stress\n"
        "- Ability to design an optimal home work environment\n\n"
        "**Career Advantages**\n"
        "- Access to a global job market\n"
        "- Builds autonomy and self-management skills\n"
        "- Diverse, portfolio-worthy projects\n\n"
        "**Conclusion**\n"
        "- Recap three key benefits\n"
        "- Call-to-action: Try one remote day this week"
    ),
    "notes": (
        "Tone: inspirational and professional. Audience: mid-career professionals (30–45). "
        "Length: ~800 words. Avoid jargon. Include at least two real statistics."
    ),
}

# Video resources — update URLs before deploying to production
VIDEOS: List[Dict[str, str]] = [
    {
        "title": "Prompt Engineering for Developers — DeepLearning.AI",
        "url": "https://www.youtube.com/watch?v=_Ah38JM8hik",
        "desc": (
            "Industry-standard short course covering chain-of-thought, few-shot prompting, "
            "and output-formatting techniques. Co-taught by OpenAI researchers."
        ),
        "duration": "~60 min",
        "level": "Intermediate",
    },
    {
        "title": "Master Prompt Engineering in 10 Minutes",
        "url": "https://www.youtube.com/watch?v=jC4v5AS4RIM",
        "desc": (
            "Fast-paced primer on the most impactful prompt patterns. "
            "Perfect for non-technical users who want immediate results."
        ),
        "duration": "~10 min",
        "level": "Beginner",
    },
    {
        "title": "Advanced Prompt Engineering Techniques",
        "url": "https://www.youtube.com/watch?v=dOxUroR57xs",
        "desc": (
            "Covers structured prompting, role assignment, system messages, "
            "and business-oriented use cases in depth."
        ),
        "duration": "~30 min",
        "level": "Advanced",
    },
    {
        "title": "How LLMs Work — Andrej Karpathy",
        "url": "https://www.youtube.com/watch?v=9gWFuqwFBiU",
        "desc": (
            "Understand how language models process prompts under the hood. "
            "Essential context for writing prompts that align with model mechanics."
        ),
        "duration": "~45 min",
        "level": "Technical",
    },
]

QUICK_TIPS: List[tuple[str, str]] = [
    ("Be Specific", "Vague prompts produce vague output. Specify who, what, format, and expected length."),
    ("Assign a Role", "'Act as a [role]…' primes the model's perspective and domain expertise."),
    ("Set the Format", "Tell the AI how to structure its output: list, table, JSON, paragraph, or code block."),
    ("One Goal Per Prompt", "Break complex tasks into sequential prompts for higher-quality results."),
    ("Show an Example", "'Write in the style of…' or 'Here is a sample output:' is extremely powerful."),
    ("Iterate", "Treat your first prompt as a rough draft. Refine based on what the model returns."),
]

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _secret(key: str, fallback: str = "") -> str:
    """Read a config value from st.secrets, falling back to env var then default."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, fallback)


def _app_url() -> str:
    return _secret("APP_URL", "http://localhost:8501")


# ---------------------------------------------------------------------------
# Pure prompt utilities  (no Streamlit calls — independently testable)
# ---------------------------------------------------------------------------


def assemble_prompt_preview(step1: str, step2: str, step3: str, notes: str) -> str:
    """
    Combine 3-step framework inputs and notes into a formatted prompt preview.

    Each non-empty section is rendered under a labelled heading, separated
    by horizontal rules. All inputs are stripped of leading/trailing whitespace.

    Args:
        step1: Task definition text from Step 1.
        step2: One-sentence goal text from Step 2.
        step3: Structured headings text from Step 3.
        notes: Additional context, constraints, or tone preferences.

    Returns:
        Multi-section formatted string, or a placeholder when all inputs are empty.
    """
    parts: list[str] = []
    if step1.strip():
        parts.append(f"TASK:\n{step1.strip()}")
    if step2.strip():
        parts.append(f"GOAL:\n{step2.strip()}")
    if step3.strip():
        parts.append(f"STRUCTURE:\n{step3.strip()}")
    if notes.strip():
        parts.append(f"ADDITIONAL CONTEXT:\n{notes.strip()}")

    if not parts:
        return "Your assembled prompt will appear here as you fill in the steps above."
    return "\n\n---\n\n".join(parts)


def validate_inputs(step1: str, step2: str, step3: str) -> dict[str, bool]:
    """
    Validate that all three required framework steps contain non-whitespace text.

    Args:
        step1, step2, step3: Text inputs from the three framework steps.

    Returns:
        Dict mapping 'step1'/'step2'/'step3' → True (valid) or False (empty/whitespace).
    """
    return {
        "step1": bool(step1.strip()),
        "step2": bool(step2.strip()),
        "step3": bool(step3.strip()),
    }


def build_meta_prompt(step1: str, step2: str, step3: str, notes: str) -> str:
    """
    Construct the meta-prompt sent to the LLM to generate a polished final prompt.

    The meta-prompt instructs the LLM to act as a prompt engineer and synthesise
    the structured user inputs into a single, production-ready prompt.

    Args:
        step1: Task definition.
        step2: One-sentence goal.
        step3: Structured headings.
        notes: Additional context or constraints.

    Returns:
        Complete meta-prompt string ready to send to the LLM endpoint.
    """
    notes_block = notes.strip() if notes.strip() else "None provided."
    return (
        "You are an expert prompt engineer. A user has provided structured inputs "
        "using a 3-step framework. Synthesise these into a single, polished, "
        "production-ready prompt that the user can paste directly into any AI assistant.\n\n"
        "Requirements for the final prompt:\n"
        "- Clear, specific, and unambiguous\n"
        "- Well-structured with appropriate markdown formatting\n"
        "- Professional and purposeful tone\n"
        "- Immediately usable without further editing\n\n"
        f"Step 1 — Task Definition:\n{step1}\n\n"
        f"Step 2 — One-Sentence Goal:\n{step2}\n\n"
        f"Step 3 — Structure & Headings:\n{step3}\n\n"
        f"Notes / Constraints:\n{notes_block}\n\n"
        "---\n\n"
        "Output ONLY the final polished prompt below. "
        "No preamble, no explanation, no meta-commentary."
    )


# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------


def call_llm(prompt: str) -> Optional[str]:
    """
    Send a prompt to the LLM server and return the generated text response.

    Uses the OpenAI-compatible /v1/chat/completions endpoint with Bearer auth.
    Surfaces user-friendly Streamlit error messages on all failure modes.

    Args:
        prompt: The fully assembled prompt string to send.

    Returns:
        Generated response text, or None if the call fails for any reason.
    """
    try:
        response = requests.post(
            LLM_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer demo",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        logger.error("LLM request timed out after %d seconds.", LLM_TIMEOUT)
        st.error("The AI server took too long to respond. Please try again in a moment.")
    except requests.exceptions.ConnectionError:
        logger.error("Cannot reach LLM server at %s.", LLM_ENDPOINT)
        st.error("Cannot reach the AI server. Please check your network connection.")
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP error from LLM: %s", exc)
        st.error(f"AI server returned an error ({exc.response.status_code}). Please try again.")
    except (KeyError, IndexError, ValueError) as exc:
        logger.error("Failed to parse LLM response: %s", exc)
        st.error("Received an unexpected response format from the AI server.")

    return None


# ---------------------------------------------------------------------------
# Stripe integration
# ---------------------------------------------------------------------------


def create_checkout_session() -> Optional[str]:
    """
    Create a Stripe Checkout Session for the monthly Pro subscription.

    Reads STRIPE_SECRET_KEY and STRIPE_PRICE_ID from st.secrets / env vars.
    Returns the hosted Stripe Checkout URL, or None on any Stripe error.
    """
    stripe.api_key = _secret("STRIPE_SECRET_KEY", "sk_test_placeholder")
    price_id = _secret("STRIPE_PRICE_ID", "price_placeholder")
    base = _app_url()

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=(
                f"{base}?payment=success&session_id={{CHECKOUT_SESSION_ID}}"
            ),
            cancel_url=f"{base}?payment=cancelled",
            metadata={"product": "Prompt Engineer Pro"},
        )
        return session.url

    except stripe.error.AuthenticationError:
        logger.error("Stripe AuthenticationError — check STRIPE_SECRET_KEY.")
        st.error("Payment system configuration error. Please contact support.")
    except stripe.error.InvalidRequestError as exc:
        logger.error("Stripe InvalidRequestError: %s", exc)
        st.error(f"Stripe configuration error: {exc.user_message}")
    except stripe.error.StripeError as exc:
        logger.error("StripeError: %s", exc)
        st.error(f"Payment system error: {exc.user_message}")
    except Exception as exc:
        logger.exception("Unexpected error creating Stripe session: %s", exc)
        st.error("An unexpected error occurred. Please try again.")

    return None


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """
    Initialise all required session state keys with safe defaults.

    This function is idempotent — existing values are never overwritten.
    Widget-bound keys (step1, step2, step3, notes) are initialised here so
    they exist before their st.text_area widgets render.
    """
    defaults: dict = {
        "subscribed": False,
        "show_paywall": False,
        "generated_prompt": None,
        "checkout_url": None,
        # Widget state keys — managed by Streamlit after first render
        "step1": "",
        "step2": "",
        "step3": "",
        "notes": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def handle_payment_redirect() -> None:
    """
    Detect a successful Stripe redirect via URL query params.

    Stripe appends ?payment=success after a completed checkout. This function
    activates the subscription in session state and clears the query param to
    prevent re-triggering on page refresh.
    """
    params = st.query_params
    if params.get("payment") == "success":
        st.session_state["subscribed"] = True
        st.session_state["show_paywall"] = False
        st.query_params.clear()
        st.toast("Payment successful! Pro access is now active.", icon="🎉")
    elif params.get("payment") == "cancelled":
        st.query_params.clear()
        st.info("Payment was cancelled. You can subscribe any time from the sidebar.")


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------


def _preview_card(text: str, border_color: str = "#6366f1", bg: str = "#f8f9fa") -> None:
    """Render styled HTML card for prompt preview sections."""
    safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""<div style="background:{bg};border:1px solid #dee2e6;
            border-left:4px solid {border_color};border-radius:8px;
            padding:18px;font-size:.9em;white-space:pre-wrap;
            min-height:80px;font-family:monospace;line-height:1.6;"
        >{safe_text}</div>""",
        unsafe_allow_html=True,
    )


def _badge(label: str, color: str = "#6366f1") -> str:
    """Return an inline HTML badge string."""
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:.78em;font-weight:600;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    """
    Render the navigation sidebar and return the currently selected guide mode.

    Contains:
    - Mode selector radio group
    - Quick Prompt Tips accordion
    - Subscription status badge + upgrade button
    """
    with st.sidebar:
        st.markdown("## ✨ Prompt Engineer")
        st.caption("Anentra AI Consulting")
        st.divider()

        mode: str = st.radio(
            "Guide Mode",
            options=[
                "Step-by-Step Guide",
                "Video Walkthrough",
                "Exercise with Example Template",
            ],
            index=0,
            key="guide_mode",
            help="Choose how you want to explore prompt engineering.",
        )

        st.divider()

        with st.expander("💡 Quick Prompt Tips", expanded=False):
            for title, body in QUICK_TIPS:
                st.markdown(f"**{title}**")
                st.caption(body)
                st.write("")

        st.divider()

        # Subscription status and upgrade path
        if st.session_state.get("subscribed"):
            st.success("✅  Pro Subscriber")
        else:
            st.info("Free Plan")
            if st.button("Upgrade to Pro →", use_container_width=True, key="sidebar_upgrade_btn"):
                url = create_checkout_session()
                if url:
                    st.session_state["checkout_url"] = url

            if st.session_state.get("checkout_url"):
                st.markdown(
                    f"[Complete payment on Stripe ↗]({st.session_state['checkout_url']})"
                )

        st.divider()
        st.caption("v1.0 · Test mode · Not for production use")

    return mode


# ---------------------------------------------------------------------------
# Mode 1 — Step-by-Step Guide
# ---------------------------------------------------------------------------


def render_step_by_step() -> None:
    """
    Render the primary interactive 3-step prompt engineering guide.

    Includes:
    - Read-only progress checklist (reflects live input state)
    - Text area for each framework step
    - Notes section
    - Live assembled prompt preview
    - 'Finish Prompt' button with subscription gate
    """
    st.header("The 3-Step Prompt Engineering Framework")
    st.caption(
        "Fill in each step below to build a structured, effective prompt for any AI model. "
        "The preview updates automatically as you type."
    )
    st.divider()

    # Progress checklist — disabled checkboxes act as read-only visual indicators
    with st.expander("Progress Checklist", expanded=True):
        st.caption("These indicators update automatically based on your inputs.")
        c1, c2, c3 = st.columns(3)
        c1.checkbox(
            "Step 1: Task defined",
            value=bool(st.session_state.get("step1", "").strip()),
            disabled=True,
            key="check_step1",
        )
        c2.checkbox(
            "Step 2: Goal written",
            value=bool(st.session_state.get("step2", "").strip()),
            disabled=True,
            key="check_step2",
        )
        c3.checkbox(
            "Step 3: Structure added",
            value=bool(st.session_state.get("step3", "").strip()),
            disabled=True,
            key="check_step3",
        )

        filled = sum([
            bool(st.session_state.get("step1", "").strip()),
            bool(st.session_state.get("step2", "").strip()),
            bool(st.session_state.get("step3", "").strip()),
        ])
        st.progress(filled / 3, text=f"{filled} of 3 required steps complete")

    st.write("")

    # Framework step inputs
    for step in FRAMEWORK_STEPS:
        with st.container():
            step_col, _ = st.columns([10, 1])
            with step_col:
                st.subheader(f"Step {step['number']}: {step['title']}")
                st.caption(step["desc"])
            st.text_area(
                label=f"Step {step['number']}",
                placeholder=step["placeholder"],
                height=140,
                key=step["key"],           # Streamlit manages value via session state
                label_visibility="collapsed",
            )
        st.write("")

    # Notes section
    st.subheader("Notes")
    st.caption(
        "Optional — add tone, audience, constraints, examples, or output format preferences."
    )
    st.text_area(
        label="Notes",
        placeholder=(
            "e.g., Tone: professional but approachable. Audience: non-technical managers. "
            "Max length: 500 words. Avoid using industry jargon."
        ),
        height=110,
        key="notes",
        label_visibility="collapsed",
    )

    st.divider()

    # Live prompt preview
    st.subheader("Assembled Prompt Preview")
    st.caption("A live view of how your inputs combine into a structured prompt.")

    preview_text = assemble_prompt_preview(
        st.session_state.get("step1", ""),
        st.session_state.get("step2", ""),
        st.session_state.get("step3", ""),
        st.session_state.get("notes", ""),
    )
    _preview_card(preview_text)

    st.write("")

    # Finish Prompt — primary CTA
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button(
            "🚀  Finish Prompt",
            type="primary",
            use_container_width=True,
            key="finish_prompt_btn",
            help="Generate an AI-polished, copy-ready version of your prompt.",
        ):
            _handle_finish_prompt()

    # Conditional UI panels
    if st.session_state.get("show_paywall"):
        _render_paywall()

    if st.session_state.get("generated_prompt"):
        _render_generated_prompt(st.session_state["generated_prompt"])


def _handle_finish_prompt() -> None:
    """
    Validate inputs, check subscription status, and invoke the LLM.

    Flow:
        1. Validate all three steps are non-empty.
        2. If not subscribed → show paywall.
        3. If subscribed → build meta-prompt and call LLM.
        4. On success → store result in session state and rerun.
    """
    validation = validate_inputs(
        st.session_state.get("step1", ""),
        st.session_state.get("step2", ""),
        st.session_state.get("step3", ""),
    )
    missing = [f"Step {i + 1}" for i, v in enumerate(validation.values()) if not v]
    if missing:
        st.warning(f"Please complete {', '.join(missing)} before finishing your prompt.")
        return

    if not st.session_state.get("subscribed"):
        st.session_state["show_paywall"] = True
        return

    with st.spinner("Generating your polished prompt…"):
        meta = build_meta_prompt(
            st.session_state.get("step1", ""),
            st.session_state.get("step2", ""),
            st.session_state.get("step3", ""),
            st.session_state.get("notes", ""),
        )
        result = call_llm(meta)

    if result:
        st.session_state["generated_prompt"] = result
        st.session_state["show_paywall"] = False
        st.rerun()


def _render_paywall() -> None:
    """Render the subscription paywall section below the Finish Prompt button."""
    st.divider()
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                    border-radius:14px;padding:36px 32px;text-align:center;color:white;">
            <h2 style="color:white;margin-bottom:8px;">Unlock Full Access</h2>
            <p style="font-size:1.05em;opacity:.92;margin-bottom:22px;">
                Subscribe to generate AI-polished, copy-ready prompts in seconds.
            </p>
            <ul style="list-style:none;padding:0;text-align:left;
                       display:inline-block;margin-bottom:22px;line-height:2em;">
                <li>✅ &nbsp;AI-polished final prompts</li>
                <li>✅ &nbsp;Unlimited generations per month</li>
                <li>✅ &nbsp;Priority LLM access</li>
                <li>✅ &nbsp;Copy-ready output every time</li>
            </ul><br/>
            <strong style="font-size:1.45em;">$9.99 / month</strong>
            <p style="opacity:.75;font-size:.85em;margin-top:6px;">
                Cancel anytime &nbsp;·&nbsp; Secure checkout via Stripe
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    _, col_pay, _ = st.columns([1, 2, 1])

    with col_pay:
        if st.button(
            "Subscribe Now — $9.99/month",
            type="primary",
            use_container_width=True,
            key="paywall_subscribe_btn",
        ):
            url = create_checkout_session()
            if url:
                st.session_state["checkout_url"] = url

        if st.session_state.get("checkout_url"):
            st.markdown(
                f"""<div style="text-align:center;margin-top:14px;">
                    <a href="{st.session_state['checkout_url']}" target="_blank"
                       style="background:#6366f1;color:white;padding:12px 28px;
                              border-radius:8px;text-decoration:none;font-weight:600;
                              font-size:.95em;">
                        Complete Payment on Stripe &nbsp;→
                    </a></div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        st.caption("Demo only — skip payment for testing:")
        if st.button(
            "Simulate Successful Payment",
            use_container_width=True,
            key="demo_payment_btn",
        ):
            st.session_state["subscribed"] = True
            st.session_state["show_paywall"] = False
            st.session_state["checkout_url"] = None
            st.rerun()


def _render_generated_prompt(prompt_text: str) -> None:
    """Display the AI-generated polished prompt with a copyable code block."""
    st.divider()
    st.subheader("Your AI-Generated Prompt")
    st.caption("This polished prompt is ready to copy and paste into any AI assistant.")

    _preview_card(prompt_text, border_color="#22c55e", bg="#f0fdf4")

    st.code(prompt_text, language=None)
    st.caption("Use the copy icon in the top-right corner of the code block above.")

    _, col_reset, _ = st.columns([1, 2, 1])
    with col_reset:
        if st.button("Generate Again", use_container_width=True, key="regen_btn"):
            st.session_state["generated_prompt"] = None
            st.rerun()


# ---------------------------------------------------------------------------
# Mode 2 — Video Walkthrough
# ---------------------------------------------------------------------------


def render_video_walkthrough() -> None:
    """
    Render the Video Walkthrough mode.

    Includes a placeholder video area, four curated YouTube resource cards,
    and a pandas-backed summary table.
    """
    st.header("Video Walkthrough")
    st.caption(
        "Watch curated tutorials to build a deeper understanding of prompt engineering."
    )
    st.divider()

    # Placeholder video embed area
    st.markdown(
        """<div style="background:#1e1e2e;border-radius:12px;height:270px;
               display:flex;align-items:center;justify-content:center;
               margin-bottom:28px;">
           <div style="text-align:center;color:#cdd6f4;">
               <div style="font-size:3em;margin-bottom:10px;">▶</div>
               <p style="font-size:1.05em;font-weight:600;margin:0;">
                   Course Video Placeholder
               </p>
               <p style="opacity:.65;font-size:.88em;margin-top:6px;">
                   Embed your video using
                   <code>st.video("url_or_local_path")</code>
               </p>
           </div>
       </div>""",
        unsafe_allow_html=True,
    )

    st.subheader("Recommended Resources")
    st.caption("Hand-picked tutorials from leading AI educators.")
    st.write("")

    for video in VIDEOS:
        with st.container():
            icon_col, body_col = st.columns([1, 10])
            with icon_col:
                st.markdown(
                    "<div style='font-size:1.9em;padding-top:5px;'>📺</div>",
                    unsafe_allow_html=True,
                )
            with body_col:
                st.markdown(f"**[{video['title']}]({video['url']})**")
                st.caption(
                    f"{video['desc']}"
                    f"&nbsp;&nbsp;·&nbsp;&nbsp;⏱ {video['duration']}"
                    f"&nbsp;&nbsp;·&nbsp;&nbsp;Level: **{video['level']}**"
                )
        st.divider()

    # Pandas summary table
    st.subheader("Resource Summary")
    df = pd.DataFrame(
        [
            {
                "Title": v["title"],
                "Duration": v["duration"],
                "Level": v["level"],
                "Link": v["url"],
            }
            for v in VIDEOS
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Mode 3 — Exercise with Example Template
# ---------------------------------------------------------------------------


def render_exercise_mode() -> None:
    """
    Render the Exercise mode with a fully worked example and a practice area.

    Users can study the example, then apply the same framework to their own
    topic. A 'Use in Step-by-Step Guide' button transfers their inputs across.
    """
    st.header("Exercise: Build Prompts with a Template")
    st.caption(
        "Study a fully worked example, then apply the same framework to your own task."
    )
    st.divider()

    # ── Worked example ───────────────────────────────────────────────────────
    st.subheader("Worked Example: Blog Post on Remote Work")
    st.info(
        "The example below shows how the 3-step framework produces a structured, "
        "effective prompt. Read through each step, then try your own task below."
    )

    with st.expander("Step 1 — Task Definition", expanded=False):
        st.markdown(f"> {EXAMPLE['step1']}")

    with st.expander("Step 2 — One-Sentence Goal", expanded=False):
        st.markdown(f"> {EXAMPLE['step2']}")

    with st.expander("Step 3 — Structure & Headings", expanded=False):
        st.code(EXAMPLE["step3"], language=None)

    with st.expander("Notes", expanded=False):
        st.markdown(f"> {EXAMPLE['notes']}")

    st.caption("Assembled preview of the example prompt:")
    _preview_card(
        assemble_prompt_preview(
            EXAMPLE["step1"],
            EXAMPLE["step2"],
            EXAMPLE["step3"],
            EXAMPLE["notes"],
        ),
        border_color="#eab308",
        bg="#fefce8",
    )

    st.divider()

    # ── Practice area ─────────────────────────────────────────────────────────
    st.subheader("Try Your Own Task")
    st.caption(
        "Apply the 3-step framework to a topic of your choice. "
        "When you're ready, transfer your inputs to the Step-by-Step Guide."
    )

    ex_step1 = st.text_area(
        "Step 1 — What should the AI do?",
        placeholder=EXAMPLE["step1"],
        height=95,
        key="ex_step1",
    )
    ex_step2 = st.text_area(
        "Step 2 — One-sentence goal",
        placeholder=EXAMPLE["step2"],
        height=82,
        key="ex_step2",
    )
    ex_step3 = st.text_area(
        "Step 3 — Headings + bullet points",
        placeholder=EXAMPLE["step3"],
        height=190,
        key="ex_step3",
    )
    ex_notes = st.text_area(
        "Notes (optional)",
        placeholder=EXAMPLE["notes"],
        height=82,
        key="ex_notes",
    )

    # Live preview only appears once the user starts typing
    if any(s.strip() for s in [ex_step1, ex_step2, ex_step3]):
        st.subheader("Practice Prompt Preview")
        _preview_card(
            assemble_prompt_preview(ex_step1, ex_step2, ex_step3, ex_notes),
            border_color="#0ea5e9",
            bg="#f0f9ff",
        )

        st.write("")
        _, col_use, _ = st.columns([1, 2, 1])
        with col_use:
            if st.button(
                "Use This in the Step-by-Step Guide →",
                use_container_width=True,
                key="use_exercise_btn",
            ):
                # Copy exercise inputs into the main guide's session state keys.
                # The guide's text_area widgets are not rendered in this mode,
                # so setting these keys is safe and takes effect on the next run.
                st.session_state["step1"] = ex_step1
                st.session_state["step2"] = ex_step2
                st.session_state["step3"] = ex_step3
                st.session_state["notes"] = ex_notes
                st.success(
                    "Inputs transferred. Select 'Step-by-Step Guide' in the sidebar to continue."
                )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Configure the Streamlit page, initialise state, and route to the active mode.

    Call order:
        1. set_page_config  (must be first Streamlit call)
        2. inject minimal global CSS
        3. init_session_state
        4. handle_payment_redirect  (reads query params from Stripe redirect)
        5. render header + sidebar → get active mode
        6. render active mode UI
        7. render footer
    """
    st.set_page_config(
        page_title="Prompt Engineering Assistant",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Minimal global CSS — keeps content readable and constrains line width
    st.markdown(
        """<style>
        .main .block-container {
            max-width: 860px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }
        div[data-testid="stTextArea"] textarea {
            font-family: inherit;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    init_session_state()
    handle_payment_redirect()

    # ── Page header ────────────────────────────────────────────────────────
    st.title("✨ Prompt Engineering Assistant")
    st.subheader(APP_SUBTITLE)
    st.caption(
        "A structured, 3-step framework for building clear, powerful prompts for "
        "ChatGPT, Claude, Gemini, and any large language model."
    )
    st.divider()

    mode = render_sidebar()

    # ── Mode routing ───────────────────────────────────────────────────────
    if mode == "Step-by-Step Guide":
        render_step_by_step()
    elif mode == "Video Walkthrough":
        render_video_walkthrough()
    else:
        render_exercise_mode()

    # ── Footer ─────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Demo application · Anentra AI Consulting · "
        "Stripe integration in test mode (no real payments processed) · "
        "LLM: Qwen 2.5 7B · Replace placeholder API keys before deploying to production."
    )


# ---------------------------------------------------------------------------
# Unit tests
# Run with:  pytest app.py -v
# ---------------------------------------------------------------------------


def test_assemble_prompt_preview_all_sections_present():
    """All four labelled sections appear when all inputs are non-empty."""
    result = assemble_prompt_preview("task", "goal", "structure", "notes")
    assert "TASK" in result
    assert "GOAL" in result
    assert "STRUCTURE" in result
    assert "ADDITIONAL CONTEXT" in result


def test_assemble_prompt_preview_empty_returns_placeholder():
    """Returns placeholder text when all inputs are empty strings."""
    result = assemble_prompt_preview("", "", "", "")
    assert "appear here" in result


def test_assemble_prompt_preview_partial_excludes_empty_sections():
    """Only non-empty sections appear in the assembled preview."""
    result = assemble_prompt_preview("task only", "", "", "")
    assert "TASK" in result
    assert "GOAL" not in result
    assert "STRUCTURE" not in result
    assert "ADDITIONAL CONTEXT" not in result


def test_assemble_prompt_preview_notes_only():
    """Notes section appears even when all framework steps are empty."""
    result = assemble_prompt_preview("", "", "", "my notes")
    assert "ADDITIONAL CONTEXT" in result
    assert "my notes" in result


def test_validate_inputs_all_valid():
    """All three steps return True when non-empty."""
    assert validate_inputs("a", "b", "c") == {
        "step1": True,
        "step2": True,
        "step3": True,
    }


def test_validate_inputs_one_missing():
    """A single missing step returns False for that key only."""
    result = validate_inputs("a", "", "c")
    assert result["step1"] is True
    assert result["step2"] is False
    assert result["step3"] is True


def test_validate_inputs_whitespace_treated_as_empty():
    """Whitespace-only strings are treated as invalid (empty)."""
    result = validate_inputs("  ", "\t", "\n")
    assert all(v is False for v in result.values())


def test_validate_inputs_all_empty():
    """All three steps return False when all inputs are empty strings."""
    result = validate_inputs("", "", "")
    assert result == {"step1": False, "step2": False, "step3": False}


def test_build_meta_prompt_contains_all_user_inputs():
    """Every user input string appears verbatim in the meta-prompt."""
    prompt = build_meta_prompt("my_task", "my_goal", "my_structure", "my_notes")
    for fragment in ("my_task", "my_goal", "my_structure", "my_notes"):
        assert fragment in prompt, f"Expected '{fragment}' in meta-prompt"


def test_build_meta_prompt_empty_notes_uses_fallback():
    """When notes are empty, the meta-prompt includes 'None provided.' fallback."""
    prompt = build_meta_prompt("t", "g", "s", "")
    assert "None provided." in prompt


def test_build_meta_prompt_contains_no_preamble_instruction():
    """Meta-prompt must instruct the LLM to output only the final prompt."""
    prompt = build_meta_prompt("t", "g", "s", "n")
    assert "No preamble" in prompt or "ONLY the final" in prompt


def test_build_meta_prompt_structure_order():
    """Steps appear in the correct order within the meta-prompt."""
    prompt = build_meta_prompt("TASK_VAL", "GOAL_VAL", "STRUCT_VAL", "NOTES_VAL")
    idx_task = prompt.index("TASK_VAL")
    idx_goal = prompt.index("GOAL_VAL")
    idx_struct = prompt.index("STRUCT_VAL")
    idx_notes = prompt.index("NOTES_VAL")
    assert idx_task < idx_goal < idx_struct < idx_notes


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

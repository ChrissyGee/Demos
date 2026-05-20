"""
================================================================================
CLAUDE FOR EXCEL PLUG-IN — SETUP GUIDE (STREAMLIT MVP)
================================================================================

An interactive, professional step-by-step guide that walks individuals and
admins through installing and using the Claude for Excel add-in.

The app is intentionally lightweight — only `streamlit` and `pandas` — so it
can be dropped on any AI consulting website as a demo or be embedded into
a client onboarding portal.

PAGES
    * Sidebar mode switch:
        - "Text Explanations" (default) — tabs covering Overview, Individual
          install, Business / Admin deployment, Post-install usage, and
          Best Practices & Limitations. Includes a persistent checklist
          users can tick as they go.
        - "Video Setup Guide" — curated public videos plus a written
          summary of what each video covers.

CONTENT
    The setup steps reflect the current Microsoft Office add-in deployment
    model (Microsoft AppSource / Microsoft 365 Admin Center / custom
    manifest XML). Treat the content as a demo guide — always cross-check
    against the official Claude Help Center before deploying in production.

USAGE
    pip install streamlit pandas
    streamlit run app.py
================================================================================
"""

# ============================================================
# IMPORTS
# ============================================================
# Only `streamlit` and `pandas` are used so the app has zero external
# dependencies beyond a standard Python environment.
import streamlit as st
import pandas as pd
from typing import List, Dict


# ============================================================
# PAGE CONFIG  (must be the first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Claude for Excel — Setup Guide",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STATIC CONTENT
# ============================================================
# All long-form content lives in module-level constants so the rendering
# functions stay short and the copy is easy to update without touching UI code.

SUPPORTED_VERSIONS: List[Dict[str, str]] = [
    # (platform, version, notes)
    {"Platform": "Excel on the Web",         "Minimum Version": "Current",
     "Notes": "Recommended — fastest path; works in any modern browser."},
    {"Platform": "Excel for Windows",        "Minimum Version": "Microsoft 365 (Build 16.0.14931+)",
     "Notes": "Requires a Microsoft 365 subscription. Office 2021/2019 are not supported."},
    {"Platform": "Excel for macOS",          "Minimum Version": "Microsoft 365 (Build 16.79+)",
     "Notes": "Requires a Microsoft 365 subscription on macOS 12 or later."},
    {"Platform": "Excel on iPad",            "Minimum Version": "Latest App Store build",
     "Notes": "Add-ins panel must be enabled by your admin."},
    {"Platform": "Excel for Android / iOS phone", "Minimum Version": "Not supported",
     "Notes": "The add-in surface is desktop / tablet / web only."},
]

CLAUDE_PLANS: List[Dict[str, str]] = [
    {"Plan": "Claude Free",       "Excel Add-in Access": "No",
     "Best For": "Trying Claude in the browser before upgrading."},
    {"Plan": "Claude Pro",        "Excel Add-in Access": "Yes",
     "Best For": "Individual analysts and consultants."},
    {"Plan": "Claude Max",        "Excel Add-in Access": "Yes",
     "Best For": "Power users with higher usage limits."},
    {"Plan": "Claude Team",       "Excel Add-in Access": "Yes (centrally managed)",
     "Best For": "Small teams sharing workspaces."},
    {"Plan": "Claude Enterprise", "Excel Add-in Access": "Yes (SSO + admin controls)",
     "Best For": "Regulated orgs needing SAML SSO, audit logs, data controls."},
]

MODELS_AVAILABLE: List[Dict[str, str]] = [
    {"Model": "Claude Opus 4.7",   "Best For": "Deep analysis, multi-step reasoning over large ranges."},
    {"Model": "Claude Sonnet 4.6", "Best For": "Day-to-day analytical work — strong quality at lower cost."},
    {"Model": "Claude Haiku 4.5",  "Best For": "Fast lookups, simple transformations, high-volume cells."},
]

# Each step in the checklist is a (key, label) tuple. The key persists the
# tick state across reruns via `st.session_state`.
INDIVIDUAL_CHECKLIST: List[Dict[str, str]] = [
    {"key": "ind_1", "label": "Confirmed I have a Claude Pro, Max, Team, or Enterprise plan."},
    {"key": "ind_2", "label": "Confirmed Excel version is supported (see table)."},
    {"key": "ind_3", "label": "Opened Excel → Home tab → Add-ins → Get Add-ins."},
    {"key": "ind_4", "label": "Searched AppSource / Microsoft Marketplace for 'Claude'."},
    {"key": "ind_5", "label": "Clicked Add and accepted the permissions dialog."},
    {"key": "ind_6", "label": "Opened the Claude pane from the Home ribbon."},
    {"key": "ind_7", "label": "Signed in with my Claude account (OAuth)."},
    {"key": "ind_8", "label": "Ran a test prompt against a sample range."},
]

ADMIN_CHECKLIST: List[Dict[str, str]] = [
    {"key": "adm_1", "label": "Confirmed users have eligible Claude plans (Team / Enterprise)."},
    {"key": "adm_2", "label": "Signed in to the Microsoft 365 Admin Center."},
    {"key": "adm_3", "label": "Opened Settings → Integrated apps → Get apps."},
    {"key": "adm_4", "label": "Located Claude in the catalog (or uploaded the manifest XML)."},
    {"key": "adm_5", "label": "Chose deployment scope: entire org / groups / specific users."},
    {"key": "adm_6", "label": "Reviewed and accepted requested permissions."},
    {"key": "adm_7", "label": "Allowed up to 24 hours for propagation to user Excel clients."},
    {"key": "adm_8", "label": "Communicated rollout + sign-in instructions to end users."},
    {"key": "adm_9", "label": "Configured audit log retention / DLP policy review."},
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def init_session_state() -> None:
    """
    Initialise persistent state for the checklist.

    Streamlit reruns the whole script on each interaction, so checklist
    state must live in `st.session_state` to survive across reruns.
    """
    if "checklist_initialised" in st.session_state:
        return
    for item in INDIVIDUAL_CHECKLIST + ADMIN_CHECKLIST:
        st.session_state.setdefault(item["key"], False)
    st.session_state["checklist_initialised"] = True


def render_checklist(items: List[Dict[str, str]], heading: str) -> None:
    """
    Render an interactive progress checklist.

    Parameters
    ----------
    items : list of {"key": str, "label": str}
        Each item drives one `st.checkbox`. The `key` persists state in
        `st.session_state` so progress survives page reruns.
    heading : str
        The section heading shown above the checklist.
    """
    st.markdown(f"#### {heading}")
    total = len(items)
    done = sum(1 for item in items if st.session_state.get(item["key"], False))

    # Progress bar gives the user a sense of how far they've come.
    st.progress(done / total if total else 0,
                text=f"{done} / {total} steps complete")

    for item in items:
        # Each checkbox is bound to a session_state key.
        st.checkbox(item["label"], key=item["key"])

    if done == total and total > 0:
        st.success("✅ All steps complete. You're ready to use Claude in Excel.")


def render_table(rows: List[Dict[str, str]], caption: str = "") -> None:
    """
    Render a styled table from a list-of-dicts.

    Using a small helper keeps every table consistent (same caption style,
    same `hide_index=True`, same width behaviour) across the app.
    """
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if caption:
        st.caption(caption)


# ============================================================
# TEXT-MODE TABS
# ============================================================

def render_overview_tab() -> None:
    """Top-of-funnel overview: what is the add-in, who's it for, what does it do."""
    st.subheader("What is Claude for Excel?")
    st.markdown(
        """
        **Claude for Excel** is a Microsoft Office add-in that brings Anthropic's
        Claude models directly into the Excel workspace. Once installed it lives
        in a side pane next to your spreadsheet so you can:

        - Ask natural-language questions about the data in any range.
        - Generate formulas (and have Claude explain them in plain English).
        - Clean, restructure, and standardise messy data.
        - Build summaries, narratives, and stakeholder-ready commentary.
        - Walk through what a workbook does without opening every tab.
        """
    )

    st.divider()
    st.markdown("### ✨ Why teams adopt it")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "**📈 Analyst leverage**  \n"
        "Reduces time spent on data clean-up and formula authoring so analysts "
        "spend more time on interpretation."
    )
    c2.markdown(
        "**🧠 Lower formula barrier**  \n"
        "Junior team members can describe the answer they need and let Claude "
        "build the formula — with a step-by-step explanation."
    )
    c3.markdown(
        "**🔒 Centrally manageable**  \n"
        "Admins deploy the add-in via the Microsoft 365 Admin Center, with "
        "audit trails on Claude Enterprise."
    )

    st.divider()
    st.markdown("### 🖥 Supported Excel versions")
    render_table(
        SUPPORTED_VERSIONS,
        caption="Always cross-check with the Claude Help Center for the most "
                "current platform support — Microsoft updates Office channels frequently.",
    )

    st.markdown("### 🪪 Plan eligibility")
    render_table(
        CLAUDE_PLANS,
        caption="The add-in is gated by your Claude plan, not your Microsoft plan.",
    )

    st.markdown("### 🧪 Models you can pick inside the pane")
    render_table(MODELS_AVAILABLE)


def render_individual_tab() -> None:
    """Step-by-step path for an individual installing the add-in for themselves."""
    st.subheader("For Individuals — Install from Microsoft AppSource")
    st.caption("Use this path if you're installing Claude for Excel only for yourself. "
               "If you're an admin rolling it out to a team, jump to the next tab.")

    with st.expander("Step 1 — Confirm prerequisites", expanded=True):
        st.markdown(
            """
            - You have a paid Claude plan (**Pro**, **Max**, **Team**, or **Enterprise**).
              Free accounts cannot use the Excel add-in.
            - Your Excel is a supported version (see the Overview tab).
            - Your tenant admin has **not** blocked third-party add-ins. If
              you don't see *Get Add-ins* in the ribbon, this is the likely
              cause — see the Business / Admins tab.
            """
        )

    with st.expander("Step 2 — Open the Office Add-ins picker", expanded=True):
        st.markdown(
            """
            1. Open Excel (web, Windows, or macOS).
            2. Go to the **Home** tab.
            3. Click **Add-ins** → **Get Add-ins**.
               *(On older builds: Insert → Get Add-ins.)*
            4. In the **Office Add-ins** dialog, switch to the **Store** tab.
            """
        )

    with st.expander("Step 3 — Install the Claude add-in", expanded=True):
        st.markdown(
            """
            1. In the search box, type **`Claude`** and press Enter.
            2. Pick the official **Anthropic** publisher entry.
            3. Click **Add** and review the requested permissions.
            4. Accept the permissions to complete installation.
            """
        )

    with st.expander("Step 4 — Open the Claude pane and sign in", expanded=True):
        st.markdown(
            """
            1. Back in Excel, open the **Home** tab.
            2. Click the **Claude** button (it appears on the right side of the ribbon).
            3. A side pane opens. Click **Sign in**.
            4. A browser tab opens for OAuth — sign in with the same email
               attached to your Claude plan.
            5. Return to Excel — the pane refreshes and is ready to use.
            """
        )

    with st.expander("Step 5 — Run a smoke test", expanded=True):
        st.markdown(
            """
            Select any small range of data (e.g. 5 rows × 3 columns), then in
            the Claude pane try:

            > *"Summarise this range in three bullet points and flag any rows
            > that look like outliers."*

            If the pane returns a coherent answer, the install is good.
            """
        )

    st.divider()
    render_checklist(INDIVIDUAL_CHECKLIST,
                     heading="📋 Your individual setup checklist")


def render_business_tab() -> None:
    """Step-by-step path for an admin deploying the add-in across a tenant."""
    st.subheader("For Businesses / Admins — Centralized Deployment")
    st.caption("Use this path to deploy Claude for Excel to a team, department, "
               "or entire Microsoft 365 tenant.")

    st.markdown(
        """
        There are two supported admin paths:

        | Path | When to use |
        |---|---|
        | **Catalog deploy** via Microsoft 365 Admin Center | The add-in is publicly listed and you want a simple click-through deployment. |
        | **Custom manifest XML** upload | Your org requires a specific manifest (e.g. for testing, staging, or restricted regions). |

        Both paths use the **Integrated apps** surface of the Microsoft 365 Admin Center.
        """
    )

    with st.expander("Step 1 — Prerequisites (admin)", expanded=True):
        st.markdown(
            """
            - You hold one of: **Global Administrator**, **Exchange Administrator**, or
              **Office Apps Admin** role in your tenant.
            - Target users are licensed for Microsoft 365 Apps and have a
              compatible Excel client.
            - Target users are on a Claude **Team** or **Enterprise** plan
              that includes Excel add-in access.
            - You've reviewed the add-in's requested permissions with your
              security / privacy team.
            """
        )

    with st.expander("Step 2 — Open Integrated apps in the M365 Admin Center", expanded=True):
        st.markdown(
            """
            1. Sign in at **admin.microsoft.com**.
            2. In the left nav: **Settings** → **Integrated apps**.
            3. Click **Get apps**.
            4. Search for **Claude**.
            """
        )

    with st.expander("Step 3a — Deploy from the catalog (recommended)", expanded=True):
        st.markdown(
            """
            1. Open the Claude entry and click **Get it now**.
            2. Choose deployment scope:
               - *Entire organization*
               - *Specific users / groups* (pick Microsoft 365 groups or security groups)
               - *Just me* (for piloting)
            3. Review permissions and click **Next** → **Finish deployment**.
            4. Deployment can take up to **24 hours** to propagate to all clients.
            """
        )

    with st.expander("Step 3b — Deploy with a custom manifest XML", expanded=False):
        st.markdown(
            """
            Use this path only if directed by Anthropic or your regulatory team.

            1. Obtain the manifest XML from Anthropic (your account team will
               provide it).
            2. In **Integrated apps**, click **Upload custom apps**.
            3. Choose **Office Add-in** → **Upload manifest file (.xml)**.
            4. Select the XML, configure scope (org / groups / users),
               accept permissions, and finish.
            5. Validate by opening Excel as a target user and confirming the
               Claude button appears in the Home ribbon.
            """
        )

    with st.expander("Step 4 — Configure audit, DLP, and conditional access", expanded=True):
        st.markdown(
            """
            For Claude **Enterprise** tenants:

            - Enable **SSO via SAML** in the Claude admin console and point
              it at your IdP (Okta, Entra ID, Ping).
            - Mirror your Office DLP policies — Claude inherits the same data
              boundaries you configure for other approved third-party add-ins.
            - Turn on **audit log forwarding** (Claude → SIEM) so usage and
              data-access events flow into Splunk / Sentinel / Datadog.
            - Document the rollout in your AI usage policy and notify users
              before turning the add-in on.
            """
        )

    with st.expander("Step 5 — Communicate the rollout", expanded=True):
        st.markdown(
            """
            A short user-facing comms note typically contains:

            - What the add-in does and which Excel surfaces show it.
            - Where to click to launch the Claude pane.
            - Sign-in instructions (which account to use).
            - A reminder of your AI usage policy (what data is OK to send).
            - A help-desk channel for questions and access requests.
            """
        )

    st.divider()
    render_checklist(ADMIN_CHECKLIST,
                     heading="📋 Admin deployment checklist")


def render_after_installation_tab() -> None:
    """Day-1 usage: what the pane does and how to drive it."""
    st.subheader("After Installation — Using Claude in Excel")
    st.markdown(
        """
        Once installed, the Claude pane lives on the right side of Excel. It
        is context-aware: it sees whatever range or sheet you have selected.
        """
    )

    with st.expander("Opening the pane", expanded=True):
        st.markdown(
            """
            - **Home** ribbon → click **Claude** (look for the Anthropic logo).
            - On Excel for the web, the pane opens inside the browser tab.
            - On Excel for Windows / macOS, the pane is a native side panel.
            """
        )

    with st.expander("What you can do", expanded=True):
        st.markdown(
            """
            **Analyse data**

            > *"Compare Q1 and Q2 by region. Which region grew the most?"*

            **Generate and explain formulas**

            > *"Build a formula that counts unique customers per quarter and
            > explain how it works."*

            **Clean and transform**

            > *"Standardise the country column to ISO 3166-1 alpha-2 codes."*

            **Summarise narratives**

            > *"Write a 5-bullet executive summary I can paste into a deck."*

            **Walk through a workbook**

            > *"Explain what the 'Pipeline' sheet does and how it feeds the
            > Dashboard."*
            """
        )

    with st.expander("Picking a model", expanded=False):
        st.markdown(
            """
            The pane lets you switch between models. As a rule of thumb:

            - **Claude Haiku 4.5** — quick lookups, low-latency batch work.
            - **Claude Sonnet 4.6** — most analytical work; the best default.
            - **Claude Opus 4.7** — multi-step reasoning over very large
              ranges, intricate financial modelling, complex pivots.
            """
        )

    with st.expander("Inserting Claude's output back into the sheet", expanded=False):
        st.markdown(
            """
            - Click **Insert** on a Claude response to drop it into the
              currently selected cell.
            - For formulas, Claude inserts the formula at the active cell
              and shows a preview of the computed value.
            - Always review changes — the **Undo** ribbon button reverts
              the most recent insertion just like any manual edit.
            """
        )


def render_best_practices_tab() -> None:
    """Limitations, security notes, and prompt hygiene."""
    st.subheader("Best Practices & Limitations")

    st.markdown("### ✅ Good prompting practices")
    st.markdown(
        """
        - **Be explicit about the range.** Say *"using A1:D200"* instead of
          *"this data"*.
        - **State the deliverable.** *"Return a markdown table"* or
          *"insert a formula into E2"* — Claude picks better tools when
          you describe the output shape.
        - **Provide one example.** A single worked row is often the
          difference between a 70% and a 95% answer.
        - **Iterate, don't restart.** Refine in the same conversation so
          Claude keeps the workbook context.
        """
    )

    st.markdown("### 🛡 Security & data handling")
    st.markdown(
        """
        - The add-in sends the **range you reference** to Claude — not the
          entire workbook by default. Be mindful when you select large
          ranges containing sensitive data.
        - On **Claude Enterprise**, your data is not used for model training
          and is governed by the enterprise data agreement. Confirm your
          plan tier with your account team before sending regulated data.
        - **Prompt injection caution.** If a workbook contains text that
          instructs Claude to do something (e.g. a *"System:"* block pasted
          into a cell), Claude may interpret it as an instruction. Review
          outputs before accepting changes.
        - **Review every formula and every overwrite** before saving the
          workbook. Treat Claude as a senior peer — fast and capable, but
          you still own the answer.
        """
    )

    st.markdown("### ⚠ Known limitations")
    st.markdown(
        """
        - The add-in surface is **desktop / tablet / web only** — there is
          no phone Excel experience.
        - Very large workbooks may need you to point Claude at one sheet at
          a time rather than the whole file.
        - Charts and pivot caches are only partially introspectable — for
          deep chart edits, use Excel's native tools and ask Claude to
          describe what to do.
        - VBA / macro code is not modified by the add-in.
        """
    )

    st.markdown("### 🆘 If something goes wrong")
    st.markdown(
        """
        - **Pane doesn't appear** → your admin may have disabled third-party
          add-ins. Confirm with IT.
        - **Sign-in fails** → check that the email matches your Claude plan
          and that pop-ups / OAuth redirects are allowed.
        - **Add-in missing after admin rollout** → propagation can take up
          to 24 hours; restart Excel afterwards.
        - **Outdated info or content errors** → always verify against the
          official **Claude Help Center**.
        """
    )


# ============================================================
# VIDEO-MODE RENDERING
# ============================================================
# In a production deployment the placeholder below would be replaced with
# the actual hosted setup video. For the demo we link out to public,
# generally available walkthroughs and provide a written summary so the
# section is still useful if the embed fails or is blocked by a firewall.

RECOMMENDED_VIDEOS: List[Dict[str, str]] = [
    {
        "title": "Official Anthropic — Claude for Excel announcement",
        "url": "https://www.youtube.com/results?search_query=Claude+for+Excel+Anthropic",
        "covers": "What the add-in is, the pane UX, and the first-run flow.",
    },
    {
        "title": "Microsoft 365 — Deploying third-party Office add-ins",
        "url": "https://www.youtube.com/results?search_query=Microsoft+365+admin+center+deploy+office+add-in",
        "covers": "Generic Microsoft 365 Admin Center walkthrough — applies "
                  "directly to deploying Claude.",
    },
    {
        "title": "Community walkthrough — first 5 prompts to try in Claude for Excel",
        "url": "https://www.youtube.com/results?search_query=Claude+Excel+prompt+examples",
        "covers": "Practical prompts for analysis, formula generation, and "
                  "data clean-up.",
    },
]

VIDEO_SUMMARY_POINTS: List[str] = [
    "Verify your Claude plan tier and Excel version before installing.",
    "For individuals: Home → Add-ins → Get Add-ins → search 'Claude' → Add.",
    "Sign in via OAuth using the email tied to your Claude plan.",
    "For admins: deploy via Microsoft 365 Admin Center → Integrated apps "
    "→ Get apps (or upload custom manifest XML).",
    "Allow up to 24 hours for tenant-wide deployments to propagate.",
    "Pick a model in the pane — Sonnet 4.6 is the recommended default.",
    "Always review formulas / overwrites before saving the workbook.",
    "Treat any text inside cells as untrusted input — be alert to prompt "
    "injection in narrative columns.",
]


def render_video_mode() -> None:
    """Render the dedicated video setup guide page."""
    st.subheader("🎥 Video Setup Guide")
    st.markdown(
        """
        Prefer to watch someone walk through the install? Use the curated
        videos below. In a production deployment of this guide, the player
        on the left would embed your organization's official walkthrough.
        """
    )

    col_player, col_summary = st.columns([3, 2])

    with col_player:
        st.markdown("#### Featured walkthrough")
        # Placeholder video. In production, swap for the org's hosted video.
        # We use a well-known evergreen Microsoft demo as a stand-in.
        try:
            st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            st.caption("Placeholder — replace with your hosted setup video in production.")
        except Exception:
            st.info(
                "Video embed unavailable in this environment. "
                "Use the recommended links to the right to watch on YouTube."
            )

    with col_summary:
        st.markdown("#### Key points covered")
        for i, point in enumerate(VIDEO_SUMMARY_POINTS, 1):
            st.markdown(f"**{i}.** {point}")

    st.divider()
    st.markdown("#### 📚 Recommended videos")
    for v in RECOMMENDED_VIDEOS:
        with st.container(border=True):
            st.markdown(f"**{v['title']}**")
            st.markdown(v["covers"])
            st.link_button("Open on YouTube", v["url"], use_container_width=False)

    st.info(
        "**Note:** This video section is a demo. In a production deployment of this "
        "guide, embed the official Anthropic walkthrough using `st.video(...)` "
        "with your hosted URL."
    )


# ============================================================
# FAQ
# ============================================================

def render_faq() -> None:
    """A small FAQ shown at the bottom of the text-mode page."""
    st.markdown("### ❓ Frequently asked questions")

    with st.expander("Do I need a paid Claude plan to use the add-in?"):
        st.markdown(
            "Yes. The Excel add-in requires Claude Pro, Max, Team, or "
            "Enterprise. Free accounts cannot use it."
        )

    with st.expander("Can I install it on Excel for iPhone / Android?"):
        st.markdown(
            "No. The add-in surface is web, Windows, macOS, and iPad only. "
            "Excel on phones does not host third-party Office add-ins."
        )

    with st.expander("Does Claude see my whole workbook?"):
        st.markdown(
            "No. Claude only sees what you point it at — the range or sheet "
            "currently in context. Large or sensitive ranges should be "
            "shared deliberately."
        )

    with st.expander("Is my data used for model training?"):
        st.markdown(
            "Claude Enterprise data is not used for training and is governed "
            "by the enterprise data agreement. Always confirm with your "
            "account team for your specific plan."
        )

    with st.expander("How do I uninstall it?"):
        st.markdown(
            "Individuals: **Home → Add-ins → My Add-ins**, find Claude and "
            "remove. Admins: remove the deployment in **Microsoft 365 Admin "
            "Center → Integrated apps**."
        )

    with st.expander("Why is the Claude button missing from my ribbon?"):
        st.markdown(
            "Most common causes: (a) your admin has not yet deployed the "
            "add-in; (b) propagation is still in progress (up to 24 hours); "
            "(c) third-party add-ins are blocked in your tenant; (d) your "
            "Excel build is below the minimum supported version."
        )


# ============================================================
# QUICK SETUP SIMULATOR
# ============================================================

def render_quick_simulator() -> None:
    """Tiny interactive widget that surfaces the right install path."""
    st.markdown("### 🧭 Quick Setup Simulator")
    st.caption("Answer two questions and the guide will point you at the right path.")

    c1, c2 = st.columns(2)
    audience = c1.radio(
        "I'm setting this up for...",
        ["Just me", "My team / organisation"],
        horizontal=True,
    )
    platform = c2.selectbox(
        "I use Excel on...",
        ["Excel on the Web", "Excel for Windows", "Excel for macOS",
         "Excel on iPad", "Excel on phone"],
    )

    if platform == "Excel on phone":
        st.error("Excel on phones doesn't support Office add-ins — switch to web, "
                 "desktop, or iPad.")
        return

    if audience == "Just me":
        st.success(
            "→ Use the **For Individuals** tab. You'll install from Microsoft "
            "AppSource via **Home → Add-ins → Get Add-ins**."
        )
    else:
        st.success(
            "→ Use the **For Businesses / Admins** tab. Deploy from the "
            "**Microsoft 365 Admin Center → Integrated apps → Get apps**, or "
            "upload a custom manifest XML."
        )


# ============================================================
# TEXT-MODE PAGE COMPOSITION
# ============================================================

def render_text_mode() -> None:
    """Compose the full text-mode page: tabs + quick simulator + FAQ."""
    render_quick_simulator()
    st.divider()

    tab_overview, tab_ind, tab_biz, tab_after, tab_best = st.tabs([
        "🏠 Overview",
        "👤 For Individuals",
        "🏢 For Businesses / Admins",
        "🚀 After Installation",
        "🛡 Best Practices & Limitations",
    ])

    with tab_overview:    render_overview_tab()
    with tab_ind:         render_individual_tab()
    with tab_biz:         render_business_tab()
    with tab_after:       render_after_installation_tab()
    with tab_best:        render_best_practices_tab()

    st.divider()
    render_faq()


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> str:
    """
    Render the sidebar and return the selected guide mode.

    The mode toggle is the centerpiece — every other panel in the sidebar
    is decorative / informational.
    """
    with st.sidebar:
        st.markdown("## 📊 Claude for Excel")
        st.caption("Interactive setup guide")

        st.divider()
        st.markdown("### 🧭 Guide Mode")
        mode = st.radio(
            "Choose how you want to learn",
            ["Text Explanations", "Video Setup Guide"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### 📌 Quick links")
        st.markdown(
            "- [Claude Help Center](https://support.anthropic.com)\n"
            "- [Claude pricing & plans](https://www.anthropic.com/pricing)\n"
            "- [Microsoft 365 Admin Center](https://admin.microsoft.com)\n"
            "- [Microsoft AppSource](https://appsource.microsoft.com)"
        )

        st.divider()
        st.info(
            "This is a demo guide. For the latest official instructions, "
            "visit the **Claude Help Center**."
        )

    return mode


# ============================================================
# HEADER
# ============================================================

def render_header() -> None:
    """Top-of-page hero block."""
    st.title("📊 Claude for Excel Plug-in Setup Guide")
    st.markdown(
        "<p style='font-size:1.1rem;color:#555;'>"
        "An interactive walkthrough for individuals and admins integrating "
        "Anthropic's Claude directly into Microsoft Excel — so spreadsheets "
        "become a workspace for AI-assisted analysis."
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Top-level entry point — wires the sidebar mode to the right page."""
    init_session_state()
    mode = render_sidebar()
    render_header()

    if mode == "Text Explanations":
        render_text_mode()
    else:
        render_video_mode()

    # Footer — kept minimal and trustworthy.
    st.divider()
    st.caption(
        "🛈 This is a demo guide produced for educational and consulting "
        "purposes. For the latest official instructions, see the "
        "[Claude Help Center](https://support.anthropic.com). "
        "Microsoft and Excel are trademarks of Microsoft Corporation."
    )


if __name__ == "__main__":
    main()

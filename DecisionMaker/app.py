"""
================================================================================
DECISION TRACKING & ANALYSIS SYSTEM — STREAMLIT MVP
================================================================================
Helps companies track and analyse where decisions are being made by logging
key decisions and using AI (simulated) to monitor and analyse them.

Features:
  - Decision Logging: structured form for capturing who/what/why
  - AI Server Monitor: simulates detecting decisions automatically
  - Reports & Analysis: dashboards, AI insights, PDF reports

Data persistence: SQLite (in-memory via session state for MVP).

Usage:
    pip install streamlit pandas matplotlib reportlab
    streamlit run app.py
================================================================================
"""

import io
import random
import sqlite3
import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

DEPARTMENTS = [
    "Engineering", "Marketing", "Sales", "Operations", "Finance",
    "HR", "Product", "Legal", "Executive", "Other",
]

IMPACT_LEVELS = ["Low", "Medium", "High"]

# Pool of pre-written mock decisions for the AI monitor simulation
AI_SIMULATED_DECISIONS = [
    {
        "who": "System AI",
        "what": "Auto-approved vendor payment above standard threshold",
        "why": "Payment schedule alignment; approved by finance rule engine",
        "department": "Finance",
        "impact": "Medium",
    },
    {
        "who": "Sales Team Lead",
        "what": "Discounted enterprise deal by 20%",
        "why": "Competitive pressure; deal would have been lost otherwise",
        "department": "Sales",
        "impact": "High",
    },
    {
        "who": "Engineering Manager",
        "what": "Deferred sprint backlog item to next cycle",
        "why": "Resource conflict with higher-priority P1 bug",
        "department": "Engineering",
        "impact": "Low",
    },
    {
        "who": "HR Director",
        "what": "Extended remote work policy for Q3",
        "why": "Employee satisfaction survey results indicated strong preference",
        "department": "HR",
        "impact": "Medium",
    },
    {
        "who": "Product Owner",
        "what": "Prioritised mobile feature over desktop parity",
        "why": "70% of users are on mobile — data-driven prioritisation",
        "department": "Product",
        "impact": "High",
    },
    {
        "who": "Finance Controller",
        "what": "Froze discretionary budget for Q2",
        "why": "Revenue tracking 8% below forecast; prudent cash management",
        "department": "Finance",
        "impact": "High",
    },
    {
        "who": "Operations Lead",
        "what": "Switched logistics provider for northern region",
        "why": "Delivery SLA breach three months running",
        "department": "Operations",
        "impact": "Medium",
    },
]


# ── Database helpers ──────────────────────────────────────────────────────────

def _create_schema(conn: sqlite3.Connection) -> None:
    """Create the decisions table if it does not already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            who         TEXT    NOT NULL,
            what        TEXT    NOT NULL,
            why         TEXT    NOT NULL,
            department  TEXT    DEFAULT '',
            impact      TEXT    DEFAULT '',
            decided_at  TEXT    NOT NULL,
            source      TEXT    DEFAULT 'manual'
        )
    """)
    conn.commit()


def get_db_connection() -> sqlite3.Connection:
    """Return the singleton in-memory SQLite connection from session state."""
    if "db_conn" not in st.session_state:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        _create_schema(conn)
        st.session_state.db_conn = conn
    return st.session_state.db_conn


# ── Pure logic functions (unit-testable) ──────────────────────────────────────

def validate_decision(record: dict) -> tuple:
    """
    Validate a decision record.
    Returns (is_valid: bool, errors: list[str]).
    """
    errors = []
    if not record.get("who", "").strip():
        errors.append("Decision maker is required.")
    if not record.get("what", "").strip():
        errors.append("Decision description is required.")
    if not record.get("why", "").strip():
        errors.append("Rationale is required.")
    return len(errors) == 0, errors


def save_decision(conn: sqlite3.Connection, record: dict) -> int:
    """Insert a validated decision into the DB. Returns the new row id."""
    cursor = conn.execute(
        """INSERT INTO decisions (who, what, why, department, impact, decided_at, source)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            record.get("who", "").strip(),
            record.get("what", "").strip(),
            record.get("why", "").strip(),
            record.get("department", ""),
            record.get("impact", ""),
            record.get("decided_at", datetime.datetime.now().isoformat()),
            record.get("source", "manual"),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def load_decisions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all decisions ordered newest-first. Returns an empty DataFrame on error."""
    try:
        df = pd.read_sql("SELECT * FROM decisions ORDER BY decided_at DESC", conn)
        if not df.empty:
            df["decided_at"] = pd.to_datetime(df["decided_at"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(
            columns=["id", "who", "what", "why", "department", "impact", "decided_at", "source"]
        )


def analyze_decisions(df: pd.DataFrame) -> dict:
    """
    Derive insights from a decisions DataFrame.
    Pure function — no side effects.
    Returns a dict with summary stats and insight strings.
    """
    if df.empty:
        return {
            "total": 0,
            "top_decision_maker": "N/A",
            "top_department": "N/A",
            "high_impact_count": 0,
            "insights": ["No decisions logged yet."],
        }

    total = len(df)
    top_maker = df["who"].value_counts().idxmax() if "who" in df.columns else "N/A"

    dept_series = df["department"][
        df["department"].notna() & (df["department"].astype(str) != "")
    ]
    top_dept = dept_series.value_counts().idxmax() if not dept_series.empty else "N/A"

    high_impact = int((df["impact"] == "High").sum()) if "impact" in df.columns else 0

    insights = []
    maker_count = int(df["who"].value_counts().iloc[0]) if total > 0 else 0
    insights.append(
        f"{top_maker} is the most active decision maker "
        f"with {maker_count} decision(s)."
    )
    if top_dept != "N/A":
        dept_count = int(dept_series.value_counts().iloc[0])
        insights.append(
            f"{top_dept} leads decision volume with {dept_count} decision(s)."
        )
    if high_impact > 0:
        pct = round(high_impact / total * 100)
        insights.append(
            f"{high_impact} decision(s) ({pct}%) carry High impact — "
            f"review whether all are warranted."
        )
    if total >= 5:
        insights.append(
            "Sufficient data for trend analysis. "
            "Consider reviewing decision patterns weekly."
        )

    return {
        "total": total,
        "top_decision_maker": top_maker,
        "top_department": top_dept,
        "high_impact_count": high_impact,
        "insights": insights,
    }


def get_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise decision count and impact breakdown by department.
    Pure function. Returns an empty DataFrame if df is empty.
    """
    if df.empty:
        return pd.DataFrame(columns=["Department", "Total", "High", "Medium", "Low"])

    rows = []
    for dept, group in df.groupby("department"):
        if not str(dept).strip():
            dept = "Unspecified"
        rows.append({
            "Department": str(dept),
            "Total": len(group),
            "High": int((group["impact"] == "High").sum()),
            "Medium": int((group["impact"] == "Medium").sum()),
            "Low": int((group["impact"] == "Low").sum()),
        })

    if not rows:
        return pd.DataFrame(columns=["Department", "Total", "High", "Medium", "Low"])

    return (
        pd.DataFrame(rows)
        .sort_values("Total", ascending=False)
        .reset_index(drop=True)
    )


def simulate_ai_detection(n: int = 2) -> list:
    """
    Return n mock AI-detected decision records sampled from the pool.
    Caps at the pool size; stamps each with the current timestamp.
    """
    n = min(max(n, 0), len(AI_SIMULATED_DECISIONS))
    sample = random.sample(AI_SIMULATED_DECISIONS, n)
    now = datetime.datetime.now().isoformat()
    result = []
    for item in sample:
        record = dict(item)  # copy so we don't mutate the constant pool
        record["decided_at"] = now
        record["source"] = "ai_monitor"
        result.append(record)
    return result


def generate_pdf_report(df: pd.DataFrame, analysis: dict) -> Optional[bytes]:
    """
    Build a professional PDF report from the decisions data and analysis dict.
    Returns bytes on success, or None if reportlab is not installed.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=inch, rightMargin=inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=12
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10, spaceAfter=4
    )

    story = []

    # Cover
    story.append(Paragraph("Decision Tracking & Analysis Report", title_style))
    story.append(
        Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            body_style,
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1f4e79")))
    story.append(Spacer(1, 0.15 * inch))

    # Summary metrics table
    story.append(Paragraph("Executive Summary", h2_style))
    metrics = [
        ["Metric", "Value"],
        ["Total Decisions", str(analysis.get("total", 0))],
        ["Top Decision Maker", str(analysis.get("top_decision_maker", "N/A"))],
        ["Top Department", str(analysis.get("top_department", "N/A"))],
        ["High Impact Decisions", str(analysis.get("high_impact_count", 0))],
    ]
    metrics_table = Table(metrics, colWidths=[3 * inch, 3.5 * inch])
    metrics_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(metrics_table)
    story.append(Spacer(1, 0.2 * inch))

    # AI Insights
    story.append(Paragraph("AI Insights", h2_style))
    for insight in analysis.get("insights", []):
        story.append(Paragraph(f"•  {insight}", body_style))
    story.append(Spacer(1, 0.2 * inch))

    # Decision log (first 20 rows)
    if not df.empty:
        story.append(Paragraph("Decision Log (most recent 20)", h2_style))
        log_data = [["#", "Who", "What (summary)", "Department", "Impact", "Date"]]
        for _, row in df.head(20).iterrows():
            date_str = ""
            if pd.notna(row.get("decided_at")):
                try:
                    date_str = pd.to_datetime(row["decided_at"]).strftime("%Y-%m-%d")
                except Exception:
                    date_str = str(row["decided_at"])[:10]
            what_short = str(row.get("what", ""))
            if len(what_short) > 48:
                what_short = what_short[:45] + "..."
            log_data.append([
                str(row.get("id", "")),
                str(row.get("who", ""))[:22],
                what_short,
                str(row.get("department", ""))[:14],
                str(row.get("impact", "")),
                date_str,
            ])
        log_table = Table(
            log_data,
            colWidths=[0.3 * inch, 1.2 * inch, 2.6 * inch, 1.0 * inch, 0.6 * inch, 0.8 * inch],
        )
        log_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(log_table)

    doc.build(story)
    return buffer.getvalue()


# ── UI — Decision Logging ─────────────────────────────────────────────────────

def render_decision_logging(conn: sqlite3.Connection) -> None:
    """Render the Decision Logging page: form + AI monitor section."""
    st.header("Log a Decision")

    try:
        df_existing = load_decisions(conn)
    except Exception:
        df_existing = pd.DataFrame()

    known_people = (
        sorted(df_existing["who"].dropna().unique().tolist())
        if not df_existing.empty else []
    )

    with st.form("decision_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            if known_people:
                person_options = known_people + ["+ Add new person"]
                person_choice = st.selectbox("Decision Maker", person_options)
                if person_choice == "+ Add new person":
                    who = st.text_input("Enter name", placeholder="e.g. Jane Smith")
                else:
                    who = person_choice
            else:
                who = st.text_input("Decision Maker", placeholder="e.g. Jane Smith")

            what = st.text_area(
                "What was decided?",
                placeholder="Describe the decision clearly.",
                height=100,
            )
            why = st.text_area(
                "Why was this decision made?",
                placeholder="Explain the rationale and context.",
                height=100,
            )

        with col2:
            department = st.selectbox("Department / Area", [""] + DEPARTMENTS)
            impact = st.selectbox("Impact Level", [""] + IMPACT_LEVELS)
            decided_date = st.date_input("Date", value=datetime.date.today())
            decided_time = st.time_input("Time", value=datetime.datetime.now().time())

        submitted = st.form_submit_button(
            "Submit Decision", use_container_width=True, type="primary"
        )

    if submitted:
        decided_at_dt = datetime.datetime.combine(decided_date, decided_time)
        record = {
            "who": who,
            "what": what,
            "why": why,
            "department": department,
            "impact": impact,
            "decided_at": decided_at_dt.isoformat(),
            "source": "manual",
        }
        valid, errors = validate_decision(record)
        if not valid:
            for err in errors:
                st.error(err)
        else:
            try:
                save_decision(conn, record)
                st.success("Decision logged successfully!")
            except Exception as e:
                st.error(f"Failed to save decision: {e}")

    # Recent decisions preview table
    try:
        df = load_decisions(conn)
        if not df.empty:
            st.subheader(f"Recent Decisions ({len(df)} total)")
            display = df[["who", "what", "department", "impact", "decided_at"]].head(10).copy()
            display.columns = ["Who", "What", "Department", "Impact", "Date"]
            st.dataframe(display, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load decisions: {e}")

    st.divider()
    render_ai_monitor(conn)


def render_ai_monitor(conn: sqlite3.Connection) -> None:
    """Render the AI Server Monitor collapsible section."""
    st.subheader("AI Server Monitor")
    st.caption(
        "Simulates an AI agent monitoring company activity and "
        "auto-detecting decisions for review."
    )

    if "ai_detected" not in st.session_state:
        st.session_state.ai_detected = []

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start AI Server Monitor", use_container_width=True):
            try:
                detected = simulate_ai_detection(n=random.randint(1, 3))
                st.session_state.ai_detected = detected
                st.success(
                    f"AI detected {len(detected)} decision(s). Review below."
                )
            except Exception as e:
                st.error(f"AI monitor error: {e}")

    with col2:
        if st.button("Clear AI Queue", use_container_width=True):
            st.session_state.ai_detected = []
            st.rerun()

    if st.session_state.ai_detected:
        st.info(
            f"{len(st.session_state.ai_detected)} AI-detected decision(s) "
            f"pending review."
        )
        # Iterate over a copy so we can safely remove items
        for i, item in enumerate(list(st.session_state.ai_detected)):
            with st.expander(
                f"AI Decision #{i + 1}: {str(item.get('what', ''))[:60]}"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    new_who = st.text_input(
                        "Who", value=item.get("who", ""), key=f"ai_who_{i}"
                    )
                    new_what = st.text_area(
                        "What", value=item.get("what", ""),
                        key=f"ai_what_{i}", height=80,
                    )
                with col_b:
                    new_why = st.text_area(
                        "Why", value=item.get("why", ""),
                        key=f"ai_why_{i}", height=80,
                    )
                    dept_idx = (
                        DEPARTMENTS.index(item["department"]) + 1
                        if item.get("department") in DEPARTMENTS else 0
                    )
                    new_dept = st.selectbox(
                        "Department", [""] + DEPARTMENTS,
                        index=dept_idx, key=f"ai_dept_{i}",
                    )
                    imp_idx = (
                        IMPACT_LEVELS.index(item["impact"]) + 1
                        if item.get("impact") in IMPACT_LEVELS else 0
                    )
                    new_impact = st.selectbox(
                        "Impact", [""] + IMPACT_LEVELS,
                        index=imp_idx, key=f"ai_impact_{i}",
                    )

                if st.button(
                    f"Save Decision #{i + 1}", key=f"ai_save_{i}", type="primary"
                ):
                    record = {
                        "who": new_who, "what": new_what, "why": new_why,
                        "department": new_dept, "impact": new_impact,
                        "decided_at": item.get(
                            "decided_at", datetime.datetime.now().isoformat()
                        ),
                        "source": "ai_monitor",
                    }
                    valid, errors = validate_decision(record)
                    if not valid:
                        for err in errors:
                            st.error(err)
                    else:
                        try:
                            save_decision(conn, record)
                            st.session_state.ai_detected.pop(i)
                            st.success("AI decision saved!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save: {e}")


# ── UI — Reports & Analysis ───────────────────────────────────────────────────

def render_reports_analysis(conn: sqlite3.Connection) -> None:
    """Render the Reports & Analysis page."""
    st.header("Reports & Analysis")

    try:
        df = load_decisions(conn)
    except Exception as e:
        st.error(f"Could not load decisions: {e}")
        return

    if df.empty:
        st.info(
            "No decisions logged yet. "
            "Go to Decision Logging to start capturing decisions."
        )
        return

    # ── Metric cards ──────────────────────────────────────────────────────────
    analysis = analyze_decisions(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Decisions", analysis["total"])
    c2.metric("Top Decision Maker", analysis["top_decision_maker"])
    c3.metric("Top Department", analysis["top_department"])
    c4.metric("High Impact", analysis["high_impact_count"])

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        try:
            maker_counts = df["who"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(maker_counts.index[::-1], maker_counts.values[::-1],
                    color="#1f4e79")
            ax.set_title("Decisions by Person")
            ax.set_xlabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not render chart: {e}")

    with col_right:
        try:
            dept_df = df[df["department"].notna() & (df["department"] != "")]
            if not dept_df.empty:
                dept_counts = dept_df["department"].value_counts()
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.pie(
                    dept_counts.values,
                    labels=dept_counts.index,
                    autopct="%1.0f%%",
                    startangle=90,
                )
                ax2.set_title("Decisions by Department")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
        except Exception as e:
            st.warning(f"Could not render chart: {e}")

    # Impact bar chart
    try:
        impact_df = df[df["impact"].notna() & (df["impact"] != "")]
        if not impact_df.empty:
            impact_counts = (
                impact_df["impact"]
                .value_counts()
                .reindex(IMPACT_LEVELS, fill_value=0)
            )
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            bar_colors = ["#4CAF50", "#FF9800", "#F44336"]
            ax3.bar(impact_counts.index, impact_counts.values, color=bar_colors)
            ax3.set_title("Decisions by Impact Level")
            ax3.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
    except Exception as e:
        st.warning(f"Could not render impact chart: {e}")

    # Department summary table
    try:
        dept_summary = get_department_summary(df)
        if not dept_summary.empty:
            st.subheader("Department Breakdown")
            st.dataframe(dept_summary, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not build department summary: {e}")

    st.divider()

    # ── AI Analysis ───────────────────────────────────────────────────────────
    st.subheader("AI Analysis")
    if st.button("Run AI Analysis", type="primary"):
        try:
            with st.spinner("Analysing decisions…"):
                result = analyze_decisions(df)
                st.session_state.last_analysis = result
        except Exception as e:
            st.error(f"Analysis failed: {e}")

    if "last_analysis" in st.session_state:
        st.success("Analysis complete!")
        for insight in st.session_state.last_analysis.get("insights", []):
            st.write(f"• {insight}")

    st.divider()

    # ── PDF Report ────────────────────────────────────────────────────────────
    st.subheader("PDF Report")
    if not REPORTLAB_AVAILABLE:
        st.warning(
            "Install `reportlab` to enable PDF generation: "
            "`pip install reportlab`"
        )
    else:
        if st.button("Generate PDF Report"):
            try:
                with st.spinner("Building report…"):
                    analysis = analyze_decisions(df)
                    pdf_bytes = generate_pdf_report(df, analysis)
                if pdf_bytes:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=(
                            f"decision_report_{datetime.date.today()}.pdf"
                        ),
                        mime="application/pdf",
                        type="primary",
                    )
                    st.success("PDF ready — click above to download.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Decision Tracker",
        page_icon="🧭",
        layout="wide",
    )
    st.title("Decision Tracking & Analysis System")
    st.caption(
        "Track and analyse where decisions are actually being made "
        "in your organisation."
    )

    try:
        conn = get_db_connection()
    except Exception as e:
        st.error(f"Database initialisation failed: {e}")
        st.stop()

    page = st.sidebar.radio(
        "Navigation",
        ["Decision Logging", "Reports & Analysis"],
        index=0,
    )

    if page == "Decision Logging":
        render_decision_logging(conn)
    else:
        render_reports_analysis(conn)


if __name__ == "__main__":
    main()

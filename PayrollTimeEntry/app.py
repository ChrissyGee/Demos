"""
================================================================================
PAYROLL MANAGER & TIME ENTRY SYSTEM — STREAMLIT MVP
================================================================================
Self-contained Streamlit app implementing the full payroll + time-entry flow.

Architecture:
  Employer registers employees (name, ID, allowed hours, pay rate) →
  Employees submit weekly time entries →
  AI agent validates entries against employer rules →
  Employer approves (individually or in bulk) →
  PDF payroll reports and pay stubs are generated on demand.

Portals:
  - Employer Dashboard  (Tab 1: Employee Setup | Tab 2: Time Review & Approval)
  - Employee Portal     (Tab 1: Time Entry    | Tab 2: Pay Stubs)

Data persistence: session state (in-memory for MVP — no external DB required).

Usage:
    pip install streamlit pandas reportlab
    streamlit run app.py
================================================================================
"""

import io
import datetime
from typing import Optional

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

PAY_RATE_DEFAULT = 25.0   # fallback hourly rate ($/hr) if not set per employee
CURRENT_MONTH = datetime.date.today().strftime("%Y-%m")
CURRENT_MONTH_LABEL = datetime.date.today().strftime("%B %Y")


# ── Session state ─────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialise all required session state keys with safe defaults."""
    defaults: dict = {
        "employees": [],      # list[dict] — registered employee records
        "time_entries": [],   # list[dict] — submitted time-entry records
        "ai_flags": [],       # list[dict] — AI validation flag records
        "chat_history": [],   # list[dict] — employer chat messages
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Pure logic functions (unit-testable) ──────────────────────────────────────

def validate_employee(record: dict) -> tuple:
    """
    Validate a new employee record.
    Returns (is_valid: bool, errors: list[str]).
    """
    errors = []
    if not str(record.get("name", "")).strip():
        errors.append("Employee name is required.")
    if not str(record.get("employee_id", "")).strip():
        errors.append("Employee ID is required.")
    try:
        hrs = float(record.get("allowed_hours_per_month", 0))
        if hrs <= 0:
            errors.append("Allowed hours per month must be a positive number.")
    except (TypeError, ValueError):
        errors.append("Allowed hours per month must be a valid number.")
    return len(errors) == 0, errors


def validate_time_entry(entry: dict, allowed_hours_per_week: float = 40.0) -> tuple:
    """
    Validate a submitted time entry against basic rules.
    Returns (is_valid: bool, errors: list[str]).
    """
    errors = []
    hours_map = entry.get("hours") or {}

    if not isinstance(hours_map, dict):
        errors.append("Time entry hours must be a mapping of day → hours.")
        return False, errors

    total = sum(hours_map.values())

    if total <= 0:
        errors.append("Total hours must be greater than zero.")

    weekly_ceiling = allowed_hours_per_week * 7
    if total > weekly_ceiling:
        errors.append(
            f"Hours ({total:.1f}) exceed the weekly ceiling "
            f"({weekly_ceiling:.0f}h for a {allowed_hours_per_week:.0f}h/day limit)."
        )

    for day, hrs in hours_map.items():
        if hrs < 0:
            errors.append(f"Hours for {day} cannot be negative.")
        if hrs > 24:
            errors.append(f"Hours for {day} cannot exceed 24.")

    return len(errors) == 0, errors


def get_total_hours(entry: dict) -> float:
    """Sum all daily hours in a time-entry record. Returns 0.0 if malformed."""
    hours_map = entry.get("hours")
    if not isinstance(hours_map, dict):
        return 0.0
    return float(sum(hours_map.values()))


def calculate_pay(hours: float, rate: float) -> float:
    """Calculate gross pay for the given hours at the given hourly rate."""
    return round(max(0.0, hours) * max(0.0, rate), 2)


def ai_validate_time_entry(entry: dict, employee: dict) -> dict:
    """
    Simulate AI validation of a time entry against employer-defined rules.
    Returns a flag dict: {employee_id, status ('ok'|'warn'|'reject'), message}.
    """
    emp_id = entry.get("employee_id", "")
    allowed = float(employee.get("allowed_hours_per_month", 160))
    total = get_total_hours(entry)

    if total <= 0:
        return {
            "employee_id": emp_id,
            "status": "reject",
            "message": "Zero hours submitted — entry rejected.",
        }
    if total > allowed:
        return {
            "employee_id": emp_id,
            "status": "reject",
            "message": (
                f"Submitted {total:.1f}h exceeds the allowed "
                f"{allowed:.0f}h/month limit."
            ),
        }
    if total > allowed * 0.9:
        return {
            "employee_id": emp_id,
            "status": "warn",
            "message": (
                f"Submitted {total:.1f}h is within 10% of the "
                f"{allowed:.0f}h monthly limit."
            ),
        }
    return {
        "employee_id": emp_id,
        "status": "ok",
        "message": "Time entry is within permitted limits.",
    }


def get_employee_by_id(employees: list, employee_id: str) -> Optional[dict]:
    """Return the employee dict matching employee_id, or None if not found."""
    for emp in employees:
        if emp.get("employee_id") == employee_id:
            return emp
    return None


def get_entries_for_employee(time_entries: list, employee_id: str) -> list:
    """Return all time entries belonging to the given employee_id."""
    return [e for e in time_entries if e.get("employee_id") == employee_id]


def summarise_employee_hours(time_entries: list) -> pd.DataFrame:
    """
    Build a flat DataFrame of one row per submitted time-entry record.
    Columns: employee_id, name, week, total_hours, status, month.
    """
    rows = []
    for entry in time_entries:
        rows.append({
            "employee_id": entry.get("employee_id", ""),
            "name": entry.get("employee_name", ""),
            "week": entry.get("week_label", ""),
            "total_hours": get_total_hours(entry),
            "status": entry.get("status", "pending"),
            "month": entry.get("month", ""),
        })
    if not rows:
        return pd.DataFrame(
            columns=["employee_id", "name", "week", "total_hours", "status", "month"]
        )
    return pd.DataFrame(rows)


def generate_payroll_report(
    employees: list, time_entries: list, month: str = CURRENT_MONTH
) -> Optional[bytes]:
    """
    Generate a PDF payroll report for the specified month.
    Returns bytes on success, or None if reportlab is unavailable.
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
        "ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=10
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=12, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10, spaceAfter=4
    )

    story = []

    story.append(Paragraph(f"Payroll Report — {month}", title_style))
    story.append(
        Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            body_style,
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a5276"))
    )
    story.append(Spacer(1, 0.15 * inch))

    month_entries = [e for e in time_entries if e.get("month") == month]

    # Per-employee summary table
    story.append(Paragraph("Employee Hours Summary", h2_style))
    header = ["Employee", "ID", "Allowed Hrs/Mo", "Submitted Hrs", "Status"]
    table_data = [header]

    all_submitted: list = []
    for emp in employees:
        emp_entries = [
            e for e in month_entries
            if e.get("employee_id") == emp.get("employee_id")
        ]
        submitted = sum(get_total_hours(e) for e in emp_entries)
        all_submitted.append(submitted)
        statuses = {e.get("status", "pending") for e in emp_entries}
        if not emp_entries:
            status_str = "no entry"
        elif statuses == {"approved"}:
            status_str = "approved"
        else:
            status_str = "pending"
        table_data.append([
            str(emp.get("name", "")),
            str(emp.get("employee_id", "")),
            str(emp.get("allowed_hours_per_month", "")),
            f"{submitted:.1f}",
            status_str,
        ])

    if len(table_data) > 1:
        col_widths = [1.8 * inch, 1.0 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch]
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf2ff")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)

    # Aggregate metrics
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Aggregate Metrics", h2_style))

    non_zero = [h for h in all_submitted if h > 0]
    if non_zero and employees:
        avg_hrs = sum(non_zero) / len(non_zero)
        top_idx = all_submitted.index(max(all_submitted))
        low_idx = all_submitted.index(min(all_submitted))
        top_name = employees[top_idx].get("name", "N/A")
        low_name = employees[low_idx].get("name", "N/A")
        approved_count = len([e for e in month_entries if e.get("status") == "approved"])

        metrics_data = [
            ["Metric", "Value"],
            ["Average Hours / Employee", f"{avg_hrs:.1f}"],
            ["Top Performer (most hours)", top_name],
            ["Lower Performer (fewest hours)", low_name],
            ["Approved Entries This Month", str(approved_count)],
        ]
        m_table = Table(metrics_data, colWidths=[3 * inch, 3.5 * inch])
        m_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf2ff")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(m_table)
    else:
        story.append(Paragraph("No time entries submitted for this month.", body_style))

    doc.build(story)
    return buffer.getvalue()


def generate_pay_stub(
    employee: dict, time_entries: list, month: str = CURRENT_MONTH
) -> Optional[bytes]:
    """
    Generate a single-employee pay stub PDF for the given month.
    Returns bytes on success, or None if reportlab is unavailable.
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
        "StubTitle", parent=styles["Title"], fontSize=16, spaceAfter=8
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=12, spaceAfter=4
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10, spaceAfter=4
    )

    story = []

    story.append(Paragraph(f"Pay Stub — {month}", title_style))
    story.append(
        Paragraph(
            f"Employee: {employee.get('name', 'N/A')}  |  "
            f"ID: {employee.get('employee_id', 'N/A')}  |  "
            f"Role: {employee.get('role', 'N/A')}",
            body_style,
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a5276"))
    )
    story.append(Spacer(1, 0.1 * inch))

    emp_entries = [
        e for e in time_entries
        if e.get("employee_id") == employee.get("employee_id")
        and e.get("month") == month
    ]
    rate = float(employee.get("pay_rate", PAY_RATE_DEFAULT))
    total_hrs = sum(get_total_hours(e) for e in emp_entries)
    gross_pay = calculate_pay(total_hrs, rate)

    story.append(Paragraph("Hours Detail", h2_style))
    if emp_entries:
        tbl_data = [["Week", "Hours", "Rate ($/hr)", "Gross ($)"]]
        for entry in emp_entries:
            hrs = get_total_hours(entry)
            tbl_data.append([
                entry.get("week_label", ""),
                f"{hrs:.1f}",
                f"{rate:.2f}",
                f"{calculate_pay(hrs, rate):.2f}",
            ])
        tbl_data.append(["TOTAL", f"{total_hrs:.1f}", "", f"{gross_pay:.2f}"])

        t = Table(tbl_data, colWidths=[2.0 * inch, 1.0 * inch, 1.5 * inch, 1.5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#eaf2ff")]),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#d4e6f1")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No time entries for this period.", body_style))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Gross Pay: ${gross_pay:.2f}", h2_style))

    doc.build(story)
    return buffer.getvalue()


def build_chat_response(prompt: str, employees: list, time_entries: list) -> str:
    """
    Generate a rule-based AI chat response from payroll data.
    Pure function — receives data as parameters rather than reading session state.
    """
    pl = prompt.lower()

    if not employees:
        return (
            "No employees are registered yet. "
            "Please set up employees in the Employer Dashboard first."
        )

    if any(kw in pl for kw in ["how many employee", "total employee", "number of employee"]):
        return f"There are **{len(employees)}** registered employee(s)."

    if any(kw in pl for kw in ["most hours", "top performer", "highest hours"]):
        if time_entries:
            totals = {
                emp["employee_id"]: sum(
                    get_total_hours(e) for e in time_entries
                    if e.get("employee_id") == emp["employee_id"]
                )
                for emp in employees
            }
            if totals:
                top_id = max(totals, key=totals.get)
                top_emp = get_employee_by_id(employees, top_id)
                name = top_emp.get("name", top_id) if top_emp else top_id
                return (
                    f"The employee with the most submitted hours is "
                    f"**{name}** with **{totals[top_id]:.1f}h**."
                )
        return "No time entries have been submitted yet."

    if "pending" in pl:
        count = sum(1 for e in time_entries if e.get("status") == "pending")
        return f"There are currently **{count}** pending time entry/entries awaiting approval."

    if "approved" in pl:
        count = sum(1 for e in time_entries if e.get("status") == "approved")
        return f"There are **{count}** approved time entry/entries."

    if any(kw in pl for kw in ["average", "avg hours"]):
        if time_entries:
            emp_totals = [
                sum(get_total_hours(e) for e in time_entries
                    if e.get("employee_id") == emp["employee_id"])
                for emp in employees
            ]
            non_zero = [h for h in emp_totals if h > 0]
            if non_zero:
                avg = sum(non_zero) / len(non_zero)
                return f"Average submitted hours per employee: **{avg:.1f}h**."
        return "Not enough data for an average calculation yet."

    return (
        f"I have access to **{len(employees)}** employee(s) and "
        f"**{len(time_entries)}** time entry/entries. "
        "Try asking about: 'top performer', 'pending approvals', 'average hours', "
        "or 'how many employees'."
    )


# ── UI — Employer Dashboard ───────────────────────────────────────────────────

def render_employer_employee_setup() -> None:
    """Employer Tab 1: Register and view employees."""
    st.subheader("Employee Setup")
    st.caption("Register employees and set their allowed hours and pay rate.")

    with st.form("add_employee_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", placeholder="e.g. Alice Johnson")
            employee_id = st.text_input("Employee ID", placeholder="e.g. EMP001")
            role = st.text_input("Role / Title", placeholder="e.g. Software Engineer")
        with col2:
            allowed_hours = st.number_input(
                "Allowed Hours / Month",
                min_value=1.0, max_value=744.0,
                value=160.0, step=8.0,
            )
            pay_rate = st.number_input(
                "Hourly Pay Rate ($)",
                min_value=0.01, max_value=9999.0,
                value=PAY_RATE_DEFAULT, step=0.50,
            )
        add_btn = st.form_submit_button(
            "Add Employee", use_container_width=True, type="primary"
        )

    if add_btn:
        record = {
            "name": name,
            "employee_id": employee_id,
            "allowed_hours_per_month": allowed_hours,
            "pay_rate": pay_rate,
            "role": role,
        }
        valid, errors = validate_employee(record)
        if not valid:
            for err in errors:
                st.error(err)
        else:
            existing_ids = {e["employee_id"] for e in st.session_state.employees}
            if employee_id in existing_ids:
                st.warning(f"Employee ID '{employee_id}' is already registered.")
            else:
                try:
                    st.session_state.employees.append(record)
                    st.success(f"Employee '{name}' added successfully.")
                except Exception as e:
                    st.error(f"Failed to add employee: {e}")

    if st.session_state.employees:
        st.subheader(f"Registered Employees ({len(st.session_state.employees)})")
        try:
            df = pd.DataFrame(st.session_state.employees)[
                ["name", "employee_id", "role", "allowed_hours_per_month", "pay_rate"]
            ]
            df.columns = ["Name", "ID", "Role", "Allowed Hrs/Mo", "Pay Rate ($/hr)"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not display employee table: {e}")
    else:
        st.info("No employees registered yet. Use the form above to add the first one.")


def render_employer_time_review() -> None:
    """Employer Tab 2: Review, filter, and approve time entries."""
    st.subheader("Time Entry Review & Approval")

    if not st.session_state.employees:
        st.warning("No employees registered. Set up employees first.")
        return

    try:
        entries_df = summarise_employee_hours(st.session_state.time_entries)
    except Exception as e:
        st.error(f"Could not load time entries: {e}")
        return

    month_df = (
        entries_df[entries_df["month"] == CURRENT_MONTH].copy()
        if not entries_df.empty else pd.DataFrame()
    )

    if month_df.empty:
        st.info(f"No time entries submitted for {CURRENT_MONTH_LABEL} yet.")
    else:
        search = st.text_input(
            "Search by employee name", placeholder="Type a name to filter…"
        )
        if search:
            month_df = month_df[
                month_df["name"].str.contains(search, case=False, na=False)
            ]

        try:
            st.dataframe(
                month_df[["name", "employee_id", "week", "total_hours", "status"]].rename(
                    columns={
                        "name": "Name", "employee_id": "ID",
                        "week": "Week", "total_hours": "Hours", "status": "Status",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.warning(f"Could not render table: {e}")

        # Individual per-employee approve buttons
        st.subheader("Individual Approvals")
        for emp in st.session_state.employees:
            emp_id = emp["employee_id"]
            emp_idxs = [
                i for i, e in enumerate(st.session_state.time_entries)
                if e.get("employee_id") == emp_id
                and e.get("month") == CURRENT_MONTH
            ]
            if not emp_idxs:
                continue
            pending_idxs = [
                i for i in emp_idxs
                if st.session_state.time_entries[i].get("status") == "pending"
            ]
            total_hrs = sum(
                get_total_hours(st.session_state.time_entries[i])
                for i in emp_idxs
            )
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(
                    f"**{emp['name']}** — "
                    f"{total_hrs:.1f}h submitted, "
                    f"{len(pending_idxs)} pending"
                )
            with col2:
                if pending_idxs and st.button(
                    "Approve", key=f"approve_{emp_id}", type="primary"
                ):
                    try:
                        for i in pending_idxs:
                            st.session_state.time_entries[i]["status"] = "approved"
                        st.success(f"Approved {len(pending_idxs)} entry/entries for {emp['name']}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Approval failed: {e}")

    # Bulk approve + PDF report row
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Approve All (This Month)", use_container_width=True):
            try:
                count = 0
                for entry in st.session_state.time_entries:
                    if (entry.get("month") == CURRENT_MONTH
                            and entry.get("status") == "pending"):
                        entry["status"] = "approved"
                        count += 1
                st.success(
                    f"Approved {count} entry/entries for {CURRENT_MONTH_LABEL}."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Bulk approval failed: {e}")

    with col_b:
        if not REPORTLAB_AVAILABLE:
            st.caption("Install `reportlab` to enable PDF reports.")
        elif st.button("Generate PDF Report", use_container_width=True):
            try:
                with st.spinner("Building payroll report…"):
                    pdf = generate_payroll_report(
                        st.session_state.employees,
                        st.session_state.time_entries,
                        CURRENT_MONTH,
                    )
                if pdf:
                    st.download_button(
                        label="Download Payroll Report PDF",
                        data=pdf,
                        file_name=f"payroll_{CURRENT_MONTH}.pdf",
                        mime="application/pdf",
                    )
            except Exception as e:
                st.error(f"Report generation failed: {e}")

    # AI validation flags panel
    active_flags = [
        f for f in st.session_state.ai_flags if f.get("status") != "ok"
    ]
    if active_flags:
        st.divider()
        st.subheader("AI Validation Flags")
        for flag in active_flags:
            if flag["status"] == "reject":
                st.error(
                    f"**{flag.get('employee_id', '')}**: {flag['message']}"
                )
            else:
                st.warning(
                    f"**{flag.get('employee_id', '')}**: {flag['message']}"
                )

    # Employer AI chat
    st.divider()
    render_employer_chat()


def render_employer_chat() -> None:
    """Employer chat interface — simple rule-based AI Q&A about payroll data."""
    st.subheader("Ask About Employees / Time Entries")
    st.caption("AI-powered Q&A (simulated — no API key required).")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("e.g. Who submitted the most hours?")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            response = build_chat_response(
                prompt,
                st.session_state.employees,
                st.session_state.time_entries,
            )
        except Exception as e:
            response = f"Sorry, an error occurred while processing your question: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


# ── UI — Employee Portal ──────────────────────────────────────────────────────

def render_employee_time_entry() -> None:
    """Employee Portal Tab 1: Weekly time entry form."""
    st.subheader("Submit Time Entry")

    if not st.session_state.employees:
        st.warning(
            "No employee profiles found. "
            "Please ask your employer to register you in the system."
        )
        return

    emp_labels = [
        f"{e['name']} ({e['employee_id']})" for e in st.session_state.employees
    ]
    selected_label = st.selectbox("Select your name", emp_labels)
    emp_idx = emp_labels.index(selected_label)
    employee = st.session_state.employees[emp_idx]

    st.info(
        f"Allowed hours / month: **{employee.get('allowed_hours_per_month', 160):.0f}h** "
        f"| Pay rate: **${employee.get('pay_rate', PAY_RATE_DEFAULT):.2f}/hr**"
    )

    # Week selector — last 5 weeks starting on Monday
    today = datetime.date.today()
    week_start_this = today - datetime.timedelta(days=today.weekday())
    week_options = [
        (week_start_this - datetime.timedelta(weeks=i)).strftime("%Y-%m-%d")
        for i in range(5)
    ]
    week_label = st.selectbox("Week starting (Monday)", week_options)
    week_start_date = datetime.date.fromisoformat(week_label)
    day_labels = [
        (week_start_date + datetime.timedelta(days=i)).strftime("%a %d %b")
        for i in range(7)
    ]

    st.subheader("Hours per Day")
    hours: dict = {}
    cols = st.columns(7)
    for i, day in enumerate(day_labels):
        with cols[i]:
            hours[day] = st.number_input(
                day, min_value=0.0, max_value=24.0,
                value=0.0, step=0.5, key=f"hrs_{week_label}_{i}",
            )

    total_hrs = sum(hours.values())
    st.metric("Total Hours This Week", f"{total_hrs:.1f}h")

    notes = st.text_area(
        "Notes (optional)",
        placeholder="Overtime reason, special circumstances, etc.",
    )

    if st.button("Submit Time Entry", type="primary", use_container_width=True):
        entry = {
            "employee_id": employee["employee_id"],
            "employee_name": employee["name"],
            "week_label": week_label,
            "month": week_start_date.strftime("%Y-%m"),
            "hours": hours,
            "notes": notes,
            "status": "pending",
            "submitted_at": datetime.datetime.now().isoformat(),
        }
        valid, errors = validate_time_entry(entry, allowed_hours_per_week=40.0)
        if not valid:
            for err in errors:
                st.error(err)
        else:
            try:
                flag = ai_validate_time_entry(entry, employee)
                if flag["status"] == "reject":
                    st.error(
                        f"AI Validation rejected: {flag['message']} — "
                        "Please adjust your hours and resubmit."
                    )
                else:
                    if flag["status"] == "warn":
                        st.warning(f"AI Note: {flag['message']}")
                    st.session_state.time_entries.append(entry)
                    # Store non-ok flags for the employer to review
                    if flag["status"] != "ok":
                        existing_ids = {f["employee_id"] for f in st.session_state.ai_flags}
                        if employee["employee_id"] not in existing_ids:
                            st.session_state.ai_flags.append(flag)
                    st.success(
                        f"Time entry submitted for week of {week_label}. "
                        "Awaiting employer approval."
                    )
            except Exception as e:
                st.error(f"Submission failed: {e}")


def render_employee_pay_stubs() -> None:
    """Employee Portal Tab 2: View and download pay stubs."""
    st.subheader("Pay Stubs")

    if not st.session_state.employees:
        st.warning("No employee profiles found.")
        return

    emp_labels = [
        f"{e['name']} ({e['employee_id']})" for e in st.session_state.employees
    ]
    selected_label = st.selectbox(
        "Select your name", emp_labels, key="stub_emp_select"
    )
    emp_idx = emp_labels.index(selected_label)
    employee = st.session_state.employees[emp_idx]

    emp_entries = get_entries_for_employee(
        st.session_state.time_entries, employee["employee_id"]
    )

    if not emp_entries:
        st.info("No time entries found for your account yet.")
        return

    months = sorted(
        {e.get("month", "") for e in emp_entries if e.get("month")},
        reverse=True,
    )
    st.write(f"Pay stubs available for **{len(months)}** month(s):")

    for month in months:
        month_hrs = sum(
            get_total_hours(e) for e in emp_entries if e.get("month") == month
        )
        gross = calculate_pay(month_hrs, float(employee.get("pay_rate", PAY_RATE_DEFAULT)))

        with st.expander(f"{month} — {month_hrs:.1f}h — ${gross:.2f} gross"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Hours", f"{month_hrs:.1f}h")
            col2.metric("Rate", f"${float(employee.get('pay_rate', PAY_RATE_DEFAULT)):.2f}/hr")
            col3.metric("Gross Pay", f"${gross:.2f}")

            if REPORTLAB_AVAILABLE:
                if st.button(f"Download Pay Stub — {month}", key=f"stub_{month}"):
                    try:
                        pdf = generate_pay_stub(
                            employee, st.session_state.time_entries, month
                        )
                        if pdf:
                            st.download_button(
                                label=f"Save {month} Pay Stub PDF",
                                data=pdf,
                                file_name=(
                                    f"paystub_{employee['employee_id']}_{month}.pdf"
                                ),
                                mime="application/pdf",
                                key=f"dl_stub_{month}",
                            )
                    except Exception as e:
                        st.error(f"Failed to generate pay stub: {e}")
            else:
                st.caption("Install `reportlab` to enable PDF pay stubs.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Payroll Manager",
        page_icon="💼",
        layout="wide",
    )
    st.title("Payroll Manager & Time Entry System")
    st.caption(f"Managing payroll for {CURRENT_MONTH_LABEL}")

    init_session_state()

    portal = st.sidebar.radio(
        "Portal",
        ["Employer Dashboard", "Employee Portal"],
        index=0,
    )

    if portal == "Employer Dashboard":
        st.header("Employer Dashboard")
        tab1, tab2 = st.tabs(["Employee Setup", "Time Entry Review & Approval"])
        with tab1:
            render_employer_employee_setup()
        with tab2:
            render_employer_time_review()
    else:
        st.header("Employee Portal")
        tab1, tab2 = st.tabs(["Time Entry", "Pay Stubs"])
        with tab1:
            render_employee_time_entry()
        with tab2:
            render_employee_pay_stubs()


if __name__ == "__main__":
    main()

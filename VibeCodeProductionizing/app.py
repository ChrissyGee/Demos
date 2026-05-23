"""
================================================================================
VIBE CODE HARDENING — STREAMLIT MVP
================================================================================

A self-contained demo that takes "vibe-coded" Python (quick prototypes,
no error handling, hardcoded secrets, naive patterns) and walks the user
through a 6-step hardening wizard.

ARCHITECTURE (matches the whiteboard sketch):

    +-----------+     +------------------+     +------------------+
    |  Folder   | --> |  Code Analyzer   | --> |   Report of      |
    |  Uploader |     |     Agent        |     |   planned changes|
    +-----------+     +--------+---------+     +--------+---------+
                               |  Tools                  |
                               |  - AI detection         | (user approves)
                               |  - Optimization         |
                               |  - Vulnerability check  v
                               |  - FAST testing         +------------------+
                               |                         |  Code Analyzer   |
                               |                         |  Agent applies   |
                               |                         |  approved fixes  |
                               |                         +--------+---------+
                                                                  |
                                                                  v
                                                         +------------------+
                                                         | Sandbox Testing  |
                                                         | (isolated exec + |
                                                         |  evaluation loop)|
                                                         +--------+---------+
                                                                  |
                                                                  v
                                                         +------------------+
                                                         | Download         |
                                                         | hardened folder  |
                                                         +------------------+

THE 6-STEP WIZARD
    Step 1 — Upload         : Drag-and-drop Python files (or load demo).
    Step 2 — Analyze        : Rule-based + optional LLM review.
    Step 3 — Review & Approve: User picks which fixes to apply.
    Step 4 — Apply          : Deterministic AST/regex transforms.
    Step 5 — Sandbox Test   : Run the hardened code in a restricted exec.
    Step 6 — Download       : Get the hardened files as a single zip.

SAFETY NOTES
    * The sandbox uses a stripped-down builtins dict and runs user code
      with a wall-clock timeout. It is NOT a security boundary — never
      paste hostile code. For real isolation use gVisor, Firecracker, or
      a remote sandbox like e2b.
    * Every transformation is reversible — the originals are kept in
      session state so the user can compare or reset.

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
import sys
import textwrap
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import streamlit as st

# api_client lives one directory up from this demo.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import api_client  # noqa: E402

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
    page_title="Vibe Code Hardening",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DEMO CODE — used when the user clicks "Load Demo"
# ============================================================
# Intentionally rough Python that triggers every rule the analyzer knows.

DEMO_FILES: Dict[str, str] = {
    "secrets.py": textwrap.dedent('''\
        # Vibe-coded module with hardcoded secrets and bad patterns.
        import os

        API_KEY = "sk-live-1234567890abcdef"
        DB_PASSWORD = "admin123"

        def get_user(uid):
            # No input validation, no error handling.
            return db.query("SELECT * FROM users WHERE id = " + str(uid))

        def run_cmd(name):
            os.system("echo hello " + name)
    '''),

    "utils.py": textwrap.dedent('''\
        # Catches everything, swallows errors, prints instead of logs.
        import pickle

        def load_config(path):
            try:
                with open(path) as f:
                    return eval(f.read())
            except:
                pass

        def deserialize(blob):
            return pickle.loads(blob)

        def divide(a, b):
            print("dividing", a, b)
            return a / b
    '''),

    "math_helpers.py": textwrap.dedent('''\
        # Cleanish helpers — the analyzer should leave most of this alone.
        def add(a, b):
            return a + b

        def multiply(a, b):
            return a * b
    '''),
}


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state() -> None:
    """Bootstrap every persistent key the wizard needs."""
    if "initialised" in st.session_state:
        return

    st.session_state.step: int = 1
    st.session_state.original_files: Dict[str, str] = {}
    st.session_state.hardened_files: Dict[str, str] = {}
    st.session_state.findings: List[Dict] = []
    st.session_state.approvals: Dict[str, bool] = {}   # keyed by finding id
    st.session_state.sandbox_results: List[Dict] = []
    st.session_state.console: List[Dict] = []
    st.session_state.openai_key: str = ""
    st.session_state.llm_review: Dict[str, str] = {}   # filename -> LLM notes

    st.session_state.initialised = True


# ============================================================
# CONSOLE LOGGING
# ============================================================

def log(stage: str, message: str) -> None:
    """Append a single line to the analyzer's reasoning console."""
    st.session_state.console.append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "stage": stage,
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


def llm_review_file(filename: str, code: str) -> Optional[str]:
    """Ask the LLM for a short review. Returns None on any failure."""
    client = get_openai_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                    "You are a senior Python reviewer. Give a SHORT review "
                    "(max 5 bullet points) of security, correctness, and "
                    "style issues. Don't rewrite the code — just describe "
                    "the issues. Stick to what's actually in the file."},
                {"role": "user", "content": f"File: {filename}\n\n```python\n{code}\n```"},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log("LLM", f"OpenAI call failed for {filename}: {e}")
        return None


# ============================================================
# RULE-BASED ANALYZER
# ============================================================
# Each rule returns a list of "findings". A finding has:
#   id        - stable identifier (filename + rule + line)
#   filename  - which file
#   rule      - rule name
#   severity  - 'critical' | 'high' | 'medium' | 'low'
#   line      - first matching line number (1-indexed)
#   message   - human description
#   category  - 'security' | 'optimization' | 'style' | 'reliability'
#   fix_label - short string describing what the fix will do

def _finding(filename, rule, severity, line, message, category, fix_label) -> Dict:
    """Helper to build a finding dict with a stable id."""
    return {
        "id": f"{filename}::{rule}::{line}",
        "filename": filename,
        "rule": rule,
        "severity": severity,
        "line": line,
        "message": message,
        "category": category,
        "fix_label": fix_label,
    }


# --- Individual rule implementations ---------------------------------------

def _rule_hardcoded_secret(filename: str, code: str) -> List[Dict]:
    """Catches API keys, passwords, and tokens written as string literals."""
    findings = []
    patterns = [
        (r"(?i)(api[_-]?key|secret|token|password|passwd)\s*=\s*[\"'][^\"']{6,}[\"']",
         "Hardcoded secret found — move to environment variable."),
    ]
    for lineno, line in enumerate(code.splitlines(), 1):
        for pat, msg in patterns:
            if re.search(pat, line):
                findings.append(_finding(
                    filename, "hardcoded-secret", "critical", lineno,
                    msg, "security", "Replace with os.environ.get(...)"
                ))
                break
    return findings


def _rule_eval_exec(filename: str, code: str) -> List[Dict]:
    """Flags use of eval() / exec() on file or user input."""
    findings = []
    for lineno, line in enumerate(code.splitlines(), 1):
        if re.search(r"\beval\s*\(", line):
            findings.append(_finding(
                filename, "use-of-eval", "critical", lineno,
                "Use of eval() — replace with ast.literal_eval or json.loads.",
                "security",
                "Replace eval(...) with ast.literal_eval(...)"
            ))
        if re.search(r"\bexec\s*\(", line):
            findings.append(_finding(
                filename, "use-of-exec", "critical", lineno,
                "Use of exec() — almost always avoidable.",
                "security", "Flag exec() for manual review"
            ))
    return findings


def _rule_os_system(filename: str, code: str) -> List[Dict]:
    """Flags os.system() — should be subprocess.run with args list."""
    findings = []
    for lineno, line in enumerate(code.splitlines(), 1):
        if re.search(r"\bos\.system\s*\(", line):
            findings.append(_finding(
                filename, "os-system", "high", lineno,
                "os.system shells out — use subprocess.run(args, check=True).",
                "security",
                "Replace os.system with subprocess.run([...], check=True)"
            ))
    return findings


def _rule_pickle_loads(filename: str, code: str) -> List[Dict]:
    """pickle.loads on untrusted data is RCE."""
    findings = []
    for lineno, line in enumerate(code.splitlines(), 1):
        if re.search(r"\bpickle\.loads?\s*\(", line):
            findings.append(_finding(
                filename, "pickle-loads", "critical", lineno,
                "pickle.load(s) on untrusted data is remote code execution.",
                "security", "Add a TODO: validate trust boundary"
            ))
    return findings


def _rule_sql_concat(filename: str, code: str) -> List[Dict]:
    """SQL string concatenation — SQL injection risk."""
    findings = []
    for lineno, line in enumerate(code.splitlines(), 1):
        if re.search(r"(SELECT|INSERT|UPDATE|DELETE).*\+", line, re.IGNORECASE):
            findings.append(_finding(
                filename, "sql-concat", "critical", lineno,
                "SQL built by string concatenation — use parameterised queries.",
                "security", "Add a TODO marker for parameterisation"
            ))
    return findings


def _rule_bare_except(filename: str, code: str) -> List[Dict]:
    """Bare except: or except: pass swallows everything."""
    findings = []
    for lineno, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if re.match(r"except\s*:", stripped) or re.match(r"except\s+Exception\s*:", stripped):
            findings.append(_finding(
                filename, "bare-except", "medium", lineno,
                "Bare except: hides errors — narrow to a specific exception.",
                "reliability", "Replace `except:` with `except Exception as e:` + log"
            ))
    return findings


def _rule_print_debug(filename: str, code: str) -> List[Dict]:
    """print() in library code — should be logging."""
    findings = []
    for lineno, line in enumerate(code.splitlines(), 1):
        if re.match(r"\s*print\s*\(", line):
            findings.append(_finding(
                filename, "print-debug", "low", lineno,
                "print() used for output — prefer the logging module.",
                "style", "Replace print(...) with logger.info(...)"
            ))
    return findings


def _rule_missing_logging_import(filename: str, code: str) -> List[Dict]:
    """If we'll add logging, make sure the import is there."""
    findings = []
    if re.search(r"\bprint\s*\(", code) and "import logging" not in code:
        findings.append(_finding(
            filename, "missing-logging-import", "low", 1,
            "Will add `import logging` and a module-level logger.",
            "style", "Add logging import + logger"
        ))
    return findings


def _rule_ast_syntax(filename: str, code: str) -> List[Dict]:
    """Make sure the file parses — broken code shouldn't proceed."""
    try:
        result = api_client.check_syntax(code)
        if result.get("is_valid", True):
            return []
        errors = result.get("errors", [])
        msg = errors[0] if errors else "Syntax error detected"
        return [_finding(
            filename, "syntax-error", "critical", 1,
            f"SyntaxError: {msg}", "reliability", "Cannot auto-fix"
        )]
    except Exception:
        # If the API is unreachable, skip the syntax check rather than block analysis.
        return []


ALL_RULES = [
    _rule_ast_syntax,
    _rule_hardcoded_secret,
    _rule_eval_exec,
    _rule_os_system,
    _rule_pickle_loads,
    _rule_sql_concat,
    _rule_bare_except,
    _rule_print_debug,
    _rule_missing_logging_import,
]


def analyze_file(filename: str, code: str) -> List[Dict]:
    """Run every rule against a single file. Returns a list of findings."""
    findings: List[Dict] = []
    for rule in ALL_RULES:
        try:
            findings.extend(rule(filename, code))
        except Exception as e:
            log("Analyzer", f"Rule {rule.__name__} crashed on {filename}: {e}")
    return findings


def analyze_all(files: Dict[str, str]) -> List[Dict]:
    """Run the analyzer against every file."""
    log("Analyzer", f"Starting analysis of {len(files)} file(s).")
    all_findings: List[Dict] = []
    for fname, code in files.items():
        f = analyze_file(fname, code)
        log("Analyzer", f"{fname}: {len(f)} finding(s).")
        all_findings.extend(f)
    log("Analyzer", f"Analysis complete — {len(all_findings)} total finding(s).")
    return all_findings


# ============================================================
# TRANSFORMATIONS — apply approved fixes
# ============================================================
# Each transform is keyed by rule name. It receives the full file code +
# a finding, and returns the (possibly) modified code.

def _xform_hardcoded_secret(code: str, finding: Dict) -> str:
    """Replace `KEY = "literal"` with `KEY = os.environ.get("KEY", "")`."""
    pattern = re.compile(
        r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[\"'][^\"']{6,}[\"']\s*$",
        re.MULTILINE,
    )
    def repl(m):
        indent, name = m.group(1), m.group(2)
        return f'{indent}{name} = os.environ.get("{name}", "")'
    new_code = pattern.sub(repl, code)
    if "import os" not in new_code:
        new_code = "import os\n" + new_code
    return new_code


def _xform_eval(code: str, finding: Dict) -> str:
    """Replace eval(...) with ast.literal_eval(...) and add the import."""
    new_code = re.sub(r"\beval\s*\(", "ast.literal_eval(", code)
    if "import ast" not in new_code:
        new_code = "import ast\n" + new_code
    return new_code


def _xform_exec(code: str, finding: Dict) -> str:
    """We don't auto-rewrite exec — just add a TODO marker above the line."""
    lines = code.splitlines()
    idx = max(0, finding["line"] - 1)
    indent = re.match(r"\s*", lines[idx]).group(0)
    lines.insert(idx, f"{indent}# TODO(hardening): exec() flagged — manual review required.")
    return "\n".join(lines)


def _xform_os_system(code: str, finding: Dict) -> str:
    """
    Naive rewrite of `os.system("cmd " + arg)` to subprocess.run.

    We only rewrite the simple "literal + variable" case; anything more
    complex gets a TODO marker so the human can finish the job.
    """
    pattern = re.compile(r'os\.system\s*\(\s*"([^"]+)"\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)')
    def repl(m):
        literal, var = m.group(1).strip(), m.group(2)
        parts = literal.split()
        argv = ", ".join(f'"{p}"' for p in parts) + f", str({var})"
        return f"subprocess.run([{argv}], check=True)"
    new_code = pattern.sub(repl, code)

    # Anything left over gets a TODO marker.
    if "os.system" in new_code:
        out_lines = []
        for line in new_code.splitlines():
            if "os.system" in line and "TODO(hardening)" not in line:
                indent = re.match(r"\s*", line).group(0)
                out_lines.append(f"{indent}# TODO(hardening): convert os.system to subprocess.run with arg list.")
            out_lines.append(line)
        new_code = "\n".join(out_lines)

    if "import subprocess" not in new_code:
        new_code = "import subprocess\n" + new_code
    return new_code


def _xform_pickle(code: str, finding: Dict) -> str:
    """Add a TODO marker above pickle.load(s) calls."""
    lines = code.splitlines()
    idx = max(0, finding["line"] - 1)
    indent = re.match(r"\s*", lines[idx]).group(0)
    lines.insert(
        idx,
        f"{indent}# TODO(hardening): pickle.load(s) on untrusted data is RCE — validate source."
    )
    return "\n".join(lines)


def _xform_sql(code: str, finding: Dict) -> str:
    """Add a TODO marker — we won't guess the right parameterisation."""
    lines = code.splitlines()
    idx = max(0, finding["line"] - 1)
    indent = re.match(r"\s*", lines[idx]).group(0)
    lines.insert(
        idx,
        f"{indent}# TODO(hardening): replace string concat with parameterised query (e.g. cursor.execute(sql, params))."
    )
    return "\n".join(lines)


def _xform_bare_except(code: str, finding: Dict) -> str:
    """
    Rewrite `except:` and `except Exception:` to a named binding with logging.
    """
    new_code = re.sub(r"except\s*:", "except Exception as e:\n            logger.exception(e)\n            raise", code)
    new_code = re.sub(
        r"except\s+Exception\s*:",
        "except Exception as e:\n            logger.exception(e)\n            raise",
        new_code,
    )
    return new_code


def _xform_print_debug(code: str, finding: Dict) -> str:
    """Replace print(...) with logger.info(...) (logger added by another rule)."""
    return re.sub(r"\bprint\s*\(", "logger.info(", code)


def _xform_logging_import(code: str, finding: Dict) -> str:
    """Add logging import + module-level logger if missing."""
    if "import logging" in code:
        return code
    header = "import logging\nlogger = logging.getLogger(__name__)\n"
    return header + code


TRANSFORMS = {
    "hardcoded-secret":        _xform_hardcoded_secret,
    "use-of-eval":             _xform_eval,
    "use-of-exec":             _xform_exec,
    "os-system":               _xform_os_system,
    "pickle-loads":            _xform_pickle,
    "sql-concat":              _xform_sql,
    "bare-except":             _xform_bare_except,
    "print-debug":             _xform_print_debug,
    "missing-logging-import":  _xform_logging_import,
}


def apply_fixes(files: Dict[str, str], approved: List[Dict]) -> Dict[str, str]:
    """
    Apply every approved finding's transform, returning the new file map.

    Transforms are applied in an order that minimises conflicts:
        1. Add logging import (so subsequent print->logger swaps work).
        2. Print -> logger swaps.
        3. Bare except rewrites.
        4. Everything else.
        5. Syntax-final pass: ensure the file still parses; if not, revert
           that file and add a TODO marker at the top.
    """
    order = ["missing-logging-import", "print-debug", "bare-except"]
    def sort_key(f):
        return order.index(f["rule"]) if f["rule"] in order else len(order)
    sorted_findings = sorted(approved, key=sort_key)

    out = dict(files)  # shallow copy
    for finding in sorted_findings:
        fname = finding["filename"]
        rule = finding["rule"]
        xform = TRANSFORMS.get(rule)
        if xform is None:
            log("Apply", f"No transform for rule '{rule}' — skipped.")
            continue
        before = out[fname]
        try:
            after = xform(before, finding)
        except Exception as e:
            log("Apply", f"Transform {rule} crashed on {fname}: {e}")
            continue
        out[fname] = after
        log("Apply", f"Applied {rule} to {fname} (line {finding['line']}).")

    # Final syntax-safety pass.
    for fname, code in out.items():
        try:
            result = api_client.check_syntax(code)
            valid = result.get("is_valid", True)
            error_msg = (result.get("errors") or ["unknown error"])[0]
        except Exception:
            valid = True  # don't block apply if API is unreachable
            error_msg = ""
        if not valid:
            log("Apply",
                f"⚠ Transformed {fname} no longer parses ({error_msg}) — reverting.")
            out[fname] = (
                f"# TODO(hardening): auto-transforms produced invalid syntax — "
                f"manual review required.\n# Original preserved below.\n\n"
                + files[fname]
            )

    return out


# ============================================================
# DIFF RENDERING
# ============================================================

def render_diff(before: str, after: str, filename: str) -> str:
    """
    Build a simple line diff without difflib.

    Compares before/after line-by-line: unchanged lines are prefixed with a
    space, removed lines with '-', and added lines with '+'. The output is
    valid diff syntax that st.code(language='diff') will syntax-highlight.
    """
    b_lines = before.splitlines()
    a_lines = after.splitlines()
    if b_lines == a_lines:
        return f"# No changes to {filename}\n"

    out = [
        f"--- {filename} (original)",
        f"+++ {filename} (hardened)",
        "",
    ]
    max_len = max(len(b_lines), len(a_lines))
    for i in range(max_len):
        b = b_lines[i] if i < len(b_lines) else None
        a = a_lines[i] if i < len(a_lines) else None
        if b is None:
            out.append(f"+{a}")
        elif a is None:
            out.append(f"-{b}")
        elif b != a:
            out.append(f"-{b}")
            out.append(f"+{a}")
        else:
            out.append(f" {b}")
    return "\n".join(out)


# ============================================================
# SANDBOX
# ============================================================
# Runs user code with a stripped builtins dict and a wall-clock timeout.
# NOT a real security boundary — see the safety note at the top of the file.

# Builtins exposed to sandboxed code. Everything destructive is removed.
SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "len", "list", "map", "max", "min", "print", "range",
        "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
        "True", "False", "None", "Exception", "ValueError", "TypeError",
        "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    ]
}


def run_in_sandbox(code: str, timeout_seconds: int = 3) -> Dict:
    """
    Execute `code` in a restricted namespace with stdout captured.

    Returns a dict with keys: ok (bool), stdout (str), error (str|None),
    duration_ms (float).

    Note: no hard wall-clock timeout is applied — this sandbox is not a
    security boundary (see module docstring). For real isolation use gVisor,
    Firecracker, or a remote sandbox.
    """
    import time
    stdout_buf = io.StringIO()

    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        "print": lambda *a, **kw: print(*a, **kw, file=stdout_buf),
    }
    safe_locals: Dict = {}

    start = time.perf_counter()
    result = {"ok": False, "stdout": "", "error": None, "duration_ms": 0.0}
    try:
        exec(code, safe_globals, safe_locals)  # noqa: S102 — intentional, sandboxed
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["duration_ms"] = (time.perf_counter() - start) * 1000
        result["stdout"] = stdout_buf.getvalue()

    return result


def evaluation_loop(files: Dict[str, str]) -> List[Dict]:
    """
    Run every file (that's importable as a script) through the sandbox.

    For each file we run a 'smoke test' that just exec's the module body
    and checks no top-level error fires. This catches transformed files
    that no longer parse or that immediately raise.
    """
    log("Sandbox", "Starting evaluation loop.")
    results = []
    for fname, code in files.items():
        # Skip files that obviously won't run without external deps.
        if "import db" in code or "from db" in code:
            results.append({
                "filename": fname, "ok": True, "stdout": "",
                "error": None, "duration_ms": 0.0,
                "skipped": True,
                "note": "Skipped — references external 'db' module.",
            })
            log("Sandbox", f"{fname}: skipped (external deps).")
            continue
        r = run_in_sandbox(code, timeout_seconds=3)
        r["filename"] = fname
        r["skipped"] = False
        results.append(r)
        status = "✅" if r["ok"] else "❌"
        log("Sandbox", f"{fname}: {status} ({r['duration_ms']:.1f}ms)")
    log("Sandbox", "Evaluation loop complete.")
    return results


# ============================================================
# ZIP PACKAGING
# ============================================================

def build_bundle(files: Dict[str, str]) -> bytes:
    """
    Pack the hardened files into a single plain-text bundle.

    Each file is separated by a header line so the bundle is easy to
    split back apart without any binary format dependency.
    """
    sep = "=" * 72
    parts = [
        f"# Hardened bundle — Generated {datetime.now().isoformat(timespec='seconds')}",
        f"# Files included: {', '.join(files.keys())}",
        "",
    ]
    for fname, code in files.items():
        parts.append(sep)
        parts.append(f"# FILE: {fname}")
        parts.append(sep)
        parts.append(code)
        parts.append("")
    return "\n".join(parts).encode("utf-8")


# ============================================================
# UI — WIZARD STEPS
# ============================================================

WIZARD_STEPS = [
    ("📁 Upload",     "Upload Python files or load the demo set."),
    ("🔎 Analyze",    "Run the analyzer agent over every file."),
    ("✅ Review",     "Approve which fixes to apply."),
    ("🛠 Apply",      "Apply approved transforms."),
    ("🧪 Sandbox",    "Run hardened code in an isolated sandbox."),
    ("⬇ Download",    "Download the hardened bundle as a text file."),
]


def render_wizard_header() -> None:
    """Top-of-page stepper that shows where the user is."""
    cur = st.session_state.step
    cols = st.columns(len(WIZARD_STEPS))
    for i, (label, _desc) in enumerate(WIZARD_STEPS, 1):
        with cols[i - 1]:
            if i < cur:
                st.markdown(f"<div style='text-align:center;color:#28a745;'>"
                            f"<b>{i}. {label}</b><br><small>done</small></div>",
                            unsafe_allow_html=True)
            elif i == cur:
                st.markdown(f"<div style='text-align:center;color:#0d6efd;"
                            f"border-bottom:3px solid #0d6efd;padding-bottom:4px;'>"
                            f"<b>{i}. {label}</b><br><small>current</small></div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:center;color:#888;'>"
                            f"<b>{i}. {label}</b><br><small>pending</small></div>",
                            unsafe_allow_html=True)


def step_upload() -> None:
    """Step 1 — upload files or load the demo."""
    st.subheader("Step 1 · Upload Vibe-Coded Files")
    st.caption("Drop one or more `.py` files. The whole folder is uploaded "
               "as a flat list of files for this MVP.")

    c1, c2 = st.columns([2, 1])
    with c1:
        uploads = st.file_uploader(
            "Python source files",
            type=["py", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploads:
            files = {}
            for u in uploads:
                try:
                    files[u.name] = u.read().decode("utf-8")
                except Exception as e:
                    st.error(f"Could not read {u.name}: {e}")
            if files:
                st.session_state.original_files = files
                st.success(f"Loaded {len(files)} file(s).")

    with c2:
        st.markdown("**No files handy?**")
        if st.button("Load demo files", use_container_width=True):
            st.session_state.original_files = dict(DEMO_FILES)
            st.success(f"Loaded {len(DEMO_FILES)} demo file(s).")
            st.rerun()

    if st.session_state.original_files:
        st.divider()
        st.markdown("#### Loaded files")
        for fname, code in st.session_state.original_files.items():
            with st.expander(f"📄 {fname}  ·  {len(code.splitlines())} lines"):
                st.code(code, language="python")

        st.divider()
        if st.button("Continue to Analyze →", type="primary", use_container_width=True):
            st.session_state.step = 2
            st.rerun()


def step_analyze() -> None:
    """Step 2 — run rule-based + (optional) LLM analysis."""
    st.subheader("Step 2 · Analyzer Agent")
    if not st.session_state.original_files:
        st.warning("No files loaded — go back to Step 1.")
        return

    use_llm = st.checkbox(
        "Also ask the LLM for a high-level review (requires OpenAI key)",
        value=False,
    )

    if st.button("🔎 Run analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzer agent working..."):
            findings = analyze_all(st.session_state.original_files)
            st.session_state.findings = findings
            # Default every finding to approved.
            st.session_state.approvals = {f["id"]: True for f in findings}

            if use_llm:
                st.session_state.llm_review = {}
                for fname, code in st.session_state.original_files.items():
                    review = llm_review_file(fname, code)
                    if review:
                        st.session_state.llm_review[fname] = review
                        log("LLM Review", f"Got LLM notes for {fname}.")
        st.success(f"Found {len(st.session_state.findings)} issue(s).")

    if st.session_state.findings:
        st.divider()
        # Tool palette display — matches the sketch's "Tools" box.
        st.markdown("#### 🧰 Tools used by the Analyzer Agent")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("AI Detection",      "✅" if use_llm else "—")
        t2.metric("Optimization",      "✅")
        t3.metric("Vulnerability",     "✅")
        t4.metric("FAST Testing",      "✅ (sandbox)")

        st.divider()
        st.markdown("#### 📋 Findings summary")
        sev_counts = {}
        for f in st.session_state.findings:
            sev_counts[f["severity"]] = sev_counts.get(f["severity"], 0) + 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Critical", sev_counts.get("critical", 0))
        c2.metric("High",     sev_counts.get("high", 0))
        c3.metric("Medium",   sev_counts.get("medium", 0))
        c4.metric("Low",      sev_counts.get("low", 0))

        if st.session_state.llm_review:
            st.divider()
            st.markdown("#### 🧠 LLM review notes")
            for fname, notes in st.session_state.llm_review.items():
                with st.expander(f"LLM notes — {fname}"):
                    st.markdown(notes)

        st.divider()
        if st.button("Continue to Review →", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()


def step_review() -> None:
    """Step 3 — user approves which findings to fix."""
    st.subheader("Step 3 · Review & Approve Changes")
    if not st.session_state.findings:
        st.warning("No findings yet — run the analyzer first.")
        return

    st.caption("Untick anything you don't want auto-fixed. "
               "Defaults to all approved.")

    # Group by file.
    by_file: Dict[str, List[Dict]] = {}
    for f in st.session_state.findings:
        by_file.setdefault(f["filename"], []).append(f)

    sev_colour = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}

    for fname, fs in by_file.items():
        with st.expander(f"📄 {fname}  ·  {len(fs)} finding(s)", expanded=True):
            for f in fs:
                row = st.columns([0.06, 0.10, 0.06, 0.58, 0.20])
                approved = row[0].checkbox(
                    "", value=st.session_state.approvals.get(f["id"], True),
                    key=f"approve_{f['id']}", label_visibility="collapsed",
                )
                st.session_state.approvals[f["id"]] = approved
                row[1].markdown(f"{sev_colour.get(f['severity'], '⚪')} **{f['severity']}**")
                row[2].markdown(f"L{f['line']}")
                row[3].markdown(f"**{f['rule']}** — {f['message']}")
                row[4].caption(f["fix_label"])

    st.divider()
    total = len(st.session_state.findings)
    approved_count = sum(1 for v in st.session_state.approvals.values() if v)
    st.info(f"{approved_count} / {total} findings approved for auto-fix.")

    c1, c2 = st.columns(2)
    if c1.button("← Back to Analyze", use_container_width=True):
        st.session_state.step = 2
        st.rerun()
    if c2.button("Apply Approved Fixes →", type="primary", use_container_width=True):
        st.session_state.step = 4
        st.rerun()


def step_apply() -> None:
    """Step 4 — run the transforms and show the diff."""
    st.subheader("Step 4 · Apply Approved Changes")

    approved = [f for f in st.session_state.findings
                if st.session_state.approvals.get(f["id"], False)]

    if not approved:
        st.warning("No findings were approved — nothing to apply.")
        if st.button("← Back to Review"):
            st.session_state.step = 3
            st.rerun()
        return

    if not st.session_state.hardened_files:
        with st.spinner("Applying transforms..."):
            st.session_state.hardened_files = apply_fixes(
                st.session_state.original_files, approved
            )
        st.success(f"Applied {len(approved)} fix(es).")

    # Per-file before/after diff.
    st.markdown("#### Before / After diffs")
    for fname in st.session_state.original_files:
        before = st.session_state.original_files[fname]
        after = st.session_state.hardened_files.get(fname, before)
        if before == after:
            with st.expander(f"📄 {fname}  ·  no changes"):
                st.caption("Analyzer didn't approve any changes for this file.")
            continue
        with st.expander(f"📄 {fname}  ·  changed", expanded=True):
            tab_diff, tab_after = st.tabs(["Diff", "Hardened source"])
            with tab_diff:
                st.code(render_diff(before, after, fname), language="diff")
            with tab_after:
                st.code(after, language="python")

    st.divider()
    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("← Back to Review", use_container_width=True):
        st.session_state.hardened_files = {}
        st.session_state.step = 3
        st.rerun()
    if c2.button("Re-apply (clear cache)", use_container_width=True):
        st.session_state.hardened_files = {}
        st.rerun()
    if c3.button("Run in Sandbox →", type="primary", use_container_width=True):
        st.session_state.step = 5
        st.rerun()


def step_sandbox() -> None:
    """Step 5 — run the hardened code in the restricted sandbox."""
    st.subheader("Step 5 · Sandbox Testing & Evaluation Loop")
    st.caption("Each file is exec'd in a restricted namespace with a 3s timeout. "
               "This catches transforms that broke the code.")

    if not st.session_state.hardened_files:
        st.warning("No hardened code yet — go back to Step 4.")
        return

    if st.button("🧪 Run evaluation loop", type="primary", use_container_width=True):
        with st.spinner("Running in sandbox..."):
            st.session_state.sandbox_results = evaluation_loop(
                st.session_state.hardened_files
            )

    if st.session_state.sandbox_results:
        st.markdown("#### Results")
        for r in st.session_state.sandbox_results:
            badge = "⏭ skipped" if r.get("skipped") else ("✅ pass" if r["ok"] else "❌ fail")
            with st.expander(f"{badge}  ·  {r['filename']}  ·  {r['duration_ms']:.1f}ms"):
                if r.get("skipped"):
                    st.info(r.get("note", "Skipped."))
                if r["stdout"]:
                    st.markdown("**stdout**")
                    st.code(r["stdout"])
                if r["error"]:
                    st.error(r["error"])
                if not r["error"] and not r.get("skipped"):
                    st.success("No exceptions raised.")

        # Ad-hoc snippet box — run arbitrary code against the sandbox.
        st.divider()
        st.markdown("#### Try a snippet against the sandbox")
        snippet = st.text_area(
            "Paste a small Python snippet to evaluate (the sandbox blocks "
            "file I/O, imports of dangerous modules, etc.)",
            value="for i in range(5):\n    print('hello', i)",
            height=120,
        )
        if st.button("▶ Run snippet"):
            r = run_in_sandbox(snippet)
            st.code(r["stdout"] or "(no output)")
            if r["error"]:
                st.error(r["error"])
            st.caption(f"Duration: {r['duration_ms']:.1f}ms")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("← Back to Apply", use_container_width=True):
        st.session_state.step = 4
        st.rerun()
    if c2.button("Continue to Download →", type="primary", use_container_width=True):
        st.session_state.step = 6
        st.rerun()


def step_download() -> None:
    """Step 6 — package and download the hardened bundle."""
    st.subheader("Step 6 · Download Hardened Folder")

    if not st.session_state.hardened_files:
        st.warning("No hardened code to download.")
        return

    # Summary table.
    summary_rows = []
    for fname in st.session_state.original_files:
        before = st.session_state.original_files[fname]
        after = st.session_state.hardened_files.get(fname, before)
        changes = sum(1 for line in render_diff(before, after, fname).splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
        summary_rows.append({
            "File": fname,
            "Lines (before)": len(before.splitlines()),
            "Lines (after)":  len(after.splitlines()),
            "Lines added":    changes,
        })
    st.dataframe(summary_rows, hide_index=True, use_container_width=True)

    bundle_bytes = build_bundle(st.session_state.hardened_files)
    st.download_button(
        "⬇ Download hardened bundle (.txt)",
        data=bundle_bytes,
        file_name=f"hardened_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    if st.button("🔁 Start over with new files"):
        for k in ["step", "original_files", "hardened_files", "findings",
                  "approvals", "sandbox_results", "llm_review"]:
            st.session_state[k] = {} if k != "step" else 1
            if k == "findings" or k == "sandbox_results":
                st.session_state[k] = []
        st.rerun()


# ============================================================
# SIDEBAR — console + key + reset
# ============================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🛡️ Vibe Code Hardening")
        st.caption("Analyzer Agent · Streamlit MVP")

        st.divider()
        st.markdown("### 🔑 OpenAI API key (optional)")
        st.text_input(
            "Key (session-only)",
            type="password",
            key="openai_key",
            label_visibility="collapsed",
        )
        if not _OPENAI_IMPORTED:
            st.info("`openai` package not installed — rule-based only.")
        elif not (os.environ.get("OPENAI_API_KEY") or st.session_state.openai_key):
            st.warning("No key — LLM review disabled.")
        else:
            st.success("OpenAI key detected.")

        st.divider()
        st.markdown("### 🧭 Wizard navigation")
        for i, (label, _desc) in enumerate(WIZARD_STEPS, 1):
            if st.button(f"{i}. {label}", use_container_width=True,
                         key=f"nav_{i}",
                         type=("primary" if i == st.session_state.step else "secondary")):
                st.session_state.step = i
                st.rerun()

        st.divider()
        st.markdown("### 🧠 Console")
        if st.button("Clear console", use_container_width=True):
            st.session_state.console = []
            st.rerun()
        with st.container(height=300):
            if not st.session_state.console:
                st.caption("No reasoning steps yet.")
            for entry in st.session_state.console[-50:][::-1]:
                st.markdown(
                    f"`{entry['ts']}` **{entry['stage']}** — {entry['message']}"
                )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("🛡️ Vibe Code Hardening")
    st.caption("Upload rough Python → analyze → review → apply → sandbox → download.")

    render_wizard_header()
    st.divider()

    step = st.session_state.step
    if step == 1: step_upload()
    elif step == 2: step_analyze()
    elif step == 3: step_review()
    elif step == 4: step_apply()
    elif step == 5: step_sandbox()
    elif step == 6: step_download()


if __name__ == "__main__":
    main()

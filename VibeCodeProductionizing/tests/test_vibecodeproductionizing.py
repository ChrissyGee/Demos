"""
Unit tests for VibeCodeProductionizing/app.py.

The rule-based analyzer and all transforms are fully deterministic.
Syntax checking uses Python's built-in ast.parse — no network calls.

Streamlit is mocked before import. st.session_state.console is a real list so
log() calls succeed without AttributeError.
"""

import os
import sys
import importlib.util
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock streamlit before importing app.
# Use importlib so the module is loaded as 'vibe_app' and does not collide
# with other 'app' modules when the full test suite runs together.
# ---------------------------------------------------------------------------
_st = MagicMock()
_st.session_state = MagicMock()
_st.session_state.console = []
_st.session_state.get = MagicMock(return_value="")
sys.modules["streamlit"] = _st

_app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
_spec = importlib.util.spec_from_file_location("vibe_app", _app_path)
app = importlib.util.module_from_spec(_spec)
sys.modules["vibe_app"] = app
_spec.loader.exec_module(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_console():
    _st.session_state.console = []


DEMO_CODE = app.DEMO_FILES["secrets.py"]
CLEAN_CODE = app.DEMO_FILES["math_helpers.py"]


# ---------------------------------------------------------------------------
# Individual rule functions
# ---------------------------------------------------------------------------

class TestRuleHardcodedSecret:

    def test_detects_api_key(self):
        code = 'API_KEY = "sk-live-abc123xyz"'
        findings = app._rule_hardcoded_secret("f.py", code)
        assert len(findings) == 1
        assert findings[0]["rule"] == "hardcoded-secret"

    def test_detects_password(self):
        code = 'DB_PASSWORD = "admin123"'
        findings = app._rule_hardcoded_secret("f.py", code)
        assert len(findings) == 1

    def test_clean_code_no_findings(self):
        assert app._rule_hardcoded_secret("f.py", CLEAN_CODE) == []

    def test_severity_is_critical(self):
        code = 'token = "secret-val-12345"'
        assert app._rule_hardcoded_secret("f.py", code)[0]["severity"] == "critical"

    def test_finding_has_all_required_keys(self):
        code = 'API_KEY = "sk-live-1234567890"'
        f = app._rule_hardcoded_secret("f.py", code)[0]
        assert {"id", "filename", "rule", "severity", "line",
                "message", "category", "fix_label"}.issubset(set(f.keys()))

    def test_category_is_security(self):
        code = 'secret = "supersecret123"'
        assert app._rule_hardcoded_secret("f.py", code)[0]["category"] == "security"


class TestRuleEvalExec:

    def test_detects_eval(self):
        code = 'result = eval(user_input)'
        findings = app._rule_eval_exec("f.py", code)
        assert any(f["rule"] == "use-of-eval" for f in findings)

    def test_detects_exec(self):
        code = 'exec(user_code)'
        findings = app._rule_eval_exec("f.py", code)
        assert any(f["rule"] == "use-of-exec" for f in findings)

    def test_no_false_positive_on_evaluate(self):
        code = '# evaluation function'
        findings = app._rule_eval_exec("f.py", code)
        assert findings == []

    def test_clean_code_no_findings(self):
        assert app._rule_eval_exec("f.py", CLEAN_CODE) == []


class TestRuleOsSystem:

    def test_detects_os_system(self):
        code = 'os.system("echo hello " + name)'
        findings = app._rule_os_system("f.py", code)
        assert len(findings) == 1
        assert findings[0]["rule"] == "os-system"

    def test_severity_is_high(self):
        code = 'os.system("cmd")'
        assert app._rule_os_system("f.py", code)[0]["severity"] == "high"

    def test_no_false_positive_on_subprocess(self):
        code = 'subprocess.run(["ls"], check=True)'
        assert app._rule_os_system("f.py", code) == []


class TestRulePickleLoads:

    def test_detects_pickle_loads(self):
        code = 'return pickle.loads(blob)'
        findings = app._rule_pickle_loads("f.py", code)
        assert len(findings) == 1
        assert findings[0]["rule"] == "pickle-loads"

    def test_severity_is_critical(self):
        code = 'data = pickle.loads(raw)'
        assert app._rule_pickle_loads("f.py", code)[0]["severity"] == "critical"

    def test_clean_code_no_findings(self):
        assert app._rule_pickle_loads("f.py", CLEAN_CODE) == []


class TestRuleSqlConcat:

    def test_detects_sql_concat(self):
        code = 'db.query("SELECT * FROM users WHERE id = " + str(uid))'
        findings = app._rule_sql_concat("f.py", code)
        assert len(findings) == 1
        assert findings[0]["rule"] == "sql-concat"

    def test_severity_is_critical(self):
        code = 'cursor.execute("DELETE FROM logs WHERE id=" + rid)'
        assert app._rule_sql_concat("f.py", code)[0]["severity"] == "critical"

    def test_clean_code_no_findings(self):
        assert app._rule_sql_concat("f.py", CLEAN_CODE) == []


class TestRuleBareExcept:

    def test_detects_bare_except(self):
        code = "try:\n    pass\nexcept:\n    pass"
        findings = app._rule_bare_except("f.py", code)
        assert len(findings) == 1
        assert findings[0]["rule"] == "bare-except"

    def test_detects_except_exception(self):
        code = "try:\n    pass\nexcept Exception:\n    pass"
        findings = app._rule_bare_except("f.py", code)
        assert len(findings) == 1

    def test_named_exception_no_finding(self):
        code = "try:\n    pass\nexcept ValueError as e:\n    pass"
        assert app._rule_bare_except("f.py", code) == []


class TestRulePrintDebug:

    def test_detects_print_call(self):
        code = '    print("dividing", a, b)'
        findings = app._rule_print_debug("f.py", code)
        assert len(findings) == 1
        assert findings[0]["rule"] == "print-debug"

    def test_severity_is_low(self):
        code = 'print("hello")'
        assert app._rule_print_debug("f.py", code)[0]["severity"] == "low"

    def test_clean_code_no_print_finding(self):
        assert app._rule_print_debug("f.py", CLEAN_CODE) == []


class TestRuleMissingLoggingImport:

    def test_adds_finding_when_print_without_logging(self):
        code = 'print("x")'
        findings = app._rule_missing_logging_import("f.py", code)
        assert len(findings) == 1
        assert findings[0]["rule"] == "missing-logging-import"

    def test_no_finding_when_logging_already_imported(self):
        code = 'import logging\nprint("x")'
        assert app._rule_missing_logging_import("f.py", code) == []

    def test_no_finding_when_no_print_in_code(self):
        assert app._rule_missing_logging_import("f.py", CLEAN_CODE) == []


class TestRuleAstSyntax:

    def test_valid_code_returns_no_findings(self):
        assert app._rule_ast_syntax("f.py", CLEAN_CODE) == []

    def test_invalid_code_returns_syntax_error_finding(self):
        findings = app._rule_ast_syntax("f.py", "def (")
        assert len(findings) == 1
        assert findings[0]["rule"] == "syntax-error"
        assert findings[0]["severity"] == "critical"

    def test_valid_simple_expression_no_findings(self):
        assert app._rule_ast_syntax("f.py", "x = 1 + 2") == []


# ---------------------------------------------------------------------------
# Finding helper
# ---------------------------------------------------------------------------

class TestFindingHelper:

    def test_stable_id_format(self):
        f = app._finding("myfile.py", "some-rule", "high", 42, "msg", "cat", "fix")
        assert f["id"] == "myfile.py::some-rule::42"

    def test_all_keys_present(self):
        f = app._finding("f.py", "rule", "low", 1, "msg", "cat", "fix")
        assert {"id", "filename", "rule", "severity", "line",
                "message", "category", "fix_label"}.issubset(set(f.keys()))


# ---------------------------------------------------------------------------
# analyze_file / analyze_all
# ---------------------------------------------------------------------------

class TestAnalyzeFile:

    def setup_method(self):
        _reset_console()

    def test_demo_secrets_py_has_findings(self):
        findings = app.analyze_file("secrets.py", app.DEMO_FILES["secrets.py"])
        assert len(findings) > 0

    def test_clean_math_helpers_has_few_findings(self):
        findings = app.analyze_file("math_helpers.py", CLEAN_CODE)
        assert len(findings) == 0

    def test_returns_list(self):
        assert isinstance(app.analyze_file("f.py", CLEAN_CODE), list)


class TestAnalyzeAll:

    def setup_method(self):
        _reset_console()

    def test_processes_all_files(self):
        findings = app.analyze_all(app.DEMO_FILES)
        filenames = {f["filename"] for f in findings}
        assert "secrets.py" in filenames
        assert "utils.py" in filenames

    def test_returns_list(self):
        assert isinstance(app.analyze_all(app.DEMO_FILES), list)

    def test_finding_count_greater_than_zero(self):
        assert len(app.analyze_all(app.DEMO_FILES)) > 0


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestXformHarcodedSecret:

    def test_replaces_literal_with_environ_get(self):
        code = 'API_KEY = "sk-live-abc123"'
        result = app._xform_hardcoded_secret(code, {})
        assert 'os.environ.get("API_KEY"' in result

    def test_adds_import_os_if_missing(self):
        code = 'PASSWORD = "hunter2xyz"'
        result = app._xform_hardcoded_secret(code, {})
        assert "import os" in result

    def test_does_not_duplicate_import_os(self):
        code = 'import os\nAPI_KEY = "sk-live-abc123"'
        result = app._xform_hardcoded_secret(code, {})
        assert result.count("import os") == 1


class TestXformEval:

    def test_replaces_eval_with_literal_eval(self):
        code = 'x = eval(user_input)'
        result = app._xform_eval(code, {})
        assert "ast.literal_eval(" in result
        assert "eval(" not in result.replace("literal_eval(", "")

    def test_adds_ast_import(self):
        code = 'x = eval(data)'
        result = app._xform_eval(code, {})
        assert "import ast" in result


class TestXformOsSystem:

    def test_simple_concat_converted_to_subprocess(self):
        code = 'os.system("echo hello " + name)'
        finding = {"line": 1}
        result = app._xform_os_system(code, finding)
        assert "subprocess.run" in result

    def test_adds_subprocess_import(self):
        code = 'os.system("echo hello " + name)'
        result = app._xform_os_system(code, {"line": 1})
        assert "import subprocess" in result

    def test_unrecognised_form_gets_todo_marker(self):
        code = 'os.system(cmd)'
        result = app._xform_os_system(code, {"line": 1})
        assert "TODO(hardening)" in result


class TestXformBareExcept:

    def test_bare_except_replaced(self):
        code = "try:\n    x()\nexcept:\n    pass"
        result = app._xform_bare_except(code, {})
        assert "except Exception as e:" in result

    def test_except_exception_replaced(self):
        code = "try:\n    x()\nexcept Exception:\n    pass"
        result = app._xform_bare_except(code, {})
        assert "except Exception as e:" in result


class TestXformPrintDebug:

    def test_print_replaced_with_logger_info(self):
        code = 'print("hello", x)'
        result = app._xform_print_debug(code, {})
        assert "logger.info(" in result
        assert "print(" not in result


class TestXformLoggingImport:

    def test_adds_import_logging(self):
        code = 'print("x")'
        result = app._xform_logging_import(code, {})
        assert "import logging" in result
        assert "logger = logging.getLogger(__name__)" in result

    def test_idempotent_if_already_imported(self):
        code = 'import logging\nlogger = logging.getLogger(__name__)\n'
        result = app._xform_logging_import(code, {})
        assert result.count("import logging") == 1


# ---------------------------------------------------------------------------
# apply_fixes
# ---------------------------------------------------------------------------

class TestApplyFixes:

    def setup_method(self):
        _reset_console()

    def _get_findings(self, filename, code):
        return [f for f in app.analyze_file(filename, code)
                if f["rule"] != "syntax-error"]

    def test_returns_dict_of_same_filenames(self):
        files = {"f.py": CLEAN_CODE}
        result = app.apply_fixes(files, [])
        assert set(result.keys()) == {"f.py"}

    def test_apply_hardcoded_secret_fix(self):
        code = 'API_KEY = "sk-live-abc123xyz"\n'
        findings = app._rule_hardcoded_secret("f.py", code)
        result = app.apply_fixes({"f.py": code}, findings)
        assert 'os.environ.get("API_KEY"' in result["f.py"]

    def test_no_approved_findings_returns_original(self):
        files = {"f.py": DEMO_CODE}
        result = app.apply_fixes(files, [])
        assert result["f.py"] == DEMO_CODE

    def test_syntax_invalid_output_gets_reverted(self):
        err = SyntaxError("bad syntax")
        err.lineno = 1
        err.msg = "bad syntax"
        with patch("ast.parse", side_effect=err):
            code = 'print("x")'
            finding = app._rule_print_debug("f.py", code)[0]
            result = app.apply_fixes({"f.py": code}, [finding])
        assert "TODO(hardening)" in result["f.py"]


# ---------------------------------------------------------------------------
# render_diff
# ---------------------------------------------------------------------------

class TestRenderDiff:

    def test_unchanged_returns_no_changes_message(self):
        result = app.render_diff("a\nb\n", "a\nb\n", "f.py")
        assert "No changes" in result

    def test_added_line_prefixed_with_plus(self):
        result = app.render_diff("a", "a\nb", "f.py")
        assert "+b" in result

    def test_removed_line_prefixed_with_minus(self):
        result = app.render_diff("a\nb", "a", "f.py")
        assert "-b" in result

    def test_header_contains_filename(self):
        result = app.render_diff("x", "y", "myfile.py")
        assert "myfile.py" in result

    def test_diff_header_lines_present(self):
        result = app.render_diff("x", "y", "f.py")
        assert "---" in result and "+++" in result

    def test_changed_line_shows_both_minus_and_plus(self):
        result = app.render_diff("old_line", "new_line", "f.py")
        assert "-old_line" in result
        assert "+new_line" in result


# ---------------------------------------------------------------------------
# run_in_sandbox
# ---------------------------------------------------------------------------

class TestRunInSandbox:

    def test_simple_expression_succeeds(self):
        result = app.run_in_sandbox("x = 1 + 1")
        assert result["ok"] is True
        assert result["error"] is None

    def test_print_output_captured(self):
        result = app.run_in_sandbox('print("hello sandbox")')
        assert "hello sandbox" in result["stdout"]

    def test_error_code_sets_ok_false(self):
        result = app.run_in_sandbox("raise ValueError('boom')")
        assert result["ok"] is False
        assert "ValueError" in result["error"]

    def test_duration_ms_is_nonnegative(self):
        result = app.run_in_sandbox("pass")
        assert result["duration_ms"] >= 0.0

    def test_result_has_required_keys(self):
        result = app.run_in_sandbox("pass")
        assert {"ok", "stdout", "error", "duration_ms"}.issubset(set(result.keys()))

    def test_blocked_import_raises_in_sandbox(self):
        result = app.run_in_sandbox("import os")
        assert result["ok"] is False

    def test_division_by_zero_captured_as_error(self):
        result = app.run_in_sandbox("x = 1 / 0")
        assert result["ok"] is False
        assert "ZeroDivision" in result["error"]


# ---------------------------------------------------------------------------
# build_bundle
# ---------------------------------------------------------------------------

class TestBuildBundle:

    def test_returns_bytes(self):
        result = app.build_bundle({"f.py": "x = 1"})
        assert isinstance(result, bytes)

    def test_contains_filename_marker(self):
        result = app.build_bundle({"myfile.py": "x = 1"}).decode("utf-8")
        assert "FILE: myfile.py" in result

    def test_contains_file_content(self):
        result = app.build_bundle({"f.py": "def foo(): pass"}).decode("utf-8")
        assert "def foo(): pass" in result

    def test_multiple_files_all_present(self):
        files = {"a.py": "x = 1", "b.py": "y = 2"}
        result = app.build_bundle(files).decode("utf-8")
        assert "FILE: a.py" in result
        assert "FILE: b.py" in result

    def test_bundle_is_utf8_decodable(self):
        try:
            app.build_bundle(app.DEMO_FILES).decode("utf-8")
        except UnicodeDecodeError:
            pytest.fail("build_bundle output is not valid UTF-8")


# ---------------------------------------------------------------------------
# evaluation_loop
# ---------------------------------------------------------------------------

class TestEvaluationLoop:

    def setup_method(self):
        _reset_console()

    def test_returns_list(self):
        result = app.evaluation_loop({"math_helpers.py": CLEAN_CODE})
        assert isinstance(result, list)

    def test_result_has_filename_key(self):
        result = app.evaluation_loop({"f.py": "x = 1"})
        assert result[0]["filename"] == "f.py"

    def test_db_reference_causes_skip(self):
        code = "import db\ndb.connect()"
        result = app.evaluation_loop({"db_code.py": code})
        assert result[0]["skipped"] is True

    def test_clean_math_code_passes(self):
        result = app.evaluation_loop({"math_helpers.py": CLEAN_CODE})
        assert result[0]["ok"] is True

    def test_result_has_ok_key(self):
        result = app.evaluation_loop({"f.py": "pass"})
        assert "ok" in result[0]

"""
Tesla Stock Price Prediction — Test Runner & Formal Report Generator
====================================================================
Runs pytest and unittest suites, collects pipeline metrics, and writes
a formal HTML + text report to reports/.
"""

import datetime
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATE_DISPLAY = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")


# ── Helpers ───────────────────────────────────────────────────────────
def run_cmd(cmd: list[str], cwd: str | Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd), timeout=600
    )


# ── 1. Run pytest ─────────────────────────────────────────────────────
def run_pytest() -> dict:
    json_path = REPORTS_DIR / f"pytest_results_{TIMESTAMP}.json"
    result = run_cmd(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/pytest/",
            f"--json-report-file={json_path}",
            "--json-report",
            "-v",
            "--tb=short",
            "-q",
        ]
    )

    # Fallback if json-report plugin not installed
    if not json_path.exists():
        result = run_cmd(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/pytest/",
                f"--junitxml={REPORTS_DIR / 'pytest_junit.xml'}",
                "-v",
                "--tb=short",
            ]
        )
        return _parse_pytest_stdout(result)

    with open(json_path) as f:
        data = json.load(f)

    tests = []
    for t in data.get("tests", []):
        tests.append(
            {
                "name": t.get("nodeid", ""),
                "outcome": t.get("outcome", "unknown"),
                "duration": t.get("duration", 0),
                "message": (t.get("call", {}) or {}).get("longrepr", ""),
            }
        )

    summary = data.get("summary", {})
    return {
        "tests": tests,
        "passed": summary.get("passed", 0),
        "failed": summary.get("failed", 0),
        "errors": summary.get("error", 0),
        "skipped": summary.get("skipped", 0),
        "total": summary.get("total", 0),
        "duration": data.get("duration", 0),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _parse_pytest_stdout(result: subprocess.CompletedProcess) -> dict:
    """Parse pytest output when json-report is not available."""
    stdout = result.stdout
    passed = stdout.count(" PASSED")
    failed = stdout.count(" FAILED")
    errors = stdout.count(" ERROR")
    skipped = stdout.count(" SKIPPED")

    tests = []
    for line in stdout.splitlines():
        if " PASSED" in line or " FAILED" in line or " ERROR" in line:
            parts = line.strip().split()
            if len(parts) >= 2:
                outcome = parts[-1].strip()
                name = parts[0]
                tests.append(
                    {
                        "name": name,
                        "outcome": outcome.lower(),
                        "duration": 0,
                        "message": "",
                    }
                )

    return {
        "tests": tests,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "total": passed + failed + errors + skipped,
        "duration": 0,
        "stdout": stdout,
        "stderr": result.stderr,
    }


# ── 2. Run unittest ──────────────────────────────────────────────────
def run_unittest() -> dict:
    result = run_cmd(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests/unittest",
            "-p",
            "test_*.py",
            "-v",
        ]
    )
    output = result.stderr + result.stdout  # unittest writes to stderr

    passed = output.count("... ok")
    failed = output.count("... FAIL")
    errors = output.count("... ERROR")

    # Extract summary line like "Ran 20 tests in 5.432s"
    total = 0
    duration = 0.0
    for line in output.splitlines():
        if line.startswith("Ran "):
            parts = line.split()
            total = int(parts[1])
            duration = float(parts[4].rstrip("s"))

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": total,
        "duration": duration,
        "stdout": output,
        "ok": result.returncode == 0,
    }


# ── 3. Run verify.py ─────────────────────────────────────────────────
def run_verify() -> dict:
    result = run_cmd([sys.executable, "verify.py"])
    output = result.stdout

    pass_c = output.count("[PASS]")
    warn_c = output.count("[WARN]")
    fail_c = output.count("[FAIL]")

    return {
        "passed": pass_c,
        "warnings": warn_c,
        "failed": fail_c,
        "stdout": output,
        "ok": result.returncode == 0,
    }


# ── 4. Collect pipeline metrics from cached data ─────────────────────
def collect_pipeline_metrics() -> dict:
    metrics = {}
    try:
        import logging

        logging.basicConfig(level=logging.WARNING)

        sys.path.insert(0, str(ROOT))
        from scr.data.data_download import load_raw_datasets, merge_raw
        from scr.data.make_dataset import (
            _engineer_features,
            prepare_features,
            split_data,
            scale_features,
            FEATURE_COLS,
            TARGET_COL,
        )
        from scr.Model.train_models import get_models, train_all_models
        from scr.Model.predict_models import (
            build_results_table,
            get_best_model,
            directional_accuracy,
        )

        datasets = load_raw_datasets()
        merged = merge_raw(datasets)
        df = _engineer_features(merged)
        model_df = prepare_features(df)
        X_train, X_test, y_train, y_test, train_df, test_df = split_data(model_df)
        X_tr_sc, X_te_sc, scaler = scale_features(X_train, X_test)

        models = get_models()
        results = train_all_models(
            models, X_train, y_train, X_tr_sc, X_test, X_te_sc, y_test
        )
        table = build_results_table(results)
        best_name, best_model, best_preds = get_best_model(results, models)
        dir_acc = directional_accuracy(y_test, best_preds)

        metrics["dataset"] = {
            "total_rows": len(df),
            "clean_rows": len(model_df),
            "features": len(FEATURE_COLS),
            "target": TARGET_COL,
            "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        }

        model_results = {}
        for name in table.index:
            model_results[name] = {
                "MAE": round(table.loc[name, "MAE"], 2),
                "RMSE": round(table.loc[name, "RMSE"], 2),
                "R2": round(table.loc[name, "R²"], 4),
                "MAPE": round(table.loc[name, "MAPE (%)"], 2),
            }
        metrics["models"] = model_results
        metrics["best_model"] = {
            "name": best_name,
            "R2": round(results[best_name]["R²"], 4),
            "MAPE": round(results[best_name]["MAPE (%)"], 2),
            "directional_accuracy": round(dir_acc, 1),
        }

    except Exception as exc:
        metrics["error"] = str(exc)

    return metrics


# ── 5. Generate reports ───────────────────────────────────────────────
def generate_text_report(
    pytest_res: dict, unittest_res: dict, verify_res: dict, metrics: dict
) -> str:
    lines = []
    w = 78
    lines.append("=" * w)
    lines.append("TESLA STOCK PRICE PREDICTION — TEST & VALIDATION REPORT")
    lines.append(f"Generated: {DATE_DISPLAY}")
    lines.append("=" * w)

    # Project overview
    lines.append("")
    lines.append("1. PROJECT OVERVIEW")
    lines.append("-" * w)
    if "dataset" in metrics:
        d = metrics["dataset"]
        lines.append(f"   Date Range      : {d['date_range']}")
        lines.append(f"   Total Rows      : {d['total_rows']:,}")
        lines.append(f"   Clean Rows      : {d['clean_rows']:,}")
        lines.append(f"   Features        : {d['features']}")
        lines.append(f"   Target          : {d['target']}")
        lines.append(f"   Training Set    : {d['train_rows']:,} rows")
        lines.append(f"   Test Set        : {d['test_rows']:,} rows")
    else:
        lines.append(f"   Error: {metrics.get('error', 'N/A')}")

    # Model performance
    lines.append("")
    lines.append("2. MODEL PERFORMANCE (TEST SET)")
    lines.append("-" * w)
    if "models" in metrics:
        lines.append(
            f"   {'Model':28s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'MAPE':>8s}"
        )
        lines.append(f"   {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for name, m in metrics["models"].items():
            lines.append(
                f"   {name:28s} {m['MAE']:8.2f} {m['RMSE']:8.2f} {m['R2']:8.4f} {m['MAPE']:7.2f}%"
            )
        b = metrics["best_model"]
        lines.append("")
        lines.append(f"   Best Model          : {b['name']}")
        lines.append(f"   Best R²             : {b['R2']}")
        lines.append(f"   Best MAPE           : {b['MAPE']}%")
        lines.append(f"   Directional Accuracy: {b['directional_accuracy']}%")

    # Pytest results
    lines.append("")
    lines.append("3. PYTEST RESULTS")
    lines.append("-" * w)
    lines.append(
        f"   Passed: {pytest_res['passed']}  |  Failed: {pytest_res['failed']}  |  "
        f"Errors: {pytest_res['errors']}  |  Skipped: {pytest_res['skipped']}  |  "
        f"Total: {pytest_res['total']}"
    )
    if pytest_res["failed"] > 0 or pytest_res["errors"] > 0:
        lines.append("")
        lines.append("   FAILED TESTS:")
        for t in pytest_res.get("tests", []):
            if t["outcome"] in ("failed", "error"):
                lines.append(f"     ✗ {t['name']}")
                if t["message"]:
                    for mline in str(t["message"]).splitlines()[:3]:
                        lines.append(f"       {mline}")

    # Unittest results
    lines.append("")
    lines.append("4. UNITTEST RESULTS")
    lines.append("-" * w)
    lines.append(
        f"   Passed: {unittest_res['passed']}  |  Failed: {unittest_res['failed']}  |  "
        f"Errors: {unittest_res['errors']}  |  Total: {unittest_res['total']}  |  "
        f"Duration: {unittest_res['duration']:.1f}s"
    )
    lines.append(f"   Status: {'PASS' if unittest_res['ok'] else 'FAIL'}")

    # Verify results
    lines.append("")
    lines.append("5. PROJECT VERIFICATION (verify.py)")
    lines.append("-" * w)
    lines.append(
        f"   Passed: {verify_res['passed']}  |  Warnings: {verify_res['warnings']}  |  "
        f"Failed: {verify_res['failed']}"
    )
    lines.append(f"   Status: {'PASS' if verify_res['ok'] else 'FAIL'}")

    # Overall verdict
    lines.append("")
    lines.append("6. OVERALL VERDICT")
    lines.append("=" * w)
    all_pass = (
        pytest_res["failed"] == 0
        and pytest_res["errors"] == 0
        and unittest_res["ok"]
        and verify_res["ok"]
    )
    total_tests = pytest_res["total"] + unittest_res["total"] + verify_res["passed"]
    total_pass = pytest_res["passed"] + unittest_res["passed"] + verify_res["passed"]
    if all_pass:
        lines.append(f"   ✓ ALL CHECKS PASSED  ({total_pass}/{total_tests} checks)")
    else:
        lines.append(
            f"   ✗ SOME CHECKS FAILED  ({total_pass}/{total_tests} checks passed)"
        )

    lines.append("")
    lines.append("=" * w)
    lines.append(f"Report saved to: reports/")
    lines.append("=" * w)

    return "\n".join(lines)


def generate_html_report(
    pytest_res: dict, unittest_res: dict, verify_res: dict, metrics: dict
) -> str:
    all_pass = (
        pytest_res["failed"] == 0
        and pytest_res["errors"] == 0
        and unittest_res["ok"]
        and verify_res["ok"]
    )
    badge_color = "#27ae60" if all_pass else "#e74c3c"
    badge_text = "ALL PASSED" if all_pass else "FAILURES DETECTED"

    model_rows = ""
    if "models" in metrics:
        for name, m in metrics["models"].items():
            best_cls = ' class="best"' if name == metrics["best_model"]["name"] else ""
            model_rows += f"""
            <tr{best_cls}>
                <td>{name}</td>
                <td>{m['MAE']:.2f}</td>
                <td>{m['RMSE']:.2f}</td>
                <td>{m['R2']:.4f}</td>
                <td>{m['MAPE']:.2f}%</td>
            </tr>"""

    pytest_rows = ""
    for t in pytest_res.get("tests", []):
        cls = "pass" if t["outcome"] == "passed" else "fail"
        icon = "✓" if t["outcome"] == "passed" else "✗"
        pytest_rows += f'<tr class="{cls}"><td>{icon}</td><td>{t["name"]}</td><td>{t["outcome"]}</td></tr>\n'

    ds = metrics.get("dataset", {})
    bm = metrics.get("best_model", {})

    html = textwrap.dedent(
        f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Tesla Stock Prediction — Test Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem; background: #f5f7fa; color: #2c3e50; }}
            h1 {{ color: #1a252f; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem; }}
            h2 {{ color: #2c3e50; margin-top: 2rem; }}
            .badge {{ display: inline-block; padding: 8px 24px; border-radius: 6px;
                      color: white; font-weight: bold; font-size: 1.1rem;
                      background: {badge_color}; }}
            table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
            th {{ background: #34495e; color: white; }}
            tr:nth-child(even) {{ background: #f9f9f9; }}
            tr.best {{ background: #d4edda; font-weight: bold; }}
            tr.pass td:first-child {{ color: #27ae60; }}
            tr.fail td:first-child {{ color: #e74c3c; font-weight: bold; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                             gap: 1rem; margin: 1rem 0; }}
            .card {{ background: white; border-radius: 8px; padding: 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                     text-align: center; }}
            .card .value {{ font-size: 1.8rem; font-weight: bold; color: #3498db; }}
            .card .label {{ font-size: 0.85rem; color: #7f8c8d; margin-top: 0.3rem; }}
            .section {{ background: white; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9rem; }}
        </style>
    </head>
    <body>
        <h1>Tesla Stock Price Prediction — Test & Validation Report</h1>
        <p class="timestamp">Generated: {DATE_DISPLAY}</p>
        <p><span class="badge">{badge_text}</span></p>

        <div class="section">
            <h2>1. Dataset Overview</h2>
            <div class="summary-grid">
                <div class="card"><div class="value">{ds.get('total_rows', 'N/A'):,}</div><div class="label">Total Rows</div></div>
                <div class="card"><div class="value">{ds.get('clean_rows', 'N/A'):,}</div><div class="label">Clean Rows</div></div>
                <div class="card"><div class="value">{ds.get('features', 'N/A')}</div><div class="label">Features</div></div>
                <div class="card"><div class="value">{ds.get('train_rows', 'N/A'):,}</div><div class="label">Train Rows</div></div>
                <div class="card"><div class="value">{ds.get('test_rows', 'N/A'):,}</div><div class="label">Test Rows</div></div>
            </div>
            <p><strong>Date Range:</strong> {ds.get('date_range', 'N/A')}</p>
            <p><strong>Target Variable:</strong> {ds.get('target', 'N/A')}</p>
        </div>

        <div class="section">
            <h2>2. Model Performance (Test Set)</h2>
            <table>
                <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th><th>MAPE</th></tr>
                {model_rows}
            </table>
            <div class="summary-grid">
                <div class="card"><div class="value">{bm.get('name', 'N/A')}</div><div class="label">Best Model</div></div>
                <div class="card"><div class="value">{bm.get('R2', 'N/A')}</div><div class="label">R²</div></div>
                <div class="card"><div class="value">{bm.get('MAPE', 'N/A')}%</div><div class="label">MAPE</div></div>
                <div class="card"><div class="value">{bm.get('directional_accuracy', 'N/A')}%</div><div class="label">Directional Accuracy</div></div>
            </div>
        </div>

        <div class="section">
            <h2>3. Pytest Results</h2>
            <div class="summary-grid">
                <div class="card"><div class="value" style="color:#27ae60">{pytest_res['passed']}</div><div class="label">Passed</div></div>
                <div class="card"><div class="value" style="color:#e74c3c">{pytest_res['failed']}</div><div class="label">Failed</div></div>
                <div class="card"><div class="value">{pytest_res['errors']}</div><div class="label">Errors</div></div>
                <div class="card"><div class="value">{pytest_res['total']}</div><div class="label">Total</div></div>
            </div>
            <details>
                <summary>Detailed test results ({pytest_res['total']} tests)</summary>
                <table>
                    <tr><th></th><th>Test</th><th>Result</th></tr>
                    {pytest_rows}
                </table>
            </details>
        </div>

        <div class="section">
            <h2>4. Unittest Results</h2>
            <div class="summary-grid">
                <div class="card"><div class="value" style="color:#27ae60">{unittest_res['passed']}</div><div class="label">Passed</div></div>
                <div class="card"><div class="value" style="color:#e74c3c">{unittest_res['failed']}</div><div class="label">Failed</div></div>
                <div class="card"><div class="value">{unittest_res['total']}</div><div class="label">Total</div></div>
                <div class="card"><div class="value">{unittest_res['duration']:.1f}s</div><div class="label">Duration</div></div>
            </div>
        </div>

        <div class="section">
            <h2>5. Project Verification (verify.py)</h2>
            <div class="summary-grid">
                <div class="card"><div class="value" style="color:#27ae60">{verify_res['passed']}</div><div class="label">Passed</div></div>
                <div class="card"><div class="value" style="color:#f39c12">{verify_res['warnings']}</div><div class="label">Warnings</div></div>
                <div class="card"><div class="value" style="color:#e74c3c">{verify_res['failed']}</div><div class="label">Failed</div></div>
            </div>
        </div>
    </body>
    </html>
    """
    )
    return html


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Running Test Suite & Generating Formal Report")
    print("=" * 60)

    print("\n[1/4] Collecting pipeline metrics ...")
    metrics = collect_pipeline_metrics()

    print("[2/4] Running pytest ...")
    pytest_res = run_pytest()
    print(
        f"      pytest: {pytest_res['passed']} passed, {pytest_res['failed']} failed, {pytest_res['total']} total"
    )

    print("[3/4] Running unittest ...")
    unittest_res = run_unittest()
    print(
        f"      unittest: {unittest_res['passed']} passed, {unittest_res['failed']} failed, {unittest_res['total']} total"
    )

    print("[4/4] Running verify.py ...")
    verify_res = run_verify()
    print(
        f"      verify: {verify_res['passed']} passed, {verify_res['warnings']} warnings, {verify_res['failed']} failed"
    )

    # Generate reports
    text_report = generate_text_report(pytest_res, unittest_res, verify_res, metrics)
    html_report = generate_html_report(pytest_res, unittest_res, verify_res, metrics)

    text_path = REPORTS_DIR / f"test_report_{TIMESTAMP}.txt"
    html_path = REPORTS_DIR / f"test_report_{TIMESTAMP}.html"

    text_path.write_text(text_report, encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")

    # Also save latest
    (REPORTS_DIR / "test_report_latest.txt").write_text(text_report, encoding="utf-8")
    (REPORTS_DIR / "test_report_latest.html").write_text(html_report, encoding="utf-8")

    print("\n" + text_report)
    print(f"\nReports saved:")
    print(f"  Text: {text_path}")
    print(f"  HTML: {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IGNORED_DIRS = {".git", ".venv", "__pycache__"}

EXPECTED_DIRS = [
    ROOT / "data",
    ROOT / "data" / "external",
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "models",
    ROOT / "scr",
    ROOT / "scr" / "data",
    ROOT / "scr" / "Model",
    ROOT / "scr" / "visuals",
]

EXPECTED_FILES = [
    ROOT / "main.py",
    ROOT / "streamlit_AdvProgrammingFinal.py",
    ROOT / "requirements.txt",
    ROOT / "scr" / "data" / "data_download.py",
    ROOT / "scr" / "data" / "make_dataset.py",
    ROOT / "scr" / "Model" / "train_models.py",
    ROOT / "scr" / "Model" / "hyper_tuning.py",
    ROOT / "scr" / "Model" / "predict_models.py",
    ROOT / "scr" / "visuals" / "visualize.py",
]

EXPECTED_ARTIFACTS = [
    ROOT / "models" / "arima.pkl",
    ROOT / "models" / "garch.pkl",
    ROOT / "models" / "lstm.pkl",
    ROOT / "data" / "processed" / "tesla_processed_dataset.csv",
]

REQUIREMENTS_IMPORT_MAP = {
    "scikit-learn": "sklearn",
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def _pass(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="PASS", detail=detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="FAIL", detail=detail)


def _warn(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="WARN", detail=detail)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def iter_python_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in IGNORED_DIRS for part in p.parts):
            continue
        paths.append(p)
    return sorted(paths)


def parse_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []

    packages: list[str] = []
    for line in read_text(path).splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        base = cleaned.split(";")[0].strip()
        for marker in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
            if marker in base:
                base = base.split(marker)[0].strip()
                break
        if base:
            packages.append(base)
    return packages


def import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_project_structure() -> list[CheckResult]:
    results: list[CheckResult] = []

    for directory in EXPECTED_DIRS:
        if directory.exists() and directory.is_dir():
            results.append(_pass("Directory exists", str(directory.relative_to(ROOT))))
        else:
            results.append(_fail("Directory exists", str(directory.relative_to(ROOT))))

    for file_path in EXPECTED_FILES:
        if file_path.exists() and file_path.is_file():
            results.append(_pass("File exists", str(file_path.relative_to(ROOT))))
        else:
            results.append(_fail("File exists", str(file_path.relative_to(ROOT))))

    return results


def check_python_syntax_all() -> list[CheckResult]:
    results: list[CheckResult] = []
    for path in iter_python_files(ROOT):
        try:
            ast.parse(read_text(path), filename=str(path))
            results.append(_pass("Syntax", str(path.relative_to(ROOT))))
        except SyntaxError as exc:
            results.append(_fail("Syntax", f"{path.relative_to(ROOT)} -> {exc}"))
    return results


def check_absolute_paths_all() -> list[CheckResult]:
    results: list[CheckResult] = []

    for path in iter_python_files(ROOT):
        text = read_text(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.strip()
                if (
                    (value.startswith("/") and len(value) > 1)
                    or value.startswith("C:/")
                    or value.startswith("C:\\\\")
                ) and value not in {"C:/", "C:\\\\"}:
                    found.append(value)

        if found:
            preview = ", ".join(found[:2])
            extra = "" if len(found) <= 2 else f" (+{len(found) - 2} more)"
            results.append(
                _warn(
                    "Absolute path literals",
                    f"{path.relative_to(ROOT)} -> {preview}{extra}",
                )
            )
        else:
            results.append(_pass("Absolute path literals", str(path.relative_to(ROOT))))

    return results


def _clear_import_cache(module_name: str) -> None:
    """Only clear project-level modules (scr.*) to avoid breaking stdlib/third-party."""
    top_level = module_name.split(".")[0]
    if top_level != "scr":
        return
    keys = {module_name}
    parts = module_name.split(".")
    for i in range(1, len(parts) + 1):
        keys.add(".".join(parts[:i]))
    for key in keys:
        if key in sys.modules:
            del sys.modules[key]


def check_imports(module_names: list[str], label: str) -> list[CheckResult]:
    results: list[CheckResult] = []

    for name in module_names:
        try:
            _clear_import_cache(name)
            importlib.import_module(name)
            results.append(_pass(label, f"{name}"))
        except Exception as exc:
            results.append(_fail(label, f"{name} -> {exc}"))

    return results


def _extract_imports_from_file(path: Path) -> list[str]:
    tree = ast.parse(read_text(path), filename=str(path))
    names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                names.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or not node.module:
                continue
            names.add(node.module)

    return sorted(names)


def check_imports_from_entry_files() -> list[CheckResult]:
    results: list[CheckResult] = []

    entry_files = [ROOT / "main.py", ROOT / "streamlit_AdvProgrammingFinal.py"]
    discovered: set[str] = set()

    for path in entry_files:
        if not path.exists():
            continue
        try:
            imports = _extract_imports_from_file(path)
            for name in imports:
                # Inspect project modules and app dependencies only.
                top_level = name.split(".")[0]
                if top_level in {"scr", "pandas", "streamlit", "pickle", "pathlib"}:
                    discovered.add(name)
        except Exception as exc:
            results.append(_fail("Parse imports", f"{path.name} -> {exc}"))

    if discovered:
        safe_imports: list[str] = []
        for name in sorted(discovered):
            top_level = name.split(".")[0]
            if (ROOT / f"{top_level}.py").exists() and top_level == "streamlit":
                results.append(
                    _warn(
                        "Entry import check",
                        f"Skipped {name}: local file {top_level}.py shadows package import",
                    )
                )
                continue
            safe_imports.append(name)

        results.extend(check_imports(safe_imports, "Entry import check"))
    else:
        results.append(
            _warn("Entry import check", "No imports discovered for inspection")
        )

    return results


def check_requirements_dependencies() -> list[CheckResult]:
    results: list[CheckResult] = []
    req_path = ROOT / "requirements.txt"
    packages = parse_requirements(req_path)

    if not packages:
        results.append(
            _warn("Requirements", "No packages parsed from requirements.txt")
        )
        return results

    results.append(_pass("Requirements", f"Parsed {len(packages)} package(s)"))

    for pkg in packages:
        module_name = REQUIREMENTS_IMPORT_MAP.get(pkg, pkg.replace("-", "_"))
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            results.append(
                _fail("Dependency available", f"{pkg} (import {module_name})")
            )
        else:
            results.append(
                _pass("Dependency available", f"{pkg} (import {module_name})")
            )

    return results


def check_module_contracts() -> list[CheckResult]:
    results: list[CheckResult] = []

    module_files = {
        "data_download": ROOT / "scr" / "data" / "data_download.py",
        "make_dataset": ROOT / "scr" / "data" / "make_dataset.py",
        "train_models": ROOT / "scr" / "Model" / "train_models.py",
        "hyper_tuning": ROOT / "scr" / "Model" / "hyper_tuning.py",
        "predict_models": ROOT / "scr" / "Model" / "predict_models.py",
        "visualize": ROOT / "scr" / "visuals" / "visualize.py",
    }

    try:
        data_download = import_from_path(
            "verify_data_download", module_files["data_download"]
        )
        make_dataset = import_from_path(
            "verify_make_dataset", module_files["make_dataset"]
        )
        train_models = import_from_path(
            "verify_train_models", module_files["train_models"]
        )
        hyper_tuning = import_from_path(
            "verify_hyper_tuning", module_files["hyper_tuning"]
        )
        predict_models = import_from_path(
            "verify_predict_models", module_files["predict_models"]
        )
        visualize = import_from_path("verify_visualize", module_files["visualize"])
    except Exception as exc:
        results.append(_fail("Module dynamic imports", str(exc)))
        return results

    expected = [
        (data_download, "download_all", 4),
        (data_download, "load_raw_datasets", 1),
        (data_download, "merge_raw", 1),
        (make_dataset, "load_phase2_data", 3),
        (make_dataset, "prepare_features", 3),
        (make_dataset, "split_data", 4),
        (make_dataset, "scale_features", 2),
        (train_models, "get_models", 0),
        (train_models, "train_all_models", 7),
        (train_models, "save_models", 2),
        (train_models, "cross_validate_models", 5),
        (hyper_tuning, "tune_all_models", 5),
        (predict_models, "evaluate_model", 2),
        (predict_models, "build_results_table", 1),
        (predict_models, "get_best_model", 2),
        (visualize, "plot_model_comparison", 2),
        (visualize, "plot_actual_vs_predicted", 5),
        (visualize, "plot_residual_analysis", 5),
        (visualize, "plot_feature_importance", 4),
    ]

    for module, symbol_name, expected_args in expected:
        if not hasattr(module, symbol_name):
            results.append(_fail("Symbol exists", symbol_name))
            continue

        obj = getattr(module, symbol_name)
        params = inspect.signature(obj).parameters
        if len(params) == expected_args:
            results.append(
                _pass("Symbol signature", f"{symbol_name}({len(params)} args)")
            )
        else:
            results.append(
                _fail(
                    "Symbol signature",
                    f"{symbol_name} expected {expected_args} args, got {len(params)}",
                )
            )

    return results


def check_artifacts() -> list[CheckResult]:
    results: list[CheckResult] = []
    for artifact in EXPECTED_ARTIFACTS:
        rel = str(artifact.relative_to(ROOT))
        if artifact.exists():
            size = artifact.stat().st_size
            results.append(_pass("Artifact exists", f"{rel} ({size} bytes)"))
        else:
            results.append(_warn("Artifact exists", f"{rel} (not found yet)"))
    return results


def check_model_predict_smoke() -> list[CheckResult]:
    results: list[CheckResult] = []
    model_path = ROOT / "models" / "arima.pkl"
    if not model_path.exists():
        results.append(_warn("Model smoke", "models/arima.pkl not found"))
        return results

    try:
        import numpy as np
    except Exception as exc:
        results.append(_fail("Model smoke", f"numpy import failed -> {exc}"))
        return results

    try:
        with model_path.open("rb") as model_file:
            model = pickle.load(model_file)

        has_predict = hasattr(model, "predict")
        has_fit = hasattr(model, "fit")
        if not (has_predict and has_fit):
            results.append(
                _fail("Model smoke", "Loaded object must expose fit() and predict()")
            )
            return results

        # 18 features matching FEATURE_COLS
        sample = np.zeros((5, 18))
        try:
            prediction = model.predict(sample)
            results.append(_pass("Model smoke", f"predict() OK -> {prediction[0]:.2f}"))
        except RuntimeError as exc:
            message = str(exc)
            if "must be fit before predict" in message:
                results.append(_pass("Model smoke", "predict() requires fit() first (expected for unfitted artifact)"))
            else:
                results.append(_fail("Model smoke", message))
    except Exception as exc:
        results.append(_fail("Model smoke", str(exc)))

    return results


def check_preprocess_smoke() -> list[CheckResult]:
    results: list[CheckResult] = []

    processed_path = ROOT / "data" / "processed" / "tesla_processed_dataset.csv"

    if not processed_path.exists():
        results.append(
            _warn("Preprocess smoke", f"Missing {processed_path.relative_to(ROOT)}")
        )
        return results

    try:
        import pandas as pd

        df = pd.read_csv(processed_path, parse_dates=["Date"])
        ok = len(df) > 0 and "TSLA_Close" in df.columns
        detail = f"rows={len(df)}, columns={len(df.columns)}, has_target={'TSLA_Close' in df.columns}"
        results.append(
            _pass("Preprocess smoke", detail)
            if ok
            else _fail("Preprocess smoke", detail)
        )
    except Exception as exc:
        results.append(_fail("Preprocess smoke", str(exc)))

    return results


def check_streamlit_upload_readiness() -> list[CheckResult]:
    results: list[CheckResult] = []
    streamlit_path = ROOT / "streamlit_AdvProgrammingFinal.py"

    if not streamlit_path.exists():
        results.append(
            _fail(
                "Streamlit upload readiness", "streamlit_AdvProgrammingFinal.py missing"
            )
        )
        return results

    text = read_text(streamlit_path)

    required_markers = [
        "st.set_page_config",
        "st.tabs",
        "st.cache_data",
        "st.cache_resource",
    ]

    for marker in required_markers:
        if marker in text:
            results.append(_pass("Streamlit marker", marker))
        else:
            results.append(_warn("Streamlit marker", f"Not found: {marker}"))

    req_text = (
        read_text(ROOT / "requirements.txt")
        if (ROOT / "requirements.txt").exists()
        else ""
    )
    if "streamlit" in req_text:
        results.append(
            _pass("Streamlit dependency", "requirements.txt contains streamlit")
        )
    else:
        results.append(
            _fail("Streamlit dependency", "requirements.txt missing streamlit")
        )

    return results


def run_deep_checks() -> list[CheckResult]:
    # Deep checks are optional because they can be expensive.
    results: list[CheckResult] = []

    try:
        from scr.data.data_download import load_raw_datasets, merge_raw
        from scr.data.make_dataset import (
            _engineer_features,
            prepare_features,
            split_data,
            scale_features,
        )
        from scr.Model import train_models, predict_models

        ext_dir = ROOT / "data" / "external"
        if not ext_dir.exists() or not any(ext_dir.glob("*.csv")):
            return [_warn("Deep check", "Skipped: no cached CSVs in data/external/")]

        datasets = load_raw_datasets()
        merged = merge_raw(datasets)
        df = _engineer_features(merged)
        model_df = prepare_features(df)
        X_train, X_test, y_train, y_test, train_df, test_df = split_data(model_df)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        models = train_models.get_models()
        all_results = train_models.train_all_models(
            models, X_train, y_train, X_train_scaled, X_test, X_test_scaled, y_test
        )

        best_name, _, best_preds = predict_models.get_best_model(all_results, models)
        best_r2 = all_results[best_name]["R²"]

        if best_r2 > 0.5:
            results.append(
                _pass("Deep check", f"Best model {best_name} R²={best_r2:.4f}")
            )
        else:
            results.append(
                _warn("Deep check", f"Best model {best_name} R²={best_r2:.4f} (low)")
            )
    except Exception as exc:
        results.append(_fail("Deep check", str(exc)))

    return results


def run_checks(deep: bool) -> list[CheckResult]:
    results: list[CheckResult] = []

    results.extend(check_project_structure())
    results.extend(check_python_syntax_all())
    results.extend(check_absolute_paths_all())

    results.extend(
        check_imports(
            [
                "scr.data.data_download",
                "scr.data.make_dataset",
                "scr.Model.train_models",
                "scr.Model.hyper_tuning",
                "scr.Model.predict_models",
                "scr.visuals.visualize",
            ],
            "Core module import",
        )
    )
    results.extend(check_imports_from_entry_files())
    results.extend(check_requirements_dependencies())
    results.extend(check_module_contracts())
    results.extend(check_preprocess_smoke())
    results.extend(check_artifacts())
    results.extend(check_model_predict_smoke())
    results.extend(check_streamlit_upload_readiness())

    if deep:
        results.extend(run_deep_checks())

    return results


def print_report(results: list[CheckResult]) -> int:
    status_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    sorted_results = sorted(
        results, key=lambda r: (status_order.get(r.status, 99), r.name, r.detail)
    )

    print("=" * 80)
    print("Comprehensive project inspection report")
    print("=" * 80)

    pass_count = 0
    warn_count = 0
    fail_count = 0

    for item in sorted_results:
        print(f"[{item.status}] {item.name}: {item.detail}")
        if item.status == "PASS":
            pass_count += 1
        elif item.status == "WARN":
            warn_count += 1
        elif item.status == "FAIL":
            fail_count += 1

    print("-" * 80)
    print(f"Summary -> PASS: {pass_count}, WARN: {warn_count}, FAIL: {fail_count}")
    print("Hint: use --deep for an extended runtime check.")

    return 1 if fail_count > 0 else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect project structure and pipeline health"
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Run extra runtime checks (slower)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_checks(deep=args.deep)
    return print_report(results)


if __name__ == "__main__":
    raise SystemExit(main())

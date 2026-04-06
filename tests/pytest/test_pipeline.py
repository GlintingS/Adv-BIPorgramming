"""Tests for the end-to-end pipeline and project structure."""

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestProjectStructure:
    """Verify expected folders and files exist."""

    EXPECTED_DIRS = [
        "data/external",
        "data/raw",
        "data/processed",
        "models",
        "scr/data",
        "scr/Model",
        "scr/visuals",
    ]

    EXPECTED_FILES = [
        "main.py",
        "streamlit_AdvProgrammingFinal.py",
        "requirements.txt",
        "scr/data/data_download.py",
        "scr/data/make_dataset.py",
        "scr/Model/train_models.py",
        "scr/Model/hyper_tuning.py",
        "scr/Model/predict_models.py",
        "scr/visuals/visualize.py",
    ]

    @pytest.mark.parametrize("rel_path", EXPECTED_DIRS)
    def test_directory_exists(self, rel_path):
        assert (PROJECT_ROOT / rel_path).is_dir()

    @pytest.mark.parametrize("rel_path", EXPECTED_FILES)
    def test_file_exists(self, rel_path):
        assert (PROJECT_ROOT / rel_path).is_file()

    @pytest.mark.parametrize("rel_path", EXPECTED_FILES)
    def test_python_syntax(self, rel_path):
        if rel_path.endswith(".py"):
            source = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
            ast.parse(source)  # raises SyntaxError on failure


class TestDependencies:
    """Verify all required packages are importable."""

    PACKAGES = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("statsmodels", "statsmodels"),
        ("arch", "arch"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("yfinance", "yfinance"),
        ("fredapi", "fredapi"),
        ("streamlit", "streamlit"),
    ]

    @pytest.mark.parametrize("pkg_name,import_name", PACKAGES)
    def test_package_importable(self, pkg_name, import_name):
        spec = importlib.util.find_spec(import_name)
        assert spec is not None, f"{pkg_name} (import {import_name}) not installed"


class TestCoreModuleImports:
    """Verify all project modules import successfully."""

    MODULES = [
        "scr.data.data_download",
        "scr.data.make_dataset",
        "scr.Model.train_models",
        "scr.Model.hyper_tuning",
        "scr.Model.predict_models",
        "scr.visuals.visualize",
    ]

    @pytest.mark.parametrize("module_name", MODULES)
    def test_module_imports(self, module_name):
        importlib.import_module(module_name)


class TestModelArtifacts:
    """Verify model pkl files and processed data exist after pipeline run."""

    EXPECTED_MODELS = [
        "arima.pkl",
        "garch.pkl",
        "lstm.pkl",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_MODELS)
    def test_model_file_exists(self, filename):
        path = PROJECT_ROOT / "models" / filename
        assert path.exists(), f"Missing model artifact: {filename}"
        assert path.stat().st_size > 0, f"Empty model file: {filename}"

    def test_processed_dataset_exists(self):
        path = PROJECT_ROOT / "data" / "processed" / "tesla_processed_dataset.csv"
        assert path.exists()

    def test_merged_dataset_exists(self):
        path = PROJECT_ROOT / "data" / "raw" / "tesla_merged_dataset.csv"
        assert path.exists()

    def test_external_csvs_exist(self):
        ext = PROJECT_ROOT / "data" / "external"
        csvs = list(ext.glob("*.csv"))
        assert len(csvs) == 7, f"Expected 7 external CSVs, found {len(csvs)}"


class TestEndToEndPipeline:
    """Run the full pipeline on cached data and verify outputs."""

    def test_full_pipeline(self, trained_models, split_data_fixture):
        from scr.Model.predict_models import build_results_table, get_best_model

        models, results = trained_models
        table = build_results_table(results)

        # All three time-series models should be present
        assert len(table) == 3

        # Best model should have decent R²
        best_name, best_model, best_preds = get_best_model(results, models)
        assert results[best_name]["R²"] > 0.5
        assert results[best_name]["MAPE (%)"] < 50

        # Predictions shape matches test set
        _, _, _, y_test, _, _ = split_data_fixture
        assert len(best_preds) == len(y_test)

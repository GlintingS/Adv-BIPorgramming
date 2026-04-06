import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_comparison(results_table, return_fig=False):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics = ["MAE", "RMSE", "R²", "MAPE (%)"]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]

    for ax, metric, color in zip(axes, metrics, colors):
        vals = results_table[metric].sort_values(ascending=(metric != "R²"))
        vals.plot(kind="barh", ax=ax, color=color, edgecolor="black")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel(metric)

    plt.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()


def plot_actual_vs_predicted(test_dates, y_test, y_pred, model_name, return_fig=False):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        test_dates, np.asarray(y_test), label="Actual", color="#1f4e79", linewidth=1.5
    )
    ax.plot(
        test_dates,
        y_pred,
        label=f"Predicted ({model_name})",
        color="#e74c3c",
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_title(
        f"Tesla Closing Price — Actual vs Predicted ({model_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("TSLA Close (USD)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()


def plot_residual_analysis(test_dates, y_test, y_pred, model_name, return_fig=False):
    residuals = np.asarray(y_test) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(test_dates, residuals, color="#2196F3", linewidth=0.8)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Residuals Over Time")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual (USD)")
    axes[0].tick_params(axis="x", rotation=45)

    sns.histplot(residuals, bins=40, kde=True, color="#FF5722", ax=axes[1])
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual (USD)")

    axes[2].scatter(y_pred, np.asarray(y_test), alpha=0.5, s=15, color="#4CAF50")
    min_val = min(np.min(y_pred), np.min(y_test))
    max_val = max(np.max(y_pred), np.max(y_test))
    axes[2].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    axes[2].set_title("Predicted vs Actual")
    axes[2].set_xlabel("Predicted (USD)")
    axes[2].set_ylabel("Actual (USD)")

    plt.suptitle(
        f"Residual Analysis — {model_name}", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()


def plot_feature_importance(model, feature_cols, model_name, return_fig=False):
    importances = model.feature_importances_
    fi_df = pd.DataFrame(
        {"Feature": feature_cols, "Importance": importances}
    ).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color="#3f51b5", edgecolor="black")
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return fi_df.sort_values("Importance", ascending=False)


def plot_all_models_overlay(test_dates, y_test, results, return_fig=False):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_dates, np.asarray(y_test), label="Actual", color="black", linewidth=2)

    palette = ["#e74c3c", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    for i, (name, metrics) in enumerate(results.items()):
        ax.plot(
            test_dates,
            metrics["Predictions"],
            label=name,
            linewidth=1.2,
            linestyle="--",
            color=palette[i % len(palette)],
        )

    ax.set_title(
        "All Models — Actual vs Predicted Tesla Close Price",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("TSLA Close (USD)")
    ax.legend(loc="best", fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()


def plot_monthly_error(monthly_err, model_name, return_fig=False):
    fig, ax = plt.subplots(figsize=(14, 5))
    monthly_err["MAE"].plot(kind="bar", color="#FF5722", edgecolor="black", ax=ax)
    ax.set_title(
        f"Monthly Mean Absolute Error — {model_name}", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE (USD)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()

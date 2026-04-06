import logging
import sys

from scr.data import make_dataset
from scr.Model import train_models, predict_models, hyper_tuning
from scr.visuals import visualize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # 1. Load Phase 2 dataset (downloads to data/external/, merged to data/raw/)
        logger.info("Loading data")
        df = make_dataset.load_phase2_data()

        # 2. Prepare features and split
        logger.info("Preparing features")
        model_df = make_dataset.prepare_features(df)
        X_train, X_test, y_train, y_test, train_df, test_df = make_dataset.split_data(
            model_df
        )
        X_train_scaled, X_test_scaled, _ = make_dataset.scale_features(X_train, X_test)

        # 3. Train all models (default hyperparameters)
        logger.info("Training models (default hyperparameters)")
        models = train_models.get_models()
        results = train_models.train_all_models(
            models, X_train, y_train, X_train_scaled, X_test, X_test_scaled, y_test
        )

        # 4. Cross-validation (default models)
        logger.info("Running cross-validation (default models)")
        cv_df = train_models.cross_validate_models(
            models, X_train, y_train, X_train_scaled
        )
        print("\n--- Cross-Validation Results (Default) ---")
        print(cv_df.to_string(index=False))

        # 5. Hyperparameter tuning
        logger.info("Running hyperparameter tuning")
        tuned_models, tuning_summary = hyper_tuning.tune_all_models(
            X_train, y_train, X_train_scaled
        )
        print("\n--- Hyperparameter Tuning Results ---")
        print(tuning_summary.to_string(index=False))

        # 6. Re-evaluate tuned models on test set
        logger.info("Evaluating tuned models on test set")
        tuned_results = train_models.train_all_models(
            tuned_models,
            X_train,
            y_train,
            X_train_scaled,
            X_test,
            X_test_scaled,
            y_test,
        )
        train_models.save_models(tuned_models)

        # 7. Evaluate best tuned model
        results_table = predict_models.build_results_table(tuned_results)
        best_name, _, best_preds = predict_models.get_best_model(tuned_results, tuned_models)
        dir_acc = predict_models.directional_accuracy(y_test, best_preds)
        monthly_err = predict_models.monthly_error_analysis(test_df, best_preds)

        # 8. Print summary
        print("\n" + "=" * 70)
        print("  PHASE 3 MODEL EVALUATION SUMMARY (TUNED)")
        print("=" * 70)
        print(
            f"\nDataset: {len(model_df)} rows  |  Features: {len(make_dataset.FEATURE_COLS)}  |  Target: TSLA_Close"
        )
        print(f"Training: {len(train_df)} rows  |  Test: {len(test_df)} rows\n")
        print("--- Test-Set Performance (sorted by RMSE) ---")
        for idx, row in results_table.iterrows():
            print(
                f"  {idx:25s}  MAE={row['MAE']:8.2f}  RMSE={row['RMSE']:8.2f}  R²={row['R²']:.4f}  MAPE={row['MAPE (%)']:.2f}%"
            )
        print(f"\nBest Model: {best_name}")
        print(
            f"  R²={tuned_results[best_name]['R²']:.4f}  MAPE={tuned_results[best_name]['MAPE (%)']:.2f}%  Dir. Accuracy={dir_acc:.1f}%"
        )
        print("=" * 70)

        # 9. Visualizations
        logger.info("Generating plots")
        test_dates = test_df["Date"].values
        visualize.plot_model_comparison(results_table)
        visualize.plot_actual_vs_predicted(test_dates, y_test, best_preds, best_name)
        visualize.plot_residual_analysis(test_dates, y_test, best_preds, best_name)
        visualize.plot_all_models_overlay(test_dates, y_test, tuned_results)
        visualize.plot_monthly_error(monthly_err, best_name)

        logger.info("Pipeline completed successfully")

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)

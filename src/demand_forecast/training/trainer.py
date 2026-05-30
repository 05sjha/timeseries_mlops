"""
demand_forecast/training/trainer.py
─────────────────────────────────────
Model training with MLflow experiment tracking.

MLOps KEY CONCEPTS demonstrated here:
  1. Experiment tracking  — every run logged to MLflow (params, metrics, artifacts)
  2. Model registry       — trained models promoted through staging → production
  3. Reproducibility      — random seeds, data version, and env pinned to each run
  4. Automated evaluation — model only promoted if it beats the champion
  5. Artifact logging     — feature importance, confusion matrix, data splits saved
"""

import logging
import os
import time
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

from demand_forecast.data.preprocessing import DataSplit

logger = logging.getLogger(__name__)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str = "") -> dict:
    """
    Compute regression metrics for demand forecasting.
    
    MAE: average absolute error in units — most interpretable for business
    MAPE: percentage error — good for comparing across products with different scale
    RMSE: penalises large errors — catches forecasting disasters
    R²: proportion of variance explained — directional quality signal
    """
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100   # as %
    rmse = np.sqrt(np.mean((y_true.values - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        f"{prefix}mae": round(mae, 4),
        f"{prefix}mape": round(mape, 4),
        f"{prefix}rmse": round(rmse, 4),
        f"{prefix}r2": round(r2, 4),
    }
    return metrics


# ── Trainer ───────────────────────────────────────────────────────────────────

class DemandForecaster:
    """
    Wraps XGBoost training with full MLflow tracking.

    MLOPS INSIGHT: The trainer class is intentionally separate from the model.
    The trainer handles the MLOps concerns (logging, registration, evaluation).
    The model handles only prediction. This separation makes it easy to swap
    the underlying algorithm without changing the MLOps plumbing.
    """

    def __init__(self, experiment_name: str, tracking_uri: str = "mlruns"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.model: XGBRegressor | None = None
        self.run_id: str | None = None

    def train(
        self,
        split: DataSplit,
        hyperparams: dict | None = None,
        run_name: str | None = None,
    ) -> str:
        """
        Train the model and log everything to MLflow.

        Returns:
            MLflow run_id — used to load the model later
        """
        if hyperparams is None:
            hyperparams = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 30,
                "random_state": 42,
            }

        run_name = run_name or f"xgb_{split.data_version}"

        with mlflow.start_run(run_name=run_name) as run:
            self.run_id = run.info.run_id
            logger.info(f"MLflow run started: {self.run_id}")

            # ── 1. Log parameters ─────────────────────────────────────────
            # EVERYTHING that affects the model goes here.
            # Future you (and reviewers) will thank present you.
            mlflow.log_params(hyperparams)
            mlflow.log_params({
                "model_type": "xgboost",
                "data_version": split.data_version,
                "n_features": len(split.feature_names),
                "train_rows": split.stats["train_rows"],
                "val_rows": split.stats["val_rows"],
                "split_date_val": split.split_date_val,
                "split_date_test": split.split_date_test,
                "train_period": split.stats["train_period"],
                "n_products": split.stats["n_products"],
                "python_version": os.popen("python --version").read().strip(),
            })

            # Log the feature list as an artifact — critical for serving
            mlflow.log_text(
                "\n".join(split.feature_names),
                "feature_names.txt"
            )
            # Log data stats
            mlflow.log_dict(split.stats, "data_stats.json")

            # ── 2. Train ──────────────────────────────────────────────────
            start = time.time()
            self.model = XGBRegressor(**hyperparams, tree_method="hist")

            # XGBoost's eval_set enables early stopping on validation loss
            # This prevents overfitting without a manual n_estimators grid search
            self.model.fit(
                split.X_train,
                split.y_train,
                eval_set=[(split.X_val, split.y_val)],
                verbose=False,
            )
            train_time = time.time() - start
            mlflow.log_metric("train_time_seconds", round(train_time, 2))
            logger.info(f"Training complete in {train_time:.1f}s")

            # ── 3. Evaluate on all splits ─────────────────────────────────
            val_preds = self.model.predict(split.X_val)
            test_preds = self.model.predict(split.X_test)
            train_preds = self.model.predict(split.X_train)

            metrics = {}
            metrics.update(compute_metrics(split.y_train, train_preds, "train_"))
            metrics.update(compute_metrics(split.y_val, val_preds, "val_"))
            metrics.update(compute_metrics(split.y_test, test_preds, "test_"))

            mlflow.log_metrics(metrics)

            # Log the gap between val and test MAE — large gap = overfit to val
            metrics["val_test_mae_gap"] = round(
                abs(metrics["val_mae"] - metrics["test_mae"]), 4
            )
            mlflow.log_metric("val_test_mae_gap", metrics["val_test_mae_gap"])

            logger.info(
                f"val_mae={metrics['val_mae']:.2f} | "
                f"test_mae={metrics['test_mae']:.2f} | "
                f"test_mape={metrics['test_mape']:.1f}% | "
                f"test_r2={metrics['test_r2']:.3f}"
            )

            # ── 4. Log feature importance ─────────────────────────────────
            importance_df = pd.DataFrame({
                "feature": split.feature_names,
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False)

            mlflow.log_text(
                importance_df.to_csv(index=False),
                "feature_importance.csv"
            )

            # ── 5. Log the model ──────────────────────────────────────────
            # mlflow.xgboost.log_model stores the model + its signature.
            # The SIGNATURE captures input schema — used to validate
            # requests at serving time before prediction even runs.
            from mlflow.models import infer_signature
            signature = infer_signature(
                split.X_val,
                self.model.predict(split.X_val)
            )

            mlflow.xgboost.log_model(
                self.model,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"{self.experiment_name}-model",
            )

            logger.info(f"Model logged to MLflow. Run ID: {self.run_id}")
            return self.run_id

    def evaluate_vs_champion(
        self,
        challenger_run_id: str,
        champion_run_id: str | None,
        primary_metric: str = "test_mae",
        promotion_threshold: float = 0.05,
    ) -> bool:
        """
        Compare challenger model against the current champion.

        MLOPS INSIGHT: Automated model promotion prevents human bias.
        We only promote if the challenger is meaningfully better —
        not just within noise.

        Returns:
            True if challenger should be promoted to production
        """
        challenger_metrics = mlflow.get_run(challenger_run_id).data.metrics
        challenger_score = challenger_metrics.get(primary_metric)

        if champion_run_id is None:
            logger.info("No champion model found — promoting challenger by default")
            return True

        champion_metrics = mlflow.get_run(champion_run_id).data.metrics
        champion_score = champion_metrics.get(primary_metric)

        # For error metrics (MAE, RMSE) lower is better
        improvement = (champion_score - challenger_score) / champion_score

        logger.info(
            f"Champion {primary_metric}: {champion_score:.4f} | "
            f"Challenger {primary_metric}: {challenger_score:.4f} | "
            f"Improvement: {improvement:.1%}"
        )

        should_promote = improvement >= promotion_threshold
        if should_promote:
            logger.info(f"Challenger beats champion by {improvement:.1%} — PROMOTE")
        else:
            logger.info(f"Challenger does not beat threshold {promotion_threshold:.0%} — KEEP CHAMPION")

        mlflow.log_metric("promotion_improvement", round(improvement, 4))
        return should_promote

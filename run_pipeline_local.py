"""
run_pipeline_local.py
─────────────────────
Runs the full Phase 1–3 pipeline locally (no cloud needed).
Execute with: python run_pipeline_local.py

This is the "smoke test" every MLOps engineer runs before
pushing to the cloud to catch issues fast and cheaply.
"""

import logging
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from demand_forecast.data.preprocessing import run_preprocessing
from demand_forecast.training.trainer import DemandForecaster

def main():
    print("\n" + "="*60)
    print("  DEMAND FORECAST — LOCAL PIPELINE RUN")
    print("="*60 + "\n")

    # ── Phase 2: Data ─────────────────────────────────────────────
    print("Phase 2: Preprocessing & Feature Engineering")
    split = run_preprocessing(
        lag_windows=[1, 2, 4, 8, 12],
        rolling_windows=[4, 8, 13],
        test_weeks=13,
        validation_weeks=13,
    )

    print(f"  Data version   : {split.data_version}")
    print(f"  Train period   : {split.stats['train_period']}")
    print(f"  Val period     : {split.stats['val_period']}")
    print(f"  Test period    : {split.stats['test_period']}")
    print(f"  Train rows     : {split.stats['train_rows']:,}")
    print(f"  Features       : {len(split.feature_names)}")
    print(f"  Products       : {split.stats['n_products']}")
    print()

    # ── Phase 3: Training + MLflow ────────────────────────────────
    print("Phase 3: Model Training + MLflow Tracking")
    trainer = DemandForecaster(
        experiment_name="demand-forecast-local",
        tracking_uri="mlruns",
    )

    run_id = trainer.train(
        split=split,
        hyperparams={
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 20,
            "random_state": 42,
        },
        run_name="local_dev_run_01",
    )

    print(f"\n  MLflow run ID  : {run_id}")
    print(f"  Experiment     : demand-forecast-local")
    print(f"\n  View results: mlflow ui --backend-store-uri mlruns")
    print("\n" + "="*60)
    print("  Pipeline run complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

"""
demand_forecast/monitoring/drift_detector.py
──────────────────────────────────────────────
Production model monitoring: data drift and performance degradation detection.

MLOPS KEY CONCEPTS:
  1. Data drift     — input feature distributions shift from training baseline
  2. Concept drift  — relationship between features and target changes
  3. Performance monitoring — MAE/MAPE tracked over time in production
  4. Automated alerts — trigger retraining when drift exceeds threshold
  5. Baseline storage — what does "normal" look like? Stored at training time.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Summary of drift detection results for one monitoring run."""
    monitoring_date: str
    features_checked: int
    features_drifted: list[str]
    drift_scores: dict[str, float]       # feature → Jensen-Shannon divergence
    overall_drift_detected: bool
    performance_mae: float | None
    performance_degraded: bool
    recommendation: str                   # "RETRAIN" | "ALERT" | "OK"


class DriftDetector:
    """
    Detects data and performance drift in production.

    MLOPS INSIGHT: This is what keeps your model honest after deployment.
    Without monitoring:
    - Silent accuracy degradation for weeks/months
    - Root cause analysis is impossible
    - Business stakeholders lose trust in ML
    With monitoring:
    - Catch drift early (before it hurts accuracy)
    - Have data to justify retraining decisions
    - Build trust through transparency
    """

    def __init__(
        self,
        baseline_stats_path: str | Path,
        drift_threshold: float = 0.15,
        mae_degradation_threshold: float = 0.20,
    ):
        """
        Args:
            baseline_stats_path: Path to the JSON file with training data statistics
                                  (saved at training time, see save_baseline())
            drift_threshold: Jensen-Shannon divergence threshold. 0 = identical,
                             1 = completely different. 0.15 is a good starting point.
            mae_degradation_threshold: Alert if production MAE degrades by this fraction
                                       vs the training MAE.
        """
        self.drift_threshold = drift_threshold
        self.mae_degradation_threshold = mae_degradation_threshold
        self.baseline_stats: dict = {}

        if Path(baseline_stats_path).exists():
            with open(baseline_stats_path) as f:
                self.baseline_stats = json.load(f)
            logger.info(f"Loaded baseline stats for {len(self.baseline_stats)} features")
        else:
            logger.warning(f"Baseline stats not found at {baseline_stats_path}")

    @staticmethod
    def save_baseline(df: pd.DataFrame, output_path: str | Path) -> None:
        """
        Save training data statistics as the monitoring baseline.

        CALL THIS AT TRAINING TIME, before deployment.
        This is the "what normal looks like" reference.
        """
        stats_dict = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            stats_dict[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "p5": float(series.quantile(0.05)),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
                # Store a histogram for JS divergence computation
                "hist_counts": np.histogram(series, bins=20)[0].tolist(),
                "hist_edges": np.histogram(series, bins=20)[1].tolist(),
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stats_dict, f, indent=2)

        logger.info(f"Baseline stats saved to {output_path} ({len(stats_dict)} features)")

    @staticmethod
    def _jensen_shannon_divergence(p_hist: list, q_hist: list) -> float:
        """
        Compute Jensen-Shannon divergence between two histograms.

        JSD measures how different two distributions are:
          0.0  = identical distributions
          0.15 = noticeable drift (our alert threshold)
          1.0  = completely different distributions

        JSD is preferred over KL divergence because it's:
        - Symmetric: JSD(P, Q) == JSD(Q, P)
        - Bounded: always in [0, 1]
        - Defined even when distributions have non-overlapping support
        """
        p = np.array(p_hist, dtype=float) + 1e-10   # add epsilon to avoid log(0)
        q = np.array(q_hist, dtype=float) + 1e-10
        p /= p.sum()
        q /= q.sum()

        m = 0.5 * (p + q)
        jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        return float(np.clip(jsd, 0, 1))

    def check_feature_drift(
        self,
        production_df: pd.DataFrame,
        features_to_check: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Compute drift score for each feature.

        Returns:
            Dict mapping feature_name → JSD score.
            Scores above self.drift_threshold indicate drift.
        """
        if not self.baseline_stats:
            logger.error("No baseline stats loaded — cannot compute drift")
            return {}

        features = features_to_check or list(self.baseline_stats.keys())
        drift_scores = {}

        for feature in features:
            if feature not in self.baseline_stats:
                continue
            if feature not in production_df.columns:
                logger.warning(f"Feature '{feature}' in baseline but not in production data")
                continue

            baseline = self.baseline_stats[feature]
            prod_series = production_df[feature].dropna()

            if len(prod_series) < 30:
                logger.warning(f"Too few samples for '{feature}' ({len(prod_series)}) — skipping")
                continue

            # Compute production histogram using the same bin edges as baseline
            bin_edges = baseline["hist_edges"]
            prod_hist, _ = np.histogram(prod_series, bins=bin_edges)

            jsd = self._jensen_shannon_divergence(baseline["hist_counts"], prod_hist)
            drift_scores[feature] = round(jsd, 4)

            if jsd >= self.drift_threshold:
                logger.warning(
                    f"DRIFT DETECTED | feature='{feature}' | "
                    f"JSD={jsd:.3f} (threshold={self.drift_threshold})"
                )

        return drift_scores

    def check_performance_degradation(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        baseline_mae: float,
    ) -> tuple[float, bool]:
        """
        Check if production MAE has degraded vs training baseline.

        Returns:
            (current_mae, is_degraded)
        """
        current_mae = float(np.mean(np.abs(y_true.values - y_pred)))
        degradation = (current_mae - baseline_mae) / baseline_mae
        is_degraded = degradation >= self.mae_degradation_threshold

        if is_degraded:
            logger.warning(
                f"PERFORMANCE DEGRADATION | "
                f"baseline_mae={baseline_mae:.2f} | "
                f"current_mae={current_mae:.2f} | "
                f"degradation={degradation:.1%}"
            )

        return current_mae, is_degraded

    def run_monitoring_check(
        self,
        production_df: pd.DataFrame,
        y_true: pd.Series | None = None,
        y_pred: np.ndarray | None = None,
        baseline_mae: float | None = None,
        monitoring_date: str | None = None,
    ) -> DriftReport:
        """
        Run the full monitoring suite and produce a DriftReport.

        Call this weekly (via the Airflow DAG) with a sample of production data.
        """
        from datetime import date
        monitoring_date = monitoring_date or str(date.today())

        # Check feature drift
        drift_scores = self.check_feature_drift(production_df)
        drifted_features = [
            f for f, score in drift_scores.items()
            if score >= self.drift_threshold
        ]
        drift_detected = len(drifted_features) > 0

        # Check performance (if actuals are available)
        current_mae = None
        performance_degraded = False
        if y_true is not None and y_pred is not None and baseline_mae is not None:
            current_mae, performance_degraded = self.check_performance_degradation(
                y_true, y_pred, baseline_mae
            )

        # Recommendation logic
        if performance_degraded or (drift_detected and len(drifted_features) >= 3):
            recommendation = "RETRAIN"
        elif drift_detected:
            recommendation = "ALERT"
        else:
            recommendation = "OK"

        report = DriftReport(
            monitoring_date=monitoring_date,
            features_checked=len(drift_scores),
            features_drifted=drifted_features,
            drift_scores=drift_scores,
            overall_drift_detected=drift_detected,
            performance_mae=current_mae,
            performance_degraded=performance_degraded,
            recommendation=recommendation,
        )

        logger.info(
            f"Monitoring check complete | "
            f"drifted_features={len(drifted_features)}/{len(drift_scores)} | "
            f"recommendation={recommendation}"
        )

        return report

"""
demand_forecast/config.py
─────────────────────────
Loads and validates pipeline_config.yaml using Pydantic v2.

WHY PYDANTIC FOR CONFIG?
  - Catches type errors at startup, not mid-pipeline
  - Documents expected types explicitly
  - Environment variable substitution built in
  - IDE autocomplete on config fields (no more config['training']['lr'] typos)
"""

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


# ── Sub-models — each section of the YAML gets its own class ─────────────

class DataConfig(BaseModel):
    raw_data_path: str
    processed_data_path: str
    target_column: str
    date_column: str
    product_column: str
    lag_windows: list[int]
    rolling_windows: list[int]
    test_weeks: int = Field(gt=0)          # gt=0 means must be > 0
    validation_weeks: int = Field(gt=0)

    @field_validator("lag_windows", "rolling_windows")
    @classmethod
    def must_be_positive(cls, v: list[int]) -> list[int]:
        if any(w <= 0 for w in v):
            raise ValueError("All window sizes must be positive integers")
        return sorted(v)   # always sort so feature names are deterministic


class TrainingConfig(BaseModel):
    experiment_name: str
    model_type: str
    hyperparameters: dict
    primary_metric: str
    promotion_threshold: float = Field(gt=0, lt=1)


class ServingConfig(BaseModel):
    endpoint_name: str
    machine_type: str
    min_replicas: int = Field(ge=1)
    max_replicas: int = Field(ge=1)
    canary_traffic_percent: int = Field(ge=0, le=100)
    canary_bake_hours: int = Field(ge=1)


class MonitoringConfig(BaseModel):
    drift_threshold: float = Field(gt=0, lt=1)
    mae_degradation_threshold: float = Field(gt=0)
    monitoring_schedule: str
    alert_email: str


class ProjectConfig(BaseModel):
    name: str
    version: str
    gcp_project: str
    gcp_region: str
    gcs_bucket: str


class PipelineConfig(BaseModel):
    """Root config object — import this everywhere."""
    project: ProjectConfig
    data: DataConfig
    training: TrainingConfig
    serving: ServingConfig
    monitoring: MonitoringConfig


# ── Loader ───────────────────────────────────────────────────────────────────

def _substitute_env_vars(text: str) -> str:
    """Replace ${VAR_NAME} placeholders with environment variable values."""
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            # In local dev, use sensible defaults rather than crashing
            defaults = {
                "GCP_PROJECT_ID": "local-dev",
                "GCS_BUCKET": "local-dev-bucket",
                "ALERT_EMAIL": "dev@example.com",
            }
            value = defaults.get(var_name, f"MISSING_{var_name}")
        return value
    return re.sub(r"\$\{([^}]+)\}", replacer, text)


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """
    Load and validate the pipeline config.

    Usage:
        from demand_forecast.config import load_config
        cfg = load_config()
        print(cfg.training.experiment_name)
    """
    if config_path is None:
        # Default: look for configs/ relative to the project root
        project_root = Path(__file__).parents[2]   # src/demand_forecast/ → root
        config_path = project_root / "configs" / "pipeline_config.yaml"

    raw_text = Path(config_path).read_text()
    substituted = _substitute_env_vars(raw_text)
    raw_dict = yaml.safe_load(substituted)

    # Pydantic validates types and constraints — raises clear errors if wrong
    return PipelineConfig(**raw_dict)


# ── Singleton — load once, reuse everywhere ───────────────────────────────
# Import this instead of calling load_config() repeatedly
cfg = load_config()

"""
demand_forecast/pipelines/vertex_pipeline.py
─────────────────────────────────────────────
Kubeflow Pipeline (KFP v2) compiled for Vertex AI Pipelines.

KUBEFLOW vs AIRFLOW — WHY BOTH?
  Airflow orchestrates the OUTER workflow: waits for data, triggers jobs,
  sends alerts, coordinates with business systems.

  Kubeflow Pipelines orchestrates the INNER ML workflow: each ML step
  runs in its own Docker container on Kubernetes. Benefits:
    - Complete isolation: preprocessing can't affect training (different containers)
    - Resource control: give GPU to training, CPU to preprocessing
    - Caching: skip steps whose inputs haven't changed (huge time saver)
    - Native GCP integration: runs on Vertex AI with no server to manage
    - Reproducibility: the pipeline.yaml file is the executable specification

KFP v2 CORE CONCEPTS:
  @component  — decorates a function to become a pipeline step (container)
  @pipeline   — decorates a function that wires components together
  Input/Output — typed references to artifacts (Dataset, Model, Metrics)
  compile()   — converts the Python definition to a pipeline.yaml spec
  
The compiled pipeline.yaml can be submitted to Vertex AI via:
  - aiplatform.PipelineJob(...).submit()
  - The Airflow DAG above
  - The GCP Console UI
"""

from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTS — each @component becomes a separate container step
# ─────────────────────────────────────────────────────────────────────────────

@dsl.component(
    # The base image used for this step's container.
    # In production: use your own image pushed to Artifact Registry.
    base_image="python:3.11-slim",
    # Packages installed in the container at runtime.
    # For faster startup, bake them into the base_image instead.
    packages_to_install=["pandas", "numpy", "scikit-learn", "xgboost", "google-cloud-storage"],
)
def preprocess_data(
    # Input parameters
    gcs_bucket: str,
    data_path: str,
    lag_windows: list,
    rolling_windows: list,
    test_weeks: int,
    validation_weeks: int,
    # Output artifacts — KFP manages the GCS paths for these automatically
    train_dataset: Output[Dataset],   # Output[Dataset] = a GCS-backed file
    val_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    data_stats: Output[Metrics],      # Output[Metrics] = logged metadata
) -> str:
    """
    Preprocessing component — runs in its own container.

    MLOPS INSIGHT: KFP components are pure functions with typed inputs/outputs.
    KFP handles:
    - Spinning up the container
    - Passing input values
    - Creating GCS paths for output artifacts
    - Caching: if inputs haven't changed, skip this step and reuse cached output
    """
    import json
    import sys
    sys.path.insert(0, "/src")   # your installed package path

    # Simulate preprocessing (in production: loads from gcs_bucket/data_path)
    import numpy as np
    import pandas as pd

    # In real pipeline: df = pd.read_csv(f"gs://{gcs_bucket}/{data_path}")
    # For now: generate synthetic data
    rng = np.random.default_rng(42)
    n_rows = 1000
    df = pd.DataFrame({
        "price": rng.uniform(2.5, 15.0, n_rows),
        "promotion_flag": rng.integers(0, 2, n_rows),
        "store_count": rng.integers(50, 500, n_rows),
        "lag_1w": rng.lognormal(6, 0.5, n_rows),
        "lag_2w": rng.lognormal(6, 0.5, n_rows),
        "lag_4w": rng.lognormal(6, 0.5, n_rows),
        "lag_8w": rng.lognormal(6, 0.5, n_rows),
        "lag_12w": rng.lognormal(6, 0.5, n_rows),
        "rolling_mean_4w": rng.lognormal(6, 0.3, n_rows),
        "rolling_mean_8w": rng.lognormal(6, 0.3, n_rows),
        "rolling_mean_13w": rng.lognormal(6, 0.3, n_rows),
        "rolling_std_4w": rng.lognormal(4, 0.5, n_rows),
        "week_of_year": rng.integers(1, 53, n_rows),
        "month": rng.integers(1, 13, n_rows),
        "quarter": rng.integers(1, 5, n_rows),
        "year": rng.integers(2021, 2024, n_rows),
        "week_sin": np.sin(rng.uniform(0, 2*np.pi, n_rows)),
        "week_cos": np.cos(rng.uniform(0, 2*np.pi, n_rows)),
        "units_sold": rng.lognormal(6, 0.8, n_rows),
    })

    # Split
    n_train = int(n_rows * 0.7)
    n_val = int(n_rows * 0.15)

    feature_cols = [c for c in df.columns if c != "units_sold"]
    target = "units_sold"

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    # KFP manages the GCS path — just write to .path
    train_df.to_csv(train_dataset.path, index=False)
    val_df.to_csv(val_dataset.path, index=False)
    test_df.to_csv(test_dataset.path, index=False)

    # Log metadata as Metrics artifact
    data_stats.log_metric("train_rows", len(train_df))
    data_stats.log_metric("val_rows", len(val_df))
    data_stats.log_metric("test_rows", len(test_df))
    data_stats.log_metric("n_features", len(feature_cols))

    return f"Preprocessed {n_rows} rows, {len(feature_cols)} features"


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "xgboost", "mlflow"],
)
def train_model(
    train_dataset: Input[Dataset],
    val_dataset: Input[Dataset],
    mlflow_tracking_uri: str,
    experiment_name: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    # Outputs
    trained_model: Output[Model],   # Model artifact — KFP manages GCS path
    train_metrics: Output[Metrics],
) -> str:
    """
    Training component — runs in its own container with its own resources.
    In production, specify accelerator_config for GPU training.
    """
    import pandas as pd
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error
    import mlflow
    import mlflow.xgboost
    import pickle

    train_df = pd.read_csv(train_dataset.path)
    val_df = pd.read_csv(val_dataset.path)

    target = "units_sold"
    feature_cols = [c for c in train_df.columns if c != target]

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_val, y_val = val_df[feature_cols], val_df[target]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_mae = mean_absolute_error(y_val, model.predict(X_val))
        mlflow.log_metric("val_mae", val_mae)

        # Log metrics to KFP artifact (visible in Vertex AI UI)
        train_metrics.log_metric("val_mae", val_mae)
        train_metrics.log_metric("run_id", run.info.run_id)

        # Save model to KFP-managed GCS path
        import os
        os.makedirs(trained_model.path, exist_ok=True)
        model.save_model(f"{trained_model.path}/model.xgb")
        with open(f"{trained_model.path}/feature_names.txt", "w") as f:
            f.write("\n".join(feature_cols))

        return run.info.run_id


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "xgboost"],
)
def evaluate_model(
    trained_model: Input[Model],
    test_dataset: Input[Dataset],
    promotion_threshold: float,
    eval_metrics: Output[Metrics],
) -> str:
    """
    Evaluate component — runs tests and decides on promotion.

    KFP INSIGHT: The return value of a component can be used as a condition
    in the pipeline (dsl.Condition) to implement branching, just like
    BranchPythonOperator in Airflow.
    """
    import pandas as pd
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    test_df = pd.read_csv(test_dataset.path)
    target = "units_sold"
    feature_cols = [c for c in test_df.columns if c != target]

    model = XGBRegressor()
    model.load_model(f"{trained_model.path}/model.xgb")

    preds = model.predict(test_df[feature_cols])
    test_mae = mean_absolute_error(test_df[target], preds)
    test_r2 = r2_score(test_df[target], preds)

    eval_metrics.log_metric("test_mae", test_mae)
    eval_metrics.log_metric("test_r2", test_r2)

    # Promotion decision: MAE under threshold
    should_promote = test_mae < 500   # simplified threshold for demo
    eval_metrics.log_metric("should_promote", int(should_promote))

    print(f"test_mae={test_mae:.2f} | test_r2={test_r2:.3f} | promote={should_promote}")
    return "true" if should_promote else "false"


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy_model(
    trained_model: Input[Model],
    project: str,
    region: str,
    endpoint_name: str,
    machine_type: str,
    canary_traffic_percent: int,
) -> str:
    """
    Deploy the model to a Vertex AI Endpoint with canary rollout.

    MLOPS INSIGHT: Canary deployment routes a small % of traffic to the new model.
    If metrics look good after 24h bake time, expand to 100%.
    This is the safety mechanism between training and full production.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    # Upload model to Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name="demand-forecast-model",
        artifact_uri=trained_model.path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
    )

    # Get or create endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

    # Deploy with canary traffic split
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=5,
        traffic_percentage=canary_traffic_percent,   # e.g. 10% to new model
    )

    return endpoint.resource_name


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE — wires components together
# ─────────────────────────────────────────────────────────────────────────────

@dsl.pipeline(
    name="demand-forecast-training-pipeline",
    description="End-to-end MLOps pipeline: preprocess to train to evaluate to deploy",
)
def demand_forecast_pipeline(
    # Pipeline-level parameters — can be overridden at submission time
    gcs_bucket: str = "my-mlops-bucket",
    data_path: str = "data/raw/sales.csv",
    project: str = "my-gcp-project",
    region: str = "us-central1",
    experiment_name: str = "demand-forecast-prod",
    mlflow_tracking_uri: str = "https://mlflow.internal",
    # Hyperparameters — makes it easy to run multiple experiments
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    promotion_threshold: float = 0.05,
    canary_traffic_percent: int = 10,
):
    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    preprocess_task = preprocess_data(
        gcs_bucket=gcs_bucket,
        data_path=data_path,
        lag_windows=[1, 2, 4, 8, 12],
        rolling_windows=[4, 8, 13],
        test_weeks=13,
        validation_weeks=13,
    )
    # Cache this step — if data_path hasn't changed, reuse the cached output
    preprocess_task.set_caching_options(enable_caching=True)

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    train_task = train_model(
        train_dataset=preprocess_task.outputs["train_dataset"],
        val_dataset=preprocess_task.outputs["val_dataset"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    # Request GPU for training (comment out if not needed)
    # train_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
    train_task.set_cpu_limit("4").set_memory_limit("16G")

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    eval_task = evaluate_model(
        trained_model=train_task.outputs["trained_model"],
        test_dataset=preprocess_task.outputs["test_dataset"],
        promotion_threshold=promotion_threshold,
    )

    # ── Step 4: Conditional deployment (if evaluation passes) ─────────────────
    # dsl.Condition = only execute these tasks if the condition is met.
    # This is the KFP equivalent of Airflow's BranchPythonOperator.
    with dsl.If(eval_task.outputs["Output"] == "true", name="should-promote"):
        deploy_task = deploy_model(
            trained_model=train_task.outputs["trained_model"],
            project=project,
            region=region,
            endpoint_name="demand-forecast-endpoint",
            machine_type="n1-standard-2",
            canary_traffic_percent=canary_traffic_percent,
        )


# ── Compile pipeline to YAML ──────────────────────────────────────────────────

if __name__ == "__main__":
    output_path = Path(__file__).parent / "demand_forecast_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=demand_forecast_pipeline,
        package_path=str(output_path),
    )
    print(f"Pipeline compiled to: {output_path}")
    print("Submit to Vertex AI with:")
    print("""
    from google.cloud import aiplatform
    aiplatform.init(project='my-project', location='us-central1')
    job = aiplatform.PipelineJob(
        display_name='demand-forecast-run',
        template_path='demand_forecast_pipeline.yaml',
    )
    job.submit()
    """)

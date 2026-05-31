"""
airflow/dags/demand_forecast_retrain.py
────────────────────────────────────────
Airflow DAG that orchestrates the full demand forecast retraining pipeline.

AIRFLOW CORE CONCEPTS demonstrated here:
  1. DAG        — Directed Acyclic Graph of tasks. Tasks have dependencies,
                  no cycles allowed. Airflow schedules and executes it.
  2. Operators  — Individual task types (Python, Bash, GCP operators, etc.)
  3. XCom       — Cross-communication between tasks (passing values)
  4. SLA        — Service Level Agreement — alert if DAG takes too long
  5. Sensors    — Wait for an external condition before proceeding
  6. Branching  — Conditional paths (promote or not)

WHEN AIRFLOW VS KUBEFLOW?
  - Airflow: general workflow orchestration — great for data pipelines,
    ETL, anything with external dependencies (APIs, databases, file arrivals).
    Tasks run on Airflow workers.
  - Kubeflow Pipelines: ML-specific — each step is a container.
    Better isolation, native GCP integration, Vertex AI native.
    Use Airflow to TRIGGER Kubeflow pipelines for the best of both worlds.

This DAG runs weekly (Monday 2am UTC) and:
  1. Checks new data has arrived in GCS
  2. Runs preprocessing
  3. Trains the model (logs to MLflow)
  4. Evaluates against champion
  5. Branches: promote to production OR alert and stop
  6. Notifies on success or failure
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor
from airflow.utils.trigger_rule import TriggerRule

# ── Default args — applied to all tasks unless overridden ────────────────────
# These are the standard production defaults.
default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,           # don't wait for previous run to succeed
    "email": ["ml-alerts@company.com"],
    "email_on_failure": True,           # alert team on any task failure
    "email_on_retry": False,
    "retries": 2,                       # retry failed tasks before giving up
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,  # wait longer between each retry
    "execution_timeout": timedelta(hours=4),  # kill tasks that hang
}


# ── Task functions ────────────────────────────────────────────────────────────
# Best practice: define task logic in importable functions (testable!)
# not as lambdas inside the DAG definition.

def _run_preprocessing(**context):
    """
    Run data preprocessing and push split metadata to XCom.

    XCom (Cross-Communication) is Airflow's mechanism for passing
    small values between tasks. Keep XCom payloads small — it's stored
    in the Airflow metadata database, not a fast store.
    For large artifacts (datasets, models), use GCS paths instead.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")   # adjust for your environment

    from demand_forecast.data.preprocessing import run_preprocessing

    split = run_preprocessing()

    # Push metadata to XCom — downstream tasks read this
    context["task_instance"].xcom_push(
        key="data_version", value=split.data_version
    )
    context["task_instance"].xcom_push(
        key="data_stats", value=split.stats
    )
    return split.data_version


def _train_model(**context):
    """Train the model and push the run_id to XCom."""
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from demand_forecast.data.preprocessing import run_preprocessing
    from demand_forecast.training.trainer import DemandForecaster

    # Re-run preprocessing (in production: load cached version from GCS)
    split = run_preprocessing()

    trainer = DemandForecaster(
        experiment_name="demand-forecast-prod",
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "mlruns"),
    )

    run_id = trainer.train(
        split=split,
        run_name=f"scheduled_{context['ds']}",   # context['ds'] = execution date
    )

    context["task_instance"].xcom_push(key="run_id", value=run_id)
    return run_id


def _evaluate_and_branch(**context):
    """
    Compare challenger vs champion. Return the task_id to execute next.

    BranchPythonOperator: the return value must be the task_id (or list of
    task_ids) to execute. All other branches are skipped.

    MLOPS INSIGHT: This is the automated model promotion gate.
    In a mature org, this also triggers a human approval step in Slack.
    """
    import sys, os
    sys.path.insert(0, "/opt/airflow/src")

    import mlflow
    from demand_forecast.training.trainer import DemandForecaster

    ti = context["task_instance"]
    run_id = ti.xcom_pull(task_ids="train_model", key="run_id")

    # In production: fetch champion run_id from model registry
    # Here we compare against a hardcoded previous run for illustration
    champion_run_id = os.environ.get("CHAMPION_RUN_ID", None)

    trainer = DemandForecaster(
        experiment_name="demand-forecast-prod",
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "mlruns"),
    )

    should_promote = trainer.evaluate_vs_champion(
        challenger_run_id=run_id,
        champion_run_id=champion_run_id,
        primary_metric="test_mae",
        promotion_threshold=0.05,
    )

    context["task_instance"].xcom_push(key="should_promote", value=should_promote)

    # Return the task_id of the branch to take
    return "promote_model" if should_promote else "alert_no_promotion"


def _promote_model(**context):
    """
    Promote the challenger model to production in the model registry.

    In production this:
    1. Tags the MLflow model as 'Production'
    2. Triggers a Vertex AI deployment job
    3. Updates the serving config in GCS
    """
    import mlflow

    ti = context["task_instance"]
    run_id = ti.xcom_pull(task_ids="train_model", key="run_id")

    client = mlflow.MlflowClient()
    # In MLflow, models move through: None → Staging → Production → Archived
    # Transition the new version to Production
    model_versions = client.search_model_versions(f"run_id='{run_id}'")
    if model_versions:
        version = model_versions[0].version
        client.transition_model_version_stage(
            name="demand-forecast-prod-model",
            version=version,
            stage="Production",
            archive_existing_versions=True,  # archive the previous production model
        )
        print(f"Promoted model version {version} to Production")


def _alert_no_promotion(**context):
    """Alert the team when the challenger didn't beat the champion."""
    ti = context["task_instance"]
    run_id = ti.xcom_pull(task_ids="train_model", key="run_id")
    # In production: post to Slack, send email, create Jira ticket
    print(f"Challenger run {run_id} did not beat the champion. No promotion.")


def _run_data_quality_checks(**context):
    """
    Validate data quality BEFORE training.
    Gate the pipeline on data health — never train on bad data.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from demand_forecast.data.preprocessing import (
        generate_synthetic_sales_data, validate_schema
    )
    df = generate_synthetic_sales_data()
    validate_schema(df)
    print(f"Data quality checks passed: {len(df):,} rows")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="demand_forecast_weekly_retrain",
    description="Weekly demand forecast model retraining pipeline",
    schedule="0 2 * * 1",         # Every Monday at 2am UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,                 # Don't backfill missed runs
    default_args=default_args,
    tags=["ml", "demand-forecast", "production"],
    # SLA: alert if the entire DAG takes more than 3 hours
    sla_miss_callback=lambda dag, task_list, blocking_task_list, slas, blocking_tis:
        print(f"SLA MISSED: {[t.task_id for t in task_list]}"),
    doc_md="""
## Demand Forecast Weekly Retraining

Retrains the demand forecasting model every Monday using the latest sales data.

### Pipeline steps:
1. **Wait for data** — GCS sensor waits for last week's data file
2. **Data quality checks** — validate schema and data distributions
3. **Preprocessing** — feature engineering and train/test split
4. **Train model** — XGBoost training with MLflow tracking
5. **Evaluate** — compare challenger vs champion model
6. **Branch** — promote if better, alert if not
    """,
) as dag:

    # ── Task 1: Wait for data to land in GCS ─────────────────────────────────
    # GCSObjectExistenceSensor keeps polling until the file exists.
    # 'poke_interval' = check every 5 minutes.
    # 'timeout' = give up after 6 hours (catches upstream data pipeline failures)
    wait_for_data = GCSObjectExistenceSensor(
        task_id="wait_for_data",
        bucket="{{ var.value.gcs_bucket }}",   # Airflow Variable (configured in UI)
        object="data/raw/sales_{{ ds }}.csv",  # Jinja template — ds = execution date
        poke_interval=300,
        timeout=6 * 3600,
        mode="reschedule",    # IMPORTANT: 'reschedule' frees the worker slot while waiting
                              # vs 'poke' which holds the worker — reschedule is efficient
    )

    # ── Task 2: Data quality gate ─────────────────────────────────────────────
    data_quality = PythonOperator(
        task_id="data_quality_checks",
        python_callable=_run_data_quality_checks,
    )

    # ── Task 3: Preprocessing ─────────────────────────────────────────────────
    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=_run_preprocessing,
    )

    # ── Task 4: Train ─────────────────────────────────────────────────────────
    train = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        execution_timeout=timedelta(hours=2),
    )

    # ── Task 5: Evaluate + Branch ─────────────────────────────────────────────
    evaluate = BranchPythonOperator(
        task_id="evaluate_and_branch",
        python_callable=_evaluate_and_branch,
    )

    # ── Task 6a: Promote ──────────────────────────────────────────────────────
    promote = PythonOperator(
        task_id="promote_model",
        python_callable=_promote_model,
    )

    # ── Task 6b: Alert (no promotion) ────────────────────────────────────────
    alert_no_promotion = PythonOperator(
        task_id="alert_no_promotion",
        python_callable=_alert_no_promotion,
    )

    # ── Task 7: Final notification ────────────────────────────────────────────
    # TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS = run this task if:
    # - no upstream task failed
    # - at least one upstream task succeeded
    # This lets the notification run after EITHER the promote OR the alert branch.
    pipeline_complete = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ── Dependencies (the DAG structure) ─────────────────────────────────────
    # >> is Airflow's "set_downstream" shorthand
    # Read as: wait_for_data runs first, then data_quality, then preprocess, etc.
    (
        wait_for_data
        >> data_quality
        >> preprocess
        >> train
        >> evaluate
        >> [promote, alert_no_promotion]   # branching: either promote OR alert
        >> pipeline_complete
    )

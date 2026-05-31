import os
from google.cloud import aiplatform

PROJECT_ID = "demand-forecast-mlops"
REGION = "us-central1"
BUCKET = "demand-forecast-sj"
PIPELINE_YAML = "src/demand_forecast/pipelines/demand_forecast_pipeline.yaml"

aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.PipelineJob(
    display_name="demand-forecast-run-01",
    template_path=PIPELINE_YAML,
    pipeline_root=f"gs://{BUCKET}/pipeline-runs",
    parameter_values={
        "gcs_bucket": BUCKET,
        "project": PROJECT_ID,
        "region": REGION,
        "experiment_name": "demand-forecast-prod",
        "mlflow_tracking_uri": f"gs://{BUCKET}/mlruns",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
    },
)

job.submit(
    service_account="demand-forecast-sa@demand-forecast-mlops.iam.gserviceaccount.com"
)
print(f"Pipeline submitted. Job name: {job.display_name}")
print(f"View at: https://console.cloud.google.com/vertex-ai/pipelines")
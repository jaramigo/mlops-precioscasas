import time
from datetime import datetime
from typing import NamedTuple

import kfp
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component, pipeline)
from kfp.v2.google.client import AIPlatformClient

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip

# --------


# ---------

# preparar data
# entrenamiento
# despliegue

# ---------

PROJECT_ID = 'mds-mlops2024'
PIPELINE_ROOT = 'gs://mlops2024-bucket/pipeline_root_precios/'
DATASET_ID = "houses"  # The Data Set ID where the view sits
VIEW_NAME = "house_data"  # BigQuery view you create for input data



@component(
    packages_to_install=["google-cloud-bigquery[pandas]==3.10.0"],
)
def export_dataset(
    project_id: str,
    dataset_id: str,
    view_name: str,
    dataset: Output[Dataset],
):
    """Exports from BigQuery to a CSV file.

    Args:
        project_id: The Project ID.
        dataset_id: The BigQuery Dataset ID. Must be pre-created in the project.
        view_name: The BigQuery view name.

    Returns:
        dataset: The Dataset artifact with exported CSV file.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    table_name = f"{project_id}.{dataset_id}.{view_name}"
    query = """
    SELECT
      *
    FROM
      `{table_name}`
    LIMIT 100
    """.format(
        table_name=table_name
    )

    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=query, job_config=job_config)
    df = query_job.result().to_dataframe()
    df.to_csv(dataset.path, index=False)



    

@component(
    packages_to_install=[
        "xgboost==1.6.2",
        "pandas==1.3.5",
        "joblib==1.1.0",
        "scikit-learn==1.0.2",
        "gcsfs==2021.11.1"
    ],
)
def randomforest_training(
    dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Trains a RF model.

    Args:
        dataset: The training dataset.

    Returns:
        model: The model artifact stores the model.joblib file.
        metrics: The metrics of the trained model.
    """
    import os
    import joblib
    from sklearn.metrics import (accuracy_score, precision_recall_curve,
                                 roc_auc_score)
    from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                         train_test_split)
    from sklearn.preprocessing import LabelEncoder

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    
    # Load the training dataset
    with open(dataset.path, "r") as train_data:
        housing = pd.read_csv(train_data)


    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    housing = housing.select_dtypes(include=numerics)
    housing.drop(['Longitude','Latitude'], axis = 'columns', inplace=True)

    housing_X = housing.drop(['Sale_Price'],axis = 'columns')
    housing_y = housing.Sale_Price

    train_reg_X, test_reg_X, train_reg_y, test_reg_y = train_test_split(housing_X,housing_y, test_size=0.2)

    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(train_reg_X,train_reg_y)
    pred_values_rf_reg = rf_regressor.predict(test_reg_X)

    rf_regressor.score(test_reg_X,test_reg_y)

    #metrics.log_metric("accuracy", (score * 100.0))
    #metrics.log_metric("framework", "xgboost")
    #metrics.log_metric("dataset_size", len(raw_data))
    #metrics.log_metric("AUC", auc)

    # Export the model to a file
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(rf_regressor, os.path.join(model.path, "model.joblib"))


@component(
    packages_to_install=["google-cloud-aiplatform==1.25.0"],
)
def deploy_randomforest_model(
    model: Input[Model],
    project_id: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    """Deploys model to Vertex AI Endpoint.

    Args:
        model: The model to deploy.
        project_id: The project ID of the Vertex AI Endpoint.

    Returns:
        vertex_endpoint: The deployed Vertex AI Endpoint.
        vertex_model: The deployed Vertex AI Model.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id)

    deployed_model = aiplatform.Model.upload(
        display_name="precio-casas-model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    )
#   bajamos los requerimientos de máquina
#   endpoint = deployed_model.deploy(machine_type="n1-standard-4")
    endpoint = deployed_model.deploy(machine_type="e2-standard-2")

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name

@dsl.pipeline(
    name="precioscasas-pipeline",
)
def pipeline():
    """A demo pipeline."""


    export_dataset_task = (
        export_dataset(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            view_name=VIEW_NAME,
        )
        .set_caching_options(False)
    )

    training_task = randomforest_training(
        dataset=export_dataset_task.outputs["dataset"],
    )

    _ = deploy_randomforest_model(
        project_id=PROJECT_ID,
        model=training_task.outputs["model"],
    )

if __name__ == '__main__':
    
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="tab_pipeline.json"
    )
    print('Pipeline compilado exitosamente')

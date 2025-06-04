import argparse
import os
import time
import wandb

from datetime import datetime
from dotenv import load_dotenv
from sagemaker.huggingface.model import HuggingFaceModel

from utils import load_yaml_config

load_dotenv()

MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")
SAGEMAKER_DOMAIN_ID = os.environ.get("SAGEMAKER_DOMAIN_ID")
SAGEMAKER_PROJECT_ID = os.environ.get("SAGEMAKER_PROJECT_ID")


def get_staged_model_path():
    """Retrieve the path to the staged model from the W&B artifact registry.
    
    Returns:
        str: Path to the staged model.tar.gz on S3.
    """
    api = wandb.Api()

    try:
        model = api.artifact(type="model", name=f"{MODEL_REGISTRY}:staged")
    except:
        raise ValueError(f"Model artifact {MODEL_REGISTRY}:staged not found.")
    
    staged_model_path = model.metadata.get("s3_path")
    if not staged_model_path:
        raise ValueError("Path to staged model not found.")
    
    return staged_model_path


def create_sagemaker_model(model_path, **model_kwargs):
    """Create a SageMaker model from the specified model path.
    
    Args:
        model_path (str): Path to the model tarball on S3.
        **model_kwargs: Additional keyword arguments for the HuggingFaceModel.

    Returns:
        HuggingFaceModel: A SageMaker HuggingFaceModel instance.
    """
    sm_model = HuggingFaceModel(
        model_data=model_path,
        role=SAGEMAKER_ROLE_ARN,
        **model_kwargs,
    )

    return sm_model


def deploy_model(sm_model, **endpoint_kwargs):
    """Deploy the SageMaker model to an endpoint.
    
    Args:
        sm_model (HuggingFaceModel): The SageMaker model to deploy.
        **endpoint_kwargs: Additional keyword arguments for the Sagemaker endpoint.
    
    Returns:
        Predictor: A SageMaker Predictor instance for the deployed endpoint.
    """
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    endpoint_name = f"hrcstagger-{now}"

    predictor = sm_model.deploy(
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        tags=[
            {"Key": "AmazonDataZoneProject", "Value": SAGEMAKER_PROJECT_ID},
            {"Key": "AmazonDataZoneDomain", "Value": SAGEMAKER_DOMAIN_ID},
        ],
        **endpoint_kwargs,
    )
    wandb.log({"endpoint_name": endpoint_name})

    return predictor


def test_endpoint(predictor):
    """Test the SageMaker endpoint with a sample input. Inference time and result are logged to W&B.
    
    Args:
        predictor (Predictor): The SageMaker Predictor instance for the endpoint.
    """
    test_data = {
        "inputs": "Alzheimer's disease poses a major challenge in healthcare, impacting millions globally. "
                        "Despite extensive research, the genetic causes are not fully understood. This project aims to "
                        "uncover the genetic factors of AD through innovative genomic analysis and integrative techniques. "
                        "Using advanced sequencing and computational methods, we will study genetic variants, gene interactions, "
                        "and epigenetic changes linked to AD, leveraging large-scale genomic data from diverse populations to ensure "
                        "broader relevance. Advanced statistical models will help identify potential biomarkers for predicting disease "
                        "onset and progression, possibly leading to new therapeutic targets and personalised medicine approaches. "
                        "Our collaboration with experts in genomics, neurology, and bioinformatics strives to foster innovation and "
                        "accelerate Alzheimer's research, aiming to enhance understanding and treatment options concerning the genetic "
                        "influences on Alzheimer's disease."
        }
    
    start = time.perf_counter()
    result = predictor.predict(test_data)
    end = time.perf_counter()
    inference_time = end - start

    print(f"Inference result: {result}")
    print(f"Inference time: {inference_time:.2f} seconds")
    wandb.log({"inference_result": result, "inference_time_sec": inference_time})


def tag_artifact():
    """Tag the staged model artifact for production use in the W&B registry."""
    artifact = wandb.use_artifact(f"{MODEL_REGISTRY}:staged", type="model")
    artifact.metadata["deployment_config"] = config
    artifact.aliases.remove("staged")
    artifact.aliases.append("prod")


def delete_endpoint(predictor):
    """Delete the SageMaker endpoint.
    
    Args:
        predictor (Predictor): The SageMaker Predictor instance for the endpoint.
    """
    predictor.delete_endpoint()
    print("Endpoint deleted.")


def deploy(config):
    """Deploy the HRCSTagger model to SageMaker using the provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing model and endpoint parameters.
    """
    model_path = get_staged_model_path()
    with wandb.init(project="grant_hrcs_tagger", job_type="staging") as run:
        wandb.log(config)

        print(f"Creating SageMaker model with path: {model_path}")
        sm_model = create_sagemaker_model(model_path=model_path,
                                        **config["model_args"])
        
        print("Deploying model to SageMaker endpoint...")
        predictor = deploy_model(sm_model, **config["endpoint_args"])
        run.alert(f"HRCSTagger endpoint created successfully: {predictor.endpoint_name}")
        
        test_endpoint(predictor)
        proceed = input("Do you want to proceed and tag this model for use in production? (Y/N): ").strip().lower()
        wandb.log({"proceed": proceed})
        if proceed == "y":
            tag_artifact()

        delete_endpoint(predictor)
        run.alert("HRCSTagger endpoint deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy HRCSTagger model to SageMaker")
    parser.add_argument("--config", type=str, help="Path to the deployment configuration file")

    args = parser.parse_args()
    
    config = load_yaml_config(args.config)

    deploy(config)

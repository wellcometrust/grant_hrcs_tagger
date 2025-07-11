import argparse
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from sagemaker.huggingface.model import HuggingFaceModel

import wandb
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
    except Exception as e:
        raise ValueError(f"Model artifact {MODEL_REGISTRY}:staged not found.") from e

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

    start = time.perf_counter()
    predictor = sm_model.deploy(
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        tags=[
            {"Key": "AmazonDataZoneProject", "Value": SAGEMAKER_PROJECT_ID},
            {"Key": "AmazonDataZoneDomain", "Value": SAGEMAKER_DOMAIN_ID},
        ],
        **endpoint_kwargs,
    )
    end = time.perf_counter()
    deploy_time = end - start

    print(f"Endpoint creation time: {deploy_time:.2f} seconds")
    wandb.log({"endpoint_name": endpoint_name, "deploy_time_sec": deploy_time})

    return predictor


def test_endpoint(predictor):
    """Test the SageMaker endpoint with a sample input. Inference time and result are logged to W&B.

    Args:
        predictor (Predictor): The SageMaker Predictor instance for the endpoint.
    """
    test_data = {
        "inputs": """
        Advancements in biomedical research hold the promise of profoundly improving health outcomes through 
        innovative approaches to diagnosis, treatment, and prevention of diseases. This grant proposes a comprehensive 
        study aimed at understanding the molecular underpinnings of AD to develop targeted therapies that 
        enhance efficacy while minimising side effects. Despite significant strides in medical science, AD
        remains a leading cause of morbidity and mortality worldwide, necessitating focused efforts to decipher its complex 
        biological mechanisms. The proposed research will employ cutting-edge techniques such as CRISPR-Cas9 gene editing, 
        high-throughput sequencing, and advanced bioinformatics analysis to map genetic and molecular pathways 
        involved in disease progression. By integrating data from genomics, proteomics, and metabolomics, we aim to identify 
        novel biomarkers for early detection and delineate key targets for therapeutic intervention.
        Our multidisciplinary team is uniquely positioned to tackle these challenges, combining expertise in molecular biology, 
        clinical medicine, and computational science. The project will not only elucidate the pathophysiology 
        of AD but will also foster translational research efforts to bridge laboratory findings with clinical applications. 
        Collaborations with leading research institutes and industry partners will ensure rigorous validation and 
        potential scalability of our findings. Ultimately, this grant seeks to contribute to personalised medicine approaches, 
        offering patients tailored treatments based on their genetic profiles, thus improving clinical outcomes and 
        quality of life. Furthermore, by advancing our understanding of AD, we aim to set a foundation for future research 
        initiatives exploring similar complex disorders. Through this endeavour, we aspire to push the boundaries of 
        medical research, paving the way for innovative solutions that address one of today’s most pressing health challenges.
        """
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
    artifact.save()


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
        sm_model = create_sagemaker_model(model_path=model_path, **config["model_args"])

        print("Deploying model to SageMaker endpoint...")
        predictor = deploy_model(sm_model, **config["endpoint_args"])
        run.alert(
            title="HRCSTagger Endpoint",
            text=f"HRCSTagger endpoint created successfully: {predictor.endpoint_name}",
        )

        test_endpoint(predictor)
        proceed = (
            input(
                "Do you want to proceed and tag this model for use in production? (Y/N): "
            )
            .strip()
            .lower()
        )
        wandb.log({"proceed": proceed})
        if proceed == "y":
            tag_artifact()

        delete_endpoint(predictor)
        run.alert(title="HRCSTagger Endpoint", text="HRCSTagger endpoint deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy HRCSTagger model to SageMaker")
    parser.add_argument(
        "--config", type=str, help="Path to the deployment configuration file"
    )

    args = parser.parse_args()

    config = load_yaml_config(args.config)

    deploy(config)

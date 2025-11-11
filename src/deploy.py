import argparse
import os
import sys
import time
from datetime import datetime

import torch
import transformers
from dotenv import load_dotenv
from sagemaker import image_uris
from sagemaker.huggingface.model import HuggingFaceModel

import wandb

load_dotenv()

MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY")


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


def get_sagemaker_image_uri(instance_type):
    """Get the SageMaker container URI for the specified instance type.

    Args:
        instance_type (str): The type of SageMaker instance.

    Returns:
        str: The container image URI for the specified instance type.
    """
    transformers_version = transformers.__version__
    torch_version = torch.__version__.split('+')[0]
    python_version = sys.version.split()[0]

    image_uri = image_uris.retrieve(
        framework="huggingface",
        region="eu-west-1",
        version=transformers_version,
        image_scope="inference",
        base_framework_version=f"pytorch{torch_version}",
        instance_type=instance_type,
        py_version=f"py{''.join(python_version.split('.')[:2])}",
    )

    return image_uri


def create_sagemaker_model(model_path, sagemaker_image_uri):
    """Create a SageMaker model from the specified model path.

    Args:
        model_path (str): Path to the model tarball on S3.
        sagemaker_image_uri (str): The SageMaker container image URI.

    Returns:
        HuggingFaceModel: A SageMaker HuggingFaceModel instance.
    """
    sm_model = HuggingFaceModel(
        model_data=model_path,
        image_uri=sagemaker_image_uri,
    )

    return sm_model


def deploy_model(sm_model, instance_type="ml.m5.xlarge"):
    """Deploy the SageMaker model to an endpoint.

    Args:
        sm_model (HuggingFaceModel): The SageMaker model to deploy.
        instance_type (str): The type of SageMaker instance to use for deployment.

    Returns:
        Predictor: A SageMaker Predictor instance for the deployed endpoint.
    """
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    endpoint_name = f"hrcstagger-{now}"

    start = time.perf_counter()
    predictor = sm_model.deploy(
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        instance_type=instance_type,
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


def tag_artifact(instance_type, sagemaker_image_uri):
    """Tag the staged model artifact for production use in the W&B registry."""
    artifact = wandb.use_artifact(f"{MODEL_REGISTRY}:staged", type="model")
    artifact.metadata["instance_type"] = instance_type
    artifact.metadata["sagemaker_image_uri"] = sagemaker_image_uri
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


def deploy(instance_type="ml.m5.xlarge"):
    """Deploy the HRCSTagger model to SageMaker using the provided configuration.

    Args:
        instance_type (str): The type of SageMaker instance to use for deployment.
    """
    model_path = get_staged_model_path()
    with wandb.init(project="grant_hrcs_tagger_stage_and_deploy", job_type="staging") as run:
        print(f"Creating SageMaker model with path: {model_path}")
        sagemaker_image_uri = get_sagemaker_image_uri(instance_type)
        sm_model = create_sagemaker_model(
            model_path=model_path, sagemaker_image_uri=sagemaker_image_uri
        )

        print("Deploying model to SageMaker endpoint...")
        predictor = deploy_model(sm_model, instance_type=instance_type)
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
            tag_artifact(
                instance_type=instance_type, sagemaker_image_uri=sagemaker_image_uri
            )

        delete_endpoint(predictor)
        run.alert(title="HRCSTagger Endpoint", text="HRCSTagger endpoint deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy HRCSTagger model to SageMaker")
    parser.add_argument(
        "--instance_type",
        type=str,
        help="Type of SageMaker instance to use",
        default="ml.m5.xlarge",
    )
    parser.add_argument(
        "--sagemaker_config",
        type=str,
        help="(Optional) path to the SageMaker configuration file. Can be a local file or an S3 URI.",
    )

    args = parser.parse_args()

    if isinstance(args.sagemaker_config, str):
        os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = args.sagemaker_config

    deploy(args.instance_type)

import argparse
import os
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline

import wandb

load_dotenv()

MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY")
SAGEMAKER_BUCKET = os.environ.get("SAGEMAKER_BUCKET")


def link_to_registry(model_name):
    """Link the model to the W&B registry.

    Args:
        model_name (str): Name of the model to link to the W&B registry.
    """
    with wandb.init(project="HRCSTagger", job_type="model_linking") as run:
        artifact = wandb.use_artifact("model_name", type="model")
        run.link_artifact(artifact, target_path=MODEL_REGISTRY)
        run.alert(
            title="Model change",
            text=f"Model {model_name} linked to registry at {MODEL_REGISTRY}.",
        )


def add_tokenizer(artifact_dir):
    """Add tokenizer to the artifact directory.

    Args:
        artifact_dir (str): Path to the artifact directory.
    """
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    tokenizer.save_pretrained(artifact_dir)


def test_inference(model_dir):
    """Test inference on the huggingface model.

    Args:
        model_dir (str): Path to the model directory.
    """
    model = pipeline("text-classification", model=model_dir)
    test_text = "This is a test sentence for HRCSTagger."
    result = model(test_text)
    print(f"Inference result: {result}")
    wandb.log({"inference_result": result})


def _create_tarball(artifact_dir, output_path):
    """Create a tarball of the artifact directory."""
    with tarfile.open(output_path, "w:gz") as tar:
        for root, _, files in os.walk(artifact_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, artifact_dir)
                tar.add(file_path, arcname=arcname)
    wandb.log({"model.tar.gz": f"{', '.join(os.listdir(artifact_dir))}"})


def upload_to_s3(model_path):
    """Upload the model tarball to S3.

    Args:
        model_path (str): Path to the model tarball.

    Returns:
        str: S3 path of the uploaded model tarball.
    """
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    full_path = Path(SAGEMAKER_BUCKET) / "hrcs_tagger" / now / "model.tar.gz"
    s3_bucket = full_path.parts[0]

    print("Uploading model to S3...")
    s3_client = boto3.client("s3")
    s3_client.upload_file(model_path, s3_bucket, str(full_path.relative_to(s3_bucket)))
    s3_client.close()

    return f"s3://{full_path}"


def package_model(root_dir, artifact_dir):
    """Package the model and upload it to S3.

    Args:
        root_dir (str): Root directory where the model will be packaged.
        artifact_dir (str): Directory containing the model artifacts.

    Returns:
        str: S3 path of the uploaded model tarball.
    """
    model_path = os.path.join(root_dir, "model.tar.gz")
    _create_tarball(artifact_dir, model_path)
    s3_path = upload_to_s3(model_path)

    print(f"Model packaged and uploaded to S3 at {s3_path}.")
    wandb.log({"s3_path": s3_path})

    return s3_path


def stage_model(model_name):
    """Stage the model for deployment.

    Args:
        model_name (str): Name of the model to stage. If not provided, the latest model will be used.
    """
    if not model_name:
        model_name = f"{MODEL_REGISTRY}:latest"

    with wandb.init(project="grant_hrcs_tagger", job_type="staging") as run:
        artifact = wandb.use_artifact(model_name, type="model")
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = artifact.download(tmpdir)
            add_tokenizer(artifact_dir)
            test_inference(artifact_dir)
            proceed = (
                input("Do you want to proceed and add the model to S3? (Y/N): ")
                .strip()
                .lower()
            )
            wandb.log({"proceed": proceed})
            if proceed == "y":
                s3_path = package_model(tmpdir, artifact_dir)
                artifact.metadata["s3_path"] = s3_path
                artifact.aliases.append("staged")
                artifact.save()
                run.alert(
                    title="Model change",
                    text=f"HRCSTagger model staged successfully at {s3_path}.",
                )


def setup_parser():
    """Setup the command line argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="HRCSTagger model CLI")
    subparsers = parser.add_subparsers(dest="command")

    linking_parser = subparsers.add_parser(
        "link", help="Link the model to the registry"
    )
    linking_parser.add_argument(
        "--model_name", type=str, help="Name of the model to link to the W&B registry."
    )

    staging_parser = subparsers.add_parser(
        "stage", help="Stage the model for deployment"
    )
    staging_parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model to stage. If not provided, the latest model will be used.",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.command == "link":
        link_to_registry(args.model_name)

    if args.command == "stage":
        stage_model(args.model_name)

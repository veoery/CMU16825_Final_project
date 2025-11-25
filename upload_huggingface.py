import os
from pathlib import Path
from huggingface_hub import HfApi

# Set your Hugging Face token as an environment variable or use hf_token parameter
# You can get your token from: https://huggingface.co/settings/tokens
api = HfApi(token=os.getenv("HF_TOKEN"))

# Path to your subset dataset
dataset_path = Path("data/Omni-CAD-subset")

if not dataset_path.exists():
    raise ValueError(f"Dataset path does not exist: {dataset_path}")

print(f"Uploading dataset from: {dataset_path.absolute()}")
print(f"To repository: chentianle11171/Omni-CAD_sub10")

api.upload_folder(
    folder_path=str(dataset_path.absolute()),
    repo_id="chentianle11171/Omni-CAD_sub10",
    repo_type="dataset",
)

print("Upload completed!")
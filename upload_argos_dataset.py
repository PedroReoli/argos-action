"""
Argos-Action-1.5B: Hugging Face Dataset Publisher
Repository: pedroreoli/argos-computer-use-mcp
Author: Pedro Lucas Reis (@pedroreoli)
License: Apache-2.0
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "pedroreoli/argos-computer-use-mcp"
DATA_DIR = Path(__file__).resolve().parent / "data"

def main():
    print(f"[ARGOS] Publishing dataset to Hugging Face Hub: {REPO_ID}...")
    api = HfApi()
    
    # 1. Valida repositório
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=False
    )
    print("[ARGOS] Repository validated on Hugging Face!")
    
    # 2. Upload de assets
    readme_path = DATA_DIR / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"Missing {readme_path}")
        
    print("[ARGOS] Uploading dataset files (train.jsonl, test.jsonl, README.md, metadata)...")
    api.upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="feat(dataset): update configs YAML, accurate token census and metadata isolation"
    )
    
    print("\n[OK] ARGOS DATASET SUCCESSFULLY PUBLISHED TO HUGGING FACE!")
    print(f"URL: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()

"""
Argos-Action v2.0: Hugging Face Hub Publisher (Dataset, Space & Model Card)
Author: Pedro Lucas Reis (@pedroreoli)
License: Apache-2.0
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

DATASET_REPO_ID = "pedroreoli/argos-computer-use-mcp"
SPACE_REPO_ID = "pedroreoli/argos-action-demo"
MODEL_REPO_ID = "pedroreoli/argos-action-1.5b"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
SPACE_DIR = ROOT_DIR / "space"

MODEL_CARD_YAML_HEADER = """---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
- agents
- computer-use
- model-context-protocol
- mcp
- accessibility-tree
- axtree
- uia
- tool-calling
- small-language-models
- slm
- lora
- qwen2.5
datasets:
- pedroreoli/argos-computer-use-mcp
---

"""


def sync_all(token: str = None):
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))

    # 1. Sync Dataset v2.0
    print(f"[ARGOS v2.0] Syncing Dataset to {DATASET_REPO_ID}...")
    create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", exist_ok=True, private=False, token=api.token)
    api.upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message="feat(dataset): release Argos v2.0 (2,000 100%-unique rows, all 6 tools balanced, 0% leakage, updated token census)",
    )
    print(f"[OK] Dataset synced: https://huggingface.co/datasets/{DATASET_REPO_ID}")

    # 2. Sync Static Space v2.0
    print(f"[ARGOS v2.0] Syncing Space to {SPACE_REPO_ID}...")
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        delete_patterns=["app.py", "requirements.txt"],
        commit_message="feat(space): align README and HTML as Static MCP Protocol & Guardrail Playground v2.0",
    )
    print(f"[OK] Space synced: https://huggingface.co/spaces/{SPACE_REPO_ID}")

    # 3. Sync Model Card v2.0 & remove post-hoc trainer_state.json
    print(f"[ARGOS v2.0] Syncing Model Card to {MODEL_REPO_ID}...")
    root_readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    model_card_content = MODEL_CARD_YAML_HEADER + root_readme
    api.upload_file(
        path_or_fileobj=model_card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="docs(model-card): document v1 holdout_v2 post-mortem results & v2.0 Defense-in-Depth MCP architecture",
    )
    try:
        api.delete_file(
            path_in_repo="trainer_state.json",
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            commit_message="chore(audit): remove post-hoc reconstructed trainer_state.json per post-mortem",
        )
        print("[OK] Removed post-hoc trainer_state.json from Model Hub.")
    except Exception:
        pass
    print(f"[OK] Model Card synced: https://huggingface.co/{MODEL_REPO_ID}")


if __name__ == "__main__":
    sync_all()

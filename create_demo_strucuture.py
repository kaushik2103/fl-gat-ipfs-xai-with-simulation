#!/usr/bin/env python3
# ============================================================
# CREATE DEMO UI STRUCTURE FOR FL-GAT-IPFS PROJECT
# ============================================================

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEMO_DIR = BASE_DIR / "demo_app"
MODEL_DIR = DEMO_DIR / "model"
UTILS_DIR = DEMO_DIR / "utils"

FILES = {
    DEMO_DIR / "app.py": "",
    MODEL_DIR / "global_model.pt": "",   # placeholder
    UTILS_DIR / "preprocessing.py": "",
    UTILS_DIR / "graph_builder.py": "",
    UTILS_DIR / "xai.py": "",
    UTILS_DIR / "report_generator.py": "",
    DEMO_DIR / "requirements.txt": "",
}

def create_structure():
    print("Creating demo_app structure...\n")

    for path, content in FILES.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w") as f:
                f.write(content)
            print(f"Created: {path}")
        else:
            print(f"Exists : {path}")

    print("\n✅ demo_app structure ready.")

if __name__ == "__main__":
    create_structure()

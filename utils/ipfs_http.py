#!/usr/bin/env python3
# ============================================================
# IPFS HTTP UTILITIES
# Stable • Python 3.12 compatible • FL Production-ready
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import shutil
import time
import requests
from pathlib import Path
from typing import Dict, List, Union

# ============================================================
# IPFS CONFIGURATION
# ============================================================

IPFS_API_URL = "http://127.0.0.1:5001/api/v0"
TIMEOUT = 300
RETRIES = 3
RETRY_DELAY = 2  # seconds

# ============================================================
# INTERNAL HELPERS
# ============================================================

def _ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def _post_with_retry(url, **kwargs):
    last_error = None
    for attempt in range(RETRIES):
        try:
            r = requests.post(url, timeout=TIMEOUT, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            last_error = e
            print(f"[IPFS] ⚠️ Retry {attempt+1}/{RETRIES} failed")
            time.sleep(RETRY_DELAY)
    raise RuntimeError(f"[IPFS] ❌ Request failed after retries: {last_error}")

def _extract_last_cid(response: requests.Response) -> str:
    """
    IPFS may return multiple JSON lines. We always take the last Hash.
    """
    lines = response.text.strip().split("\n")
    last = json.loads(lines[-1])
    return last["Hash"]

# ============================================================
# ADD FILE TO IPFS
# ============================================================

def ipfs_add_file(file_path: Union[str, Path]) -> str:
    """
    Upload a single file to IPFS.
    Returns CID.
    """
    path = Path(file_path)
    _ensure_exists(path)

    with path.open("rb") as f:
        r = _post_with_retry(
            f"{IPFS_API_URL}/add",
            files={"file": f},
        )

    cid = _extract_last_cid(r)
    return cid

# ============================================================
# ADD DIRECTORY TO IPFS (RECURSIVE)
# ============================================================

def ipfs_add_directory(dir_path: Union[str, Path]) -> str:
    """
    Upload a directory recursively to IPFS.
    Returns root CID.
    """
    dir_path = Path(dir_path)
    _ensure_exists(dir_path)

    files = []
    for file in dir_path.rglob("*"):
        if file.is_file():
            files.append(
                (
                    "file",
                    (
                        str(file.relative_to(dir_path)),
                        file.open("rb"),
                    ),
                )
            )

    r = _post_with_retry(
        f"{IPFS_API_URL}/add",
        files=files,
        params={
            "recursive": "true",
            "wrap-with-directory": "true",
        },
    )

    return _extract_last_cid(r)

# ============================================================
# FETCH FILE FROM IPFS
# ============================================================

def ipfs_fetch_file(cid: str, output_path: Union[str, Path]):
    """
    Fetch a single file from IPFS using CID.
    """
    output_path = Path(output_path)
    _ensure_parent(output_path)

    r = _post_with_retry(
        f"{IPFS_API_URL}/cat",
        params={"arg": cid},
        stream=True,
    )

    with output_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# ============================================================
# FETCH DIRECTORY FROM IPFS
# ============================================================

def ipfs_fetch_directory(cid: str, output_dir: Union[str, Path]):
    """
    Fetch a directory from IPFS using CID.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _post_with_retry(
        f"{IPFS_API_URL}/get",
        params={"arg": cid},
    )

    extracted_dir = Path(cid)
    if extracted_dir.exists() and extracted_dir.is_dir():
        for item in extracted_dir.iterdir():
            shutil.move(str(item), output_dir / item.name)
        extracted_dir.rmdir()

# ============================================================
# METADATA BUILDERS (TRACEABILITY CORE)
# ============================================================

def build_client_metadata(
    client_id: int,
    round_id: int,
    model_cid: str,
    metrics_cid: str,
    metrics: Dict,
) -> Dict:
    return {
        "type": "client_update",
        "client_id": client_id,
        "round": round_id,
        "model_cid": model_cid,
        "metrics_cid": metrics_cid,
        "metrics": metrics,
    }

def build_global_metadata(
    round_id: int,
    global_model_cid: str,
    global_metrics_cid: str,
    clients: List[int],
) -> Dict:
    return {
        "type": "global_update",
        "round": round_id,
        "global_model_cid": global_model_cid,
        "global_metrics_cid": global_metrics_cid,
        "participating_clients": clients,
    }

# ============================================================
# SAVE + UPLOAD METADATA JSON
# ============================================================

def ipfs_add_metadata(metadata: Dict, output_path: Union[str, Path]) -> str:
    """
    Save metadata JSON locally and upload to IPFS.
    Returns CID.
    """
    output_path = Path(output_path)
    _ensure_parent(output_path)

    with output_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return ipfs_add_file(output_path)

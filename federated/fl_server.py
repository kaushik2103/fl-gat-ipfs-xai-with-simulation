#!/usr/bin/env python3
# ============================================================
# FLOWER SERVER
# FedProx + FL-SAFE Security
# CPU Global Evaluation (OOM-PROOF)
# IPFS Traceability
# Python 3.12 compatible
# ============================================================

import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import torch
import flwr as fl
import gc

from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.data import Data

from model.gat_residual_bn import StrongResidualGAT
from utils.ipfs_http import (
    ipfs_add_file,
    ipfs_add_metadata,
    build_global_metadata,
)

# ============================================================
# CONFIG
# ============================================================

DEVICE_EVAL = "cpu"              # 🔥 HARD-FIX (NO CUDA HERE)
NUM_CLASSES = 7                  # MUST match clients

LOG_DIR = Path("server_logs")
MODEL_DIR = LOG_DIR / "models"
METRIC_DIR = LOG_DIR / "metrics"
IPFS_DIR = LOG_DIR / "ipfs"

for d in (MODEL_DIR, METRIC_DIR, IPFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Security thresholds (FL-safe) ----
NORM_MULTIPLIER = 10.0
COSINE_THRESHOLD = 0.10
SYBIL_SIM_THRESHOLD = 0.9995
SYBIL_NORM_RATIO = 0.30


# ============================================================
# GLOBAL MODEL EVALUATION (CPU ONLY)
# ============================================================

def evaluate_global_model(
    parameters: List[np.ndarray],
    test_graph_path: str,
) -> Dict[str, float]:
    """
    CPU-based global evaluation (OOM-safe).
    """

    print("[SERVER] 🧪 Evaluating global model on CPU")

    gc.collect()
    torch.cuda.empty_cache()  # safety even if unused

    data: Data = torch.load(
        test_graph_path,
        map_location="cpu",
        weights_only=False,
    )

    model = StrongResidualGAT(
        in_channels=data.x.size(1),
        hidden_channels=128,
        num_classes=NUM_CLASSES,
        heads=4,
        dropout=0.2,
        attn_dropout=0.2,
        edge_dropout=0.1,
    ).to("cpu")

    state_dict = dict(
        zip(
            model.state_dict().keys(),
            [torch.tensor(p, device="cpu") for p in parameters],
        )
    )
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1).numpy()
        targets = data.y.numpy()

    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        preds,
        average="macro",
        zero_division=0,
    )

    # ---- CLEANUP ----
    del model, data, logits
    gc.collect()

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# FL-SAFE MALICIOUS CLIENT DETECTION
# ============================================================

def detect_malicious_updates(
    results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]
) -> bool:
    """
    FL-safe poisoning, backdoor, sybil detection.
    """

    updates = []
    for _, fit_res in results:
        ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
        flat = np.concatenate([p.reshape(-1) for p in ndarrays])
        updates.append(flat.astype(np.float32))

    updates = np.stack(updates)
    norms = np.linalg.norm(updates, axis=1)
    median_norm = np.median(norms)

    # ---- Poisoning ----
    if norms.max() > NORM_MULTIPLIER * median_norm:
        print("[SECURITY] ❌ Poisoning detected")
        return True

    # ---- Backdoor ----
    base = updates[0] / (np.linalg.norm(updates[0]) + 1e-8)
    for u in updates[1:]:
        sim = np.dot(base, u / (np.linalg.norm(u) + 1e-8))
        if sim < COSINE_THRESHOLD:
            print("[SECURITY] ❌ Backdoor detected")
            return True

    # ---- Sybil (FIXED false positives) ----
    for i in range(len(updates)):
        for j in range(i + 1, len(updates)):
            sim = np.dot(
                updates[i] / (np.linalg.norm(updates[i]) + 1e-8),
                updates[j] / (np.linalg.norm(updates[j]) + 1e-8),
            )
            if (
                sim > SYBIL_SIM_THRESHOLD
                and norms[i] < SYBIL_NORM_RATIO * median_norm
                and norms[j] < SYBIL_NORM_RATIO * median_norm
            ):
                print("[SECURITY] ❌ Sybil detected")
                return True

    print("[SECURITY] ✅ Updates clean")
    return False


# ============================================================
# SECURE FEDPROX STRATEGY
# ============================================================

class SecureFedProx(fl.server.strategy.FedProx):

    def __init__(self, test_graph_path: str, **kwargs):
        super().__init__(**kwargs)
        self.test_graph_path = test_graph_path

    def aggregate_fit(self, rnd: int, results, failures):
        print(f"\n[SERVER] 🔄 Aggregation round {rnd}")

        if not results:
            return None, {}

        if detect_malicious_updates(results):
            print("[SERVER] ❌ Aggregation blocked")
            return None, {}

        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is None:
            return None, {}

        aggregated_parameters, _ = aggregated
        ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

        # ---- SAVE GLOBAL MODEL ----
        model_path = MODEL_DIR / f"global_round_{rnd}.pt"
        torch.save(ndarrays, model_path)
        model_cid = ipfs_add_file(model_path)

        # ---- CPU EVALUATION ----
        metrics = evaluate_global_model(ndarrays, self.test_graph_path)
        metrics_path = METRIC_DIR / f"round_{rnd}_metrics.json"

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        metrics_cid = ipfs_add_file(metrics_path)

        # ---- METADATA ----
        client_ids = [int(r.metrics["client_id"]) for _, r in results]

        meta = build_global_metadata(
            round_id=rnd,
            global_model_cid=model_cid,
            global_metrics_cid=metrics_cid,
            clients=client_ids,
        )

        meta_path = IPFS_DIR / f"round_{rnd}_meta.json"
        meta_cid = ipfs_add_metadata(meta, meta_path)

        print(
            f"[IPFS] 🌐 Round {rnd} stored | "
            f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f}"
        )

        return aggregated_parameters, metrics


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Flower Server (FedProx + Secure FL, CPU Eval)"
    )
    parser.add_argument("--server_addr", default="0.0.0.0:8080")
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--min_clients", type=int, default=4)
    parser.add_argument("--test_graph", required=True)
    args = parser.parse_args()

    strategy = SecureFedProx(
        test_graph_path=args.test_graph,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        proximal_mu=0.05,
    )

    fl.server.start_server(
        server_address=args.server_addr,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )


if __name__ == "__main__":
    main()

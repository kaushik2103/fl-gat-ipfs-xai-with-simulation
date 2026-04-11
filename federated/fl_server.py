#!/usr/bin/env python3
# ============================================================
# FLOWER SERVER (FALSE-POSITIVE SAFE FINAL VERSION)
# FedProx + Robust FL Security + Warmup + Majority Rule
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

DEVICE_EVAL = "cpu"
NUM_CLASSES = 2

LOG_DIR = Path("server_logs")
MODEL_DIR = LOG_DIR / "models"
METRIC_DIR = LOG_DIR / "metrics"
IPFS_DIR = LOG_DIR / "ipfs"

for d in (MODEL_DIR, METRIC_DIR, IPFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ✅ RELAXED THRESHOLDS (FALSE POSITIVE FIX)
NORM_MULTIPLIER = 5.0
COSINE_THRESHOLD = 0.1
SYBIL_SIM_THRESHOLD = 0.995
SYBIL_NORM_RATIO = 0.5

# ✅ NEW
WARMUP_ROUNDS = 3
MALICIOUS_RATIO_THRESHOLD = 0.5


# ============================================================
# GLOBAL MODEL EVALUATION
# ============================================================

def evaluate_global_model(parameters, test_graph_path):

    print("[SERVER] 🧪 Evaluating global model")

    data: Data = torch.load(test_graph_path, map_location="cpu")

    model = StrongResidualGAT(
        in_channels=data.x.size(1),
        hidden_channels=128,
        num_classes=NUM_CLASSES,
        heads=4,
    ).to("cpu")

    state_dict = dict(
        zip(model.state_dict().keys(), [torch.tensor(p) for p in parameters])
    )

    model.load_state_dict(state_dict, strict=True)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1).numpy()
        targets = data.y.numpy()

    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# 🚨 MALICIOUS DETECTION (FINAL SAFE)
# ============================================================

def detect_malicious_updates(results):

    updates = []

    for _, fit_res in results:
        ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
        flat = np.concatenate([p.reshape(-1) for p in ndarrays])
        updates.append(flat.astype(np.float32))

    updates = np.stack(updates)

    norms = np.linalg.norm(updates, axis=1)
    median_norm = np.median(norms)

    malicious_clients = set()

    print(f"[DEBUG] Norms: {norms}")

    # ========================================================
    # 1. POISONING DETECTION
    # ========================================================
    for i, norm in enumerate(norms):
        if norm > NORM_MULTIPLIER * median_norm:
            print(f"[SECURITY] ❌ Poisoning (Client {i})")
            malicious_clients.add(i)

    # ========================================================
    # 2. BACKDOOR DETECTION
    # ========================================================
    global_update = np.mean(updates, axis=0)
    global_update /= (np.linalg.norm(global_update) + 1e-8)

    for i, u in enumerate(updates):
        u_norm = u / (np.linalg.norm(u) + 1e-8)
        sim = np.dot(u_norm, global_update)

        if sim < COSINE_THRESHOLD:
            print(f"[SECURITY] ❌ Backdoor (Client {i})")
            malicious_clients.add(i)

    # ========================================================
    # 3. SYBIL DETECTION
    # ========================================================
    for i in range(len(updates)):
        for j in range(i + 1, len(updates)):

            ui = updates[i] / (np.linalg.norm(updates[i]) + 1e-8)
            uj = updates[j] / (np.linalg.norm(updates[j]) + 1e-8)

            sim = np.dot(ui, uj)

            if (
                sim > SYBIL_SIM_THRESHOLD
                and norms[i] < SYBIL_NORM_RATIO * median_norm
                and norms[j] < SYBIL_NORM_RATIO * median_norm
            ):
                print(f"[SECURITY] ❌ Sybil ({i},{j})")
                malicious_clients.add(i)
                malicious_clients.add(j)

    print(f"[SECURITY] Detected malicious: {malicious_clients}")

    return malicious_clients


# ============================================================
# FEDPROX STRATEGY WITH SAFE AGGREGATION
# ============================================================

class SecureFedProx(fl.server.strategy.FedProx):

    def __init__(self, test_graph_path, **kwargs):
        super().__init__(**kwargs)
        self.test_graph_path = test_graph_path

    def aggregate_fit(self, rnd, results, failures):

        print(f"\n[SERVER] 🔄 Round {rnd}")

        if not results:
            return None, {}

        # ✅ WARM-UP (skip detection early)
        if rnd <= WARMUP_ROUNDS:
            print("[SECURITY] ⏳ Warm-up phase (skipping detection)")
            malicious_clients = set()
        else:
            malicious_clients = detect_malicious_updates(results)

        total_clients = len(results)
        malicious_ratio = len(malicious_clients) / total_clients

        print(f"[SECURITY] Malicious Ratio: {malicious_ratio:.2f}")

        # ✅ BLOCK ONLY IF MAJORITY MALICIOUS
        if malicious_ratio > MALICIOUS_RATIO_THRESHOLD:
            print("[SERVER] ❌ Aggregation BLOCKED (Major attack)")
            return None, {}

        # ✅ FILTER MALICIOUS CLIENTS (NOT FULL STOP)
        clean_results = [
            res for idx, res in enumerate(results)
            if idx not in malicious_clients
        ]

        if len(clean_results) == 0:
            print("[SERVER] ❌ No clean clients left")
            return None, {}

        aggregated = super().aggregate_fit(rnd, clean_results, failures)
        if aggregated is None:
            return None, {}

        parameters, _ = aggregated
        ndarrays = fl.common.parameters_to_ndarrays(parameters)

        # SAVE MODEL
        model_path = MODEL_DIR / f"global_round_{rnd}.pt"
        torch.save(ndarrays, model_path)
        model_cid = ipfs_add_file(model_path)

        # EVALUATE
        metrics = evaluate_global_model(ndarrays, self.test_graph_path)

        metrics_path = METRIC_DIR / f"round_{rnd}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        metrics_cid = ipfs_add_file(metrics_path)

        # METADATA
        client_ids = [int(r.metrics["client_id"]) for _, r in clean_results]

        meta = build_global_metadata(
            round_id=rnd,
            global_model_cid=model_cid,
            global_metrics_cid=metrics_cid,
            clients=client_ids,
        )

        meta_path = IPFS_DIR / f"round_{rnd}.json"
        ipfs_add_metadata(meta, meta_path)

        print(
            f"[SERVER] ✅ Round {rnd} | "
            f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f}"
        )

        return parameters, metrics


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_addr", default="0.0.0.0:8080")
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--min_clients", type=int, default=7)
    parser.add_argument("--test_graph", required=True)

    args = parser.parse_args()

    strategy = SecureFedProx(
        test_graph_path=args.test_graph,
        fraction_fit=1.0,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        proximal_mu=0.001,
    )

    fl.server.start_server(
        server_address=args.server_addr,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )


if __name__ == "__main__":
    main()
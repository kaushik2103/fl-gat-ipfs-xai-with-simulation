#!/usr/bin/env python3
# ============================================================
# FLOWER SERVER (FINAL WITH FULL REPORTING + IPFS)
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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from torch_geometric.data import Data

from model.gat_residual_bn import StrongResidualGAT
from utils.ipfs_http import ipfs_add_file, ipfs_add_metadata, build_global_metadata


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cpu"
NUM_CLASSES = 2

LOG_DIR = Path("server_logs")
MODEL_DIR = LOG_DIR / "models"
REPORT_DIR = LOG_DIR / "reports"
IPFS_DIR = LOG_DIR / "ipfs"

for d in (MODEL_DIR, REPORT_DIR, IPFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Tracking across rounds
GLOBAL_HISTORY = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
}


# ============================================================
# EVALUATION + CONFUSION MATRIX
# ============================================================

def evaluate_and_report(parameters, test_graph_path, rnd):

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

    # Metrics
    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )

    # Store history
    GLOBAL_HISTORY["accuracy"].append(acc)
    GLOBAL_HISTORY["precision"].append(precision)
    GLOBAL_HISTORY["recall"].append(recall)
    GLOBAL_HISTORY["f1"].append(f1)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================
    cm = confusion_matrix(targets, preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - Round {rnd}")
    cm_path = REPORT_DIR / f"cm_round_{rnd}.png"
    plt.savefig(cm_path)
    plt.close()

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================
    report = classification_report(targets, preds)

    report_path = REPORT_DIR / f"classification_round_{rnd}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Upload to IPFS
    cm_cid = ipfs_add_file(cm_path)
    report_cid = ipfs_add_file(report_path)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm_cid": cm_cid,
        "report_cid": report_cid,
    }


# ============================================================
# GLOBAL REPORT (ALL ROUNDS)
# ============================================================

def generate_global_report():

    rounds = list(range(1, len(GLOBAL_HISTORY["accuracy"]) + 1))

    # Accuracy Plot
    plt.figure()
    plt.plot(rounds, GLOBAL_HISTORY["accuracy"], label="Accuracy")
    plt.plot(rounds, GLOBAL_HISTORY["f1"], label="F1 Score")
    plt.legend()
    plt.xlabel("Rounds")
    plt.title("Accuracy & F1 over Rounds")
    acc_plot_path = REPORT_DIR / "global_accuracy.png"
    plt.savefig(acc_plot_path)
    plt.close()

    # Precision Recall Plot
    plt.figure()
    plt.plot(rounds, GLOBAL_HISTORY["precision"], label="Precision")
    plt.plot(rounds, GLOBAL_HISTORY["recall"], label="Recall")
    plt.legend()
    plt.title("Precision vs Recall")
    pr_plot_path = REPORT_DIR / "precision_recall.png"
    plt.savefig(pr_plot_path)
    plt.close()

    # Save JSON
    global_report_path = REPORT_DIR / "global_metrics.json"
    with open(global_report_path, "w") as f:
        json.dump(GLOBAL_HISTORY, f, indent=2)

    # Upload
    acc_cid = ipfs_add_file(acc_plot_path)
    pr_cid = ipfs_add_file(pr_plot_path)
    json_cid = ipfs_add_file(global_report_path)

    print(f"[SERVER] 🌐 Global Report Uploaded")

    return {
        "accuracy_plot": acc_cid,
        "precision_recall_plot": pr_cid,
        "metrics": json_cid,
    }


# ============================================================
# STRATEGY
# ============================================================

class SecureFedProx(fl.server.strategy.FedProx):

    def __init__(self, test_graph_path, **kwargs):
        super().__init__(**kwargs)
        self.test_graph_path = test_graph_path

    def aggregate_fit(self, rnd, results, failures):

        print(f"\n[SERVER] 🔄 Round {rnd}")

        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is None:
            return None, {}

        parameters, _ = aggregated
        ndarrays = fl.common.parameters_to_ndarrays(parameters)

        # Save model
        model_path = MODEL_DIR / f"global_round_{rnd}.pt"
        torch.save(ndarrays, model_path)
        model_cid = ipfs_add_file(model_path)

        # Evaluate + generate round report
        metrics = evaluate_and_report(
            ndarrays,
            self.test_graph_path,
            rnd
        )

        # Metadata
        meta = build_global_metadata(
            round_id=rnd,
            global_model_cid=model_cid,
            global_metrics_cid=metrics["report_cid"],
            clients=[int(r.metrics["client_id"]) for _, r in results],
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
        proximal_mu=0.01,
    )

    fl.server.start_server(
        server_address=args.server_addr,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )

    # AFTER TRAINING → GLOBAL REPORT
    generate_global_report()


if __name__ == "__main__":
    main()
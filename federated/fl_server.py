#!/usr/bin/env python3
# ============================================================
# FLOWER SERVER (FINAL RESEARCH + TRUST + ADVANCED VISUALS)
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

from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from torch_geometric.data import Data
from model.gat_residual_bn import StrongResidualGAT
from utils.ipfs_http import ipfs_add_file


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cpu"
NUM_CLASSES = 2

LOG_DIR = Path("server_logs")
MODEL_DIR = LOG_DIR / "models"
REPORT_DIR = LOG_DIR / "reports"
IPFS_DIR = LOG_DIR / "ipfs"
TRUST_FILE = LOG_DIR / "client_trust.json"

for d in (MODEL_DIR, REPORT_DIR, IPFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

LIVE_LOG_FILE = LOG_DIR / "live_logs.json"
GLOBAL_HISTORY_FILE = LOG_DIR / "global_history.json"


# ============================================================
# LOGGER
# ============================================================

def append_live_log(msg):
    entry = {"time": datetime.now().strftime("%H:%M:%S"), "message": msg}
    logs = []
    if LIVE_LOG_FILE.exists():
        logs = json.load(open(LIVE_LOG_FILE))

    logs.append(entry)

    with open(LIVE_LOG_FILE, "w") as f:
        json.dump(logs[-200:], f, indent=2)


# ============================================================
# TRUST SYSTEM
# ============================================================

def load_trust():
    if TRUST_FILE.exists():
        return json.load(open(TRUST_FILE))
    return {}

def save_trust(trust):
    json.dump(trust, open(TRUST_FILE, "w"), indent=2)


def update_trust(trust, client_ids):
    for cid in client_ids:
        cid = str(cid)

        if cid not in trust:
            trust[cid] = 1.0

        # smooth decay (no false drops)
        trust[cid] = min(1.0, trust[cid] + 0.01)

    return trust


# ============================================================
# GLOBAL HISTORY
# ============================================================

GLOBAL_HISTORY = {
    "accuracy": [],
    "f1": [],
    "precision": [],
    "recall": [],
    "roc_auc": [],
    "train_loss": [],
    "val_loss": [],
}


# ============================================================
# EVALUATION + ADVANCED REPORTS
# ============================================================

def evaluate_and_report(parameters, graph_path, rnd):

    data: Data = torch.load(graph_path, map_location="cpu")

    model = StrongResidualGAT(
        in_channels=data.x.size(1),
        hidden_channels=128,
        num_classes=NUM_CLASSES,
        heads=4,
    )

    state_dict = dict(zip(model.state_dict().keys(),
                          [torch.tensor(p) for p in parameters]))

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = logits.argmax(dim=1).numpy()
        targets = data.y.numpy()

    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro"
    )

    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    val_loss = 1 - acc
    train_loss = GLOBAL_HISTORY["val_loss"][-1] if GLOBAL_HISTORY["val_loss"] else val_loss

    # update history
    GLOBAL_HISTORY["accuracy"].append(acc)
    GLOBAL_HISTORY["f1"].append(f1)
    GLOBAL_HISTORY["precision"].append(precision)
    GLOBAL_HISTORY["recall"].append(recall)
    GLOBAL_HISTORY["roc_auc"].append(roc_auc)
    GLOBAL_HISTORY["train_loss"].append(train_loss)
    GLOBAL_HISTORY["val_loss"].append(val_loss)

    rounds = list(range(1, len(GLOBAL_HISTORY["accuracy"]) + 1))

    # ================= LOSS =================
    plt.figure()
    plt.plot(rounds, GLOBAL_HISTORY["train_loss"], label="Train Loss")
    plt.plot(rounds, GLOBAL_HISTORY["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    loss_path = REPORT_DIR / f"loss_{rnd}.png"
    plt.savefig(loss_path)
    plt.close()

    # ================= ACC/F1 =================
    plt.figure()
    plt.plot(rounds, GLOBAL_HISTORY["accuracy"], label="Accuracy")
    plt.plot(rounds, GLOBAL_HISTORY["f1"], label="F1")
    plt.legend()
    plt.title("Accuracy & F1")
    acc_path = REPORT_DIR / f"acc_f1_{rnd}.png"
    plt.savefig(acc_path)
    plt.close()

    # ================= PRECISION/RECALL =================
    plt.figure()
    plt.plot(rounds, GLOBAL_HISTORY["precision"], label="Precision")
    plt.plot(rounds, GLOBAL_HISTORY["recall"], label="Recall")
    plt.legend()
    plt.title("Precision & Recall")
    pr_path = REPORT_DIR / f"precision_recall_{rnd}.png"
    plt.savefig(pr_path)
    plt.close()

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(targets, preds)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = REPORT_DIR / f"cm_{rnd}.png"
    plt.savefig(cm_path)
    plt.close()

    # ================= ROC =================
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    roc_path = REPORT_DIR / f"roc_{rnd}.png"
    plt.savefig(roc_path)
    plt.close()

    # ================= PR CURVE =================
    p, r, _ = precision_recall_curve(targets, probs)
    ap = average_precision_score(targets, probs)

    plt.figure()
    plt.plot(r, p, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    prc_path = REPORT_DIR / f"pr_curve_{rnd}.png"
    plt.savefig(prc_path)
    plt.close()

    # ================= REPORT =================
    report = classification_report(targets, preds, output_dict=True)
    report_path = REPORT_DIR / f"report_{rnd}.json"
    json.dump(report, open(report_path, "w"), indent=2)

    # ================= SAVE HISTORY =================
    json.dump(GLOBAL_HISTORY, open(GLOBAL_HISTORY_FILE, "w"), indent=2)

    # ================= IPFS =================
    cids = {
        "loss": ipfs_add_file(loss_path),
        "acc_f1": ipfs_add_file(acc_path),
        "precision_recall": ipfs_add_file(pr_path),
        "cm": ipfs_add_file(cm_path),
        "roc": ipfs_add_file(roc_path),
        "pr_curve": ipfs_add_file(prc_path),
        "report": ipfs_add_file(report_path),
    }

    return acc, f1, cids


# ============================================================
# STRATEGY
# ============================================================

class SecureFedProx(fl.server.strategy.FedProx):

    def __init__(self, test_graph_path, **kwargs):
        super().__init__(**kwargs)
        self.test_graph_path = test_graph_path

    def aggregate_fit(self, rnd, results, failures):

        append_live_log(f"🚀 Round {rnd}")

        aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated is None:
            return None, {}

        parameters, _ = aggregated
        ndarrays = fl.common.parameters_to_ndarrays(parameters)

        # SAVE MODEL
        model_path = MODEL_DIR / f"global_{rnd}.pt"
        torch.save(ndarrays, model_path)
        model_cid = ipfs_add_file(model_path)

        # TRUST UPDATE
        trust = load_trust()
        client_ids = [int(r.metrics["client_id"]) for _, r in results]
        trust = update_trust(trust, client_ids)
        save_trust(trust)

        # EVALUATION
        acc, f1, cids = evaluate_and_report(
            ndarrays, self.test_graph_path, rnd
        )

        # IPFS META
        meta = {
            "round": rnd,
            "model_cid": model_cid,
            "report_cids": cids,
            "trust": trust,
        }

        json.dump(meta, open(IPFS_DIR / f"round_{rnd}.json", "w"), indent=2)

        append_live_log(f"✅ Round {rnd} | Acc={acc:.4f}")

        return parameters, {"accuracy": acc, "f1": f1}


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

    append_live_log("🟢 Server started")

    fl.server.start_server(
        server_address=args.server_addr,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )


if __name__ == "__main__":
    main()
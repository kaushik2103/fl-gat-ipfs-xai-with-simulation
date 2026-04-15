#!/usr/bin/env python3
# ============================================================
# FLOWER CLIENT (RESEARCH-GRADE + ADVANCED VISUALIZATION)
# ============================================================

import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
import numpy as np
import flwr as fl
import matplotlib.pyplot as plt

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

from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from model.gat_residual_bn import StrongResidualGAT
from utils.ipfs_http import ipfs_add_file


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 2
MAX_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-6
DEVICE = "cuda"
BATCH_SIZE = 256
NUM_NEIGHBORS = [8, 4, 2]


# ============================================================
# CLIENT
# ============================================================

class GATFlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id, graph_path, output_dir):

        self.client_id = client_id
        self.output_dir = Path(output_dir) / f"client_{client_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.output_dir / "client_logs.json"
        self.status_file = self.output_dir / "client_status.json"

        self.data: Data = torch.load(graph_path, map_location="cpu")

        self.train_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.train_mask,
            num_neighbors=NUM_NEIGHBORS,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        self.val_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.test_mask,
            num_neighbors=[-1],
            batch_size=BATCH_SIZE,
        )

        self.model = StrongResidualGAT(
            in_channels=self.data.x.size(1),
            hidden_channels=128,
            num_classes=NUM_CLASSES,
            heads=4,
        ).to(DEVICE)

        self.optimizer = AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.scaler = GradScaler()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.metrics_per_epoch = []

        self.last_preds = None
        self.last_targets = None
        self.last_probs = None

    # ========================================================
    # FLOWER API
    # ========================================================

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(
            zip(self.model.state_dict().keys(),
                [torch.tensor(p, device=DEVICE) for p in parameters])
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        for epoch in range(MAX_EPOCHS):

            train_loss = self._train_epoch()
            val_loss, metrics = self._evaluate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.metrics_per_epoch.append(metrics)

            print(f"[CLIENT {self.client_id}] Epoch {epoch+1} | "
                  f"Train={train_loss:.4f} | Val={val_loss:.4f} | "
                  f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f}")

        payload_cid = self._generate_full_report()

        return self.get_parameters({}), int(len(self.data.x)), {
            "client_id": int(self.client_id),
            "accuracy": float(metrics["accuracy"]),
            "f1": float(metrics["f1"]),
            "ipfs_payload_cid": str(payload_cid),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        _, metrics = self._evaluate()

        return float(1.0 - metrics["accuracy"]), int(len(self.data.x)), {
            "accuracy": float(metrics["accuracy"]),
            "f1": float(metrics["f1"]),
        }

    # ========================================================
    # TRAIN
    # ========================================================

    def _train_epoch(self):

        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            batch = batch.to(DEVICE)

            self.optimizer.zero_grad()

            with autocast():
                logits = self.model(batch.x, batch.edge_index)
                loss = self.criterion(
                    logits[:batch.batch_size],
                    batch.y[:batch.batch_size]
                )

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # ========================================================
    # EVALUATE
    # ========================================================

    def _evaluate(self):

        self.model.eval()
        preds, targets, probs = [], [], []
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(DEVICE)

                logits = self.model(batch.x, batch.edge_index)

                loss = self.criterion(
                    logits[:batch.batch_size],
                    batch.y[:batch.batch_size]
                )

                total_loss += loss.item()

                prob = torch.softmax(logits, dim=1)

                preds.append(logits[:batch.batch_size].argmax(dim=1).cpu().numpy())
                targets.append(batch.y[:batch.batch_size].cpu().numpy())
                probs.append(prob[:batch.batch_size, 1].cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        probs = np.concatenate(probs)

        self.last_preds = preds
        self.last_targets = targets
        self.last_probs = probs

        acc = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0
        )

        return total_loss / len(self.val_loader), {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ========================================================
    # REPORT GENERATION (UPGRADED VISUALS)
    # ========================================================

    def _generate_full_report(self):

        preds = self.last_preds
        targets = self.last_targets
        probs = self.last_probs

        accs = [m["accuracy"] for m in self.metrics_per_epoch]
        f1s = [m["f1"] for m in self.metrics_per_epoch]
        precs = [m["precision"] for m in self.metrics_per_epoch]
        recs = [m["recall"] for m in self.metrics_per_epoch]

        epochs = list(range(1, len(accs)+1))

        # LOSS
        plt.figure()
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Val Loss")
        plt.legend()
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        loss_path = self.output_dir / "loss.png"
        plt.savefig(loss_path)
        plt.close()

        # ACC + F1
        plt.figure()
        plt.plot(epochs, accs, label="Accuracy")
        plt.plot(epochs, f1s, label="F1")
        plt.legend()
        plt.title("Accuracy vs F1")
        acc_path = self.output_dir / "acc_f1.png"
        plt.savefig(acc_path)
        plt.close()

        # PRECISION / RECALL
        plt.figure()
        plt.plot(epochs, precs, label="Precision")
        plt.plot(epochs, recs, label="Recall")
        plt.legend()
        plt.title("Precision vs Recall")
        pr_path = self.output_dir / "precision_recall.png"
        plt.savefig(pr_path)
        plt.close()

        # ROC (WITH AUC)
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        roc_path = self.output_dir / "roc.png"
        plt.savefig(roc_path)
        plt.close()

        # PR CURVE (WITH AP)
        p, r, _ = precision_recall_curve(targets, probs)
        ap = average_precision_score(targets, probs)

        plt.figure()
        plt.plot(r, p, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        prc_path = self.output_dir / "pr_curve.png"
        plt.savefig(prc_path)
        plt.close()

        # CONFUSION MATRIX (WITH NUMBERS)
        cm = confusion_matrix(targets, preds)

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                         ha="center",
                         color="white" if cm[i, j] > thresh else "black")

        cm_path = self.output_dir / "cm.png"
        plt.savefig(cm_path)
        plt.close()

        # REPORT
        report = classification_report(targets, preds, output_dict=True)
        report_path = self.output_dir / "report.json"
        json.dump(report, open(report_path, "w"), indent=2)

        # IPFS
        cids = {
            "loss": ipfs_add_file(loss_path),
            "acc_f1": ipfs_add_file(acc_path),
            "precision_recall": ipfs_add_file(pr_path),
            "roc": ipfs_add_file(roc_path),
            "pr_curve": ipfs_add_file(prc_path),
            "cm": ipfs_add_file(cm_path),
            "report": ipfs_add_file(report_path),
        }

        payload = {
            "client_id": int(self.client_id),
            "cids": cids
        }

        payload_path = self.output_dir / "payload.json"
        json.dump(payload, open(payload_path, "w"), indent=2)

        return ipfs_add_file(payload_path)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--graph", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="client_outputs")
    parser.add_argument("--server_addr", type=str, default="127.0.0.1:8080")

    args = parser.parse_args()

    client = GATFlowerClient(
        args.client_id,
        args.graph,
        args.output_dir,
    )

    fl.client.start_numpy_client(
        server_address=args.server_addr,
        client=client,
    )


if __name__ == "__main__":
    main()
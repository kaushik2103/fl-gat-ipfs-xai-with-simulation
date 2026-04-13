#!/usr/bin/env python3
# ============================================================
# FLOWER CLIENT (FINAL RESEARCH VERSION)
# With Full Reporting + Plots + IPFS Upload
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
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
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

        self.data: Data = torch.load(graph_path, map_location="cpu")

        # ---------------- LOADERS ----------------
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

        # ---------------- MODEL ----------------
        self.model = StrongResidualGAT(
            in_channels=self.data.x.size(1),
            hidden_channels=128,
            num_classes=NUM_CLASSES,
            heads=4,
        ).to(DEVICE)

        self.optimizer = AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.scaler = GradScaler()

        self.criterion = torch.nn.CrossEntropyLoss()

        # ---------------- TRACKERS ----------------
        self.train_losses = []
        self.val_losses = []
        self.metrics_per_epoch = []

    # ========================================================
    # FLOWER API
    # ========================================================

    def get_parameters(self, config):
        return [p.cpu().numpy() for p in self.model.state_dict().values()]

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
                  f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
                  f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f}")

        # Save reports
        payload_cid = self._generate_full_report()

        return self.get_parameters({}), len(self.data.x), {
            "client_id": self.client_id,
            "ipfs_payload_cid": payload_cid,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        _, metrics = self._evaluate()
        return 1.0 - metrics["accuracy"], len(self.data.x), metrics

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
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # ========================================================
    # EVALUATE
    # ========================================================

    def _evaluate(self):

        self.model.eval()

        preds, targets = [], []
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

                preds.append(logits[:batch.batch_size].argmax(dim=1).cpu().numpy())
                targets.append(batch.y[:batch.batch_size].cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

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
    # REPORT GENERATION
    # ========================================================

    def _generate_full_report(self):

        # ---------------- LOSS CURVES ----------------
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.legend()
        plt.title("Loss Curve")
        loss_path = self.output_dir / "loss_curve.png"
        plt.savefig(loss_path)
        plt.close()

        # ---------------- FINAL CONFUSION MATRIX ----------------
        _, final_metrics = self._evaluate()
        preds, targets = self._get_preds_targets()

        cm = confusion_matrix(targets, preds)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        cm_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # ---------------- CLASSIFICATION REPORT ----------------
        report = classification_report(targets, preds, output_dict=True)

        report_path = self.output_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # ---------------- METRICS HISTORY ----------------
        metrics_path = self.output_dir / "metrics_history.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_per_epoch, f, indent=2)

        # ---------------- UPLOAD ----------------
        loss_cid = ipfs_add_file(loss_path)
        cm_cid = ipfs_add_file(cm_path)
        report_cid = ipfs_add_file(report_path)

        payload = {
            "client_id": self.client_id,
            "loss_curve": loss_cid,
            "confusion_matrix": cm_cid,
            "classification_report": report_cid,
        }

        payload_path = self.output_dir / "payload.json"
        with open(payload_path, "w") as f:
            json.dump(payload, f, indent=2)

        payload_cid = ipfs_add_file(payload_path)

        print(f"[CLIENT {self.client_id}] 📊 Report CID: {payload_cid}")

        return payload_cid

    def _get_preds_targets(self):

        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(DEVICE)
                logits = self.model(batch.x, batch.edge_index)

                preds.append(logits[:batch.batch_size].argmax(dim=1).cpu().numpy())
                targets.append(batch.y[:batch.batch_size].cpu().numpy())

        return np.concatenate(preds), np.concatenate(targets)


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
#!/usr/bin/env python3
# ============================================================
# FLOWER CLIENT
# Simple Residual GAT + Mini-batch GPU Training (FAST)
# IPFS Storage via HTTP API
# Python 3.12 compatible
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

from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from model.gat_residual_bn import StrongResidualGAT
from utils.ipfs_http import ipfs_add_file


# ============================================================
# CONFIG (MATCHES YOUR DATA + FAST FL)
# ============================================================

NUM_CLASSES = 7              # ✔ confirmed
MAX_EPOCHS = 4            # ✔ FL best practice
LR = 2e-3 #2e-3                    # ✔ stable for non-IID
WEIGHT_DECAY = 5e-4

DEVICE = "cuda"
BATCH_SIZE = 128

# 🔥 Reduced expansion = MASSIVE speedup
NUM_NEIGHBORS = [8, 4, 2]


# ============================================================
# FLOWER CLIENT
# ============================================================

class GATFlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id: int, graph_path: str, output_dir: str):
        self.client_id = client_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[CLIENT {self.client_id}] Loading graph → {graph_path}")
        self.data: Data = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False,
        )

        # ----------------------------------------------------
        # MINI-BATCH LOADERS
        # ----------------------------------------------------
        self.train_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.train_mask,
            num_neighbors=NUM_NEIGHBORS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            persistent_workers=False,
        )

        self.val_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.test_mask,
            num_neighbors=[-1],
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            persistent_workers=False,
        )

        # ----------------------------------------------------
        # MODEL (SIMPLE + FAST)
        # ----------------------------------------------------
        self.model = StrongResidualGAT(
            in_channels=self.data.x.size(1),   # 🔥 auto = 57
            hidden_channels=128,
            num_classes=NUM_CLASSES,
            heads=4,
            dropout=0.2,
            attn_dropout=0.2,
            edge_dropout=0.1,
        ).to(DEVICE)

        # ----------------------------------------------------
        # OPTIMIZER + AMP
        # ----------------------------------------------------
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        self.scaler = GradScaler()

        # ----------------------------------------------------
        # CLASS-WEIGHTED LOSS (NORMALIZED)
        # ----------------------------------------------------
        labels = self.data.y[self.data.train_mask].numpy()
        counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)

        weights = counts.sum() / (counts + 1e-6)
        weights = weights / weights.mean()    # 🔥 critical

        self.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, device=DEVICE),
            reduction="mean",
        )

        self.best_f1 = 0.0
        self.best_state = None

    # ========================================================
    # FLOWER API
    # ========================================================

    def get_parameters(self, config):
        return [
            p.detach().cpu().numpy()
            for p in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters):
        state_dict = dict(
            zip(
                self.model.state_dict().keys(),
                [torch.tensor(p, device=DEVICE) for p in parameters],
            )
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(1, MAX_EPOCHS + 1):
            total_loss, steps = 0.0, 0

            for batch in self.train_loader:
                batch = batch.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    logits = self.model(batch.x, batch.edge_index)
                    loss = self.criterion(
                        logits[: batch.batch_size],
                        batch.y[: batch.batch_size],
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            metrics = self._evaluate_local()

            print(
                f"[CLIENT {self.client_id}] "
                f"Epoch {epoch:02d} | "
                f"Loss={avg_loss:.4f} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f}"
            )

            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                self.best_state = self.model.state_dict()

        # Restore best model
        self.model.load_state_dict(self.best_state)

        payload_cid = self._save_and_upload(metrics)
        torch.cuda.empty_cache()

        return self.get_parameters({}), int(self.data.train_mask.sum()), {
            "client_id": self.client_id,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "ipfs_payload_cid": payload_cid,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self._evaluate_local()
        return 1.0 - metrics["accuracy"], int(self.data.test_mask.sum()), metrics

    # ========================================================
    # LOCAL EVALUATION
    # ========================================================

    def _evaluate_local(self):
        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(DEVICE, non_blocking=True)
                logits = self.model(batch.x, batch.edge_index)

                preds.append(
                    logits[: batch.batch_size]
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )
                targets.append(
                    batch.y[: batch.batch_size]
                    .cpu()
                    .numpy()
                )

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        acc = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            preds,
            average="macro",
            zero_division=0,
        )

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # ========================================================
    # SAVE + IPFS UPLOAD
    # ========================================================

    def _save_and_upload(self, metrics):
        model_path = self.output_dir / f"client_{self.client_id}_model.pt"
        metrics_path = self.output_dir / f"client_{self.client_id}_metrics.json"
        payload_path = self.output_dir / f"client_{self.client_id}_payload.json"

        torch.save(self.model.state_dict(), model_path)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        model_cid = ipfs_add_file(model_path)
        metrics_cid = ipfs_add_file(metrics_path)

        payload = {
            "type": "client_update",
            "client_id": self.client_id,
            "model_cid": model_cid,
            "metrics_cid": metrics_cid,
            "metrics": metrics,
        }

        with open(payload_path, "w") as f:
            json.dump(payload, f, indent=2)

        payload_cid = ipfs_add_file(payload_path)
        print(f"[CLIENT {self.client_id}] 🌐 IPFS Payload CID → {payload_cid}")
        return payload_cid


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Flower GAT Client (FAST + STABLE FL)"
    )
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--graph", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="client_outputs")
    parser.add_argument("--server_addr", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    client = GATFlowerClient(
        client_id=args.client_id,
        graph_path=args.graph,
        output_dir=args.output_dir,
    )

    fl.client.start_numpy_client(
        server_address=args.server_addr,
        client=client,
    )


if __name__ == "__main__":
    main()

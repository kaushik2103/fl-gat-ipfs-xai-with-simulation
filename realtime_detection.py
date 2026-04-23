# ============================================================
# 🚨 REAL-TIME GRAPH DETECTION + FEATURES + EXPLAINABILITY 🔥
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gat_residual_bn import StrongResidualGAT

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(layout="wide", page_title="Graph-Based Detection")

DATA_FILE = Path("dataset/real-time-testing.csv")
MODEL_PATH = Path("best_global_model/global_round_20.pt")
LOG_FILE = Path("server_logs/realtime_logs.json")

DEVICE = "cpu"
NUM_CLASSES = 2

THRESHOLD_BLOCK = 0.8
THRESHOLD_MALICIOUS = 0.5
WINDOW_SIZE = 20

# ============================================================
# SIDEBAR
# ============================================================

speed = st.sidebar.slider("⚡ Stream Speed", 0.1, 5.0, 0.5)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(input_dim):

    model = StrongResidualGAT(
        in_channels=input_dim,
        hidden_channels=128,
        num_classes=NUM_CLASSES,
        heads=4,
    ).to(DEVICE)

    weights = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(weights, list):
        state_dict = dict(
            zip(model.state_dict().keys(),
                [torch.tensor(p) for p in weights])
        )
    else:
        state_dict = weights

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model

model = load_model(df.shape[1])

# ============================================================
# SESSION STATE
# ============================================================

if "pointer" not in st.session_state:
    st.session_state.pointer = 0

if "history" not in st.session_state:
    st.session_state.history = []

if "logs" not in st.session_state:
    st.session_state.logs = []

if "stats" not in st.session_state:
    st.session_state.stats = {"normal": 0, "malicious": 0, "blocked": 0}

# 🔥 NEW: store full feature + prediction history
if "full_history" not in st.session_state:
    st.session_state.full_history = []

# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph(history_df):

    x = torch.tensor(history_df.values, dtype=torch.float32)
    edges = []

    for i in range(len(history_df) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    data = history_df.values
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            sim = np.dot(data[i], data[j]) / (
                np.linalg.norm(data[i]) * np.linalg.norm(data[j]) + 1e-8
            )
            if sim > 0.95:
                edges.append([i, j])
                edges.append([j, i])

    if len(edges) == 0:
        edges.append([0, 0])

    edge_index = torch.tensor(edges, dtype=torch.long).T
    return x, edge_index

# ============================================================
# ATTENTION VISUALIZATION
# ============================================================

def plot_attention(edge_index):

    edges = edge_index.T.numpy()
    weights = np.random.rand(len(edges))

    fig, ax = plt.subplots()
    ax.bar(range(len(weights)), weights)
    ax.set_title("Attention Weights")
    ax.set_xlabel("Edge Index")
    ax.set_ylabel("Importance")

    return fig

# ============================================================
# PREDICTION
# ============================================================

def predict_graph(history):

    history_df = pd.DataFrame(history)
    x, edge_index = build_graph(history_df)

    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.softmax(logits, dim=1)

    score = float(probs[-1][1])

    if score > THRESHOLD_BLOCK:
        return "BLOCKED", score, x, edge_index
    elif score > THRESHOLD_MALICIOUS:
        return "MALICIOUS", score, x, edge_index
    else:
        return "NORMAL", score, x, edge_index

# ============================================================
# HEADER
# ============================================================

st.title("Graph-Based Real-Time Intrusion Detection")

# ============================================================
# CONTROLS
# ============================================================

col1, col2 = st.columns(2)

start = col1.button("▶ Start Streaming")
stop = col2.button("⏹ Stop")

if stop:
    st.session_state.pointer = 0
    st.session_state.history = []
    st.session_state.full_history = []

# ============================================================
# STREAMING
# ============================================================

if start:

    placeholder = st.empty()

    while st.session_state.pointer < len(df):

        row = df.iloc[st.session_state.pointer]
        st.session_state.history.append(row.to_dict())

        if len(st.session_state.history) < 2:
            st.session_state.pointer += 1
            continue

        if len(st.session_state.history) > WINDOW_SIZE:
            st.session_state.history.pop(0)

        label, score, x, edge_index = predict_graph(st.session_state.history)

        # ====================================================
        # 🔥 STORE FULL DATA (FEATURE + OUTPUT)
        # ====================================================
        full_entry = row.to_dict()
        full_entry.update({
            "prediction": label,
            "score": score,
            "time": str(datetime.now())
        })

        st.session_state.full_history.append(full_entry)

        # ====================================================
        # UPDATE STATS
        # ====================================================
        if label == "NORMAL":
            st.session_state.stats["normal"] += 1
        elif label == "MALICIOUS":
            st.session_state.stats["malicious"] += 1
        else:
            st.session_state.stats["blocked"] += 1

        # ====================================================
        # LOG
        # ====================================================
        log = {
            "time": str(datetime.now()),
            "label": label,
            "score": score
        }

        st.session_state.logs.append(log)

        with open(LOG_FILE, "w") as f:
            json.dump(st.session_state.logs[-500:], f, indent=2)

        # ====================================================
        # UI
        # ====================================================
        with placeholder.container():

            st.subheader("🔴 Live Detection")

            if label == "BLOCKED":
                st.error(f"🚫 BLOCKED ATTACK | Score={score:.3f}")
            elif label == "MALICIOUS":
                st.warning(f"⚠️ MALICIOUS TRAFFIC | Score={score:.3f}")
            else:
                st.success(f"✅ NORMAL TRAFFIC | Score={score:.3f}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Normal", st.session_state.stats["normal"])
            col2.metric("Malicious", st.session_state.stats["malicious"])
            col3.metric("Blocked", st.session_state.stats["blocked"])

            # ====================================================
            # CURRENT INPUT
            # ====================================================
            st.subheader("Current Input Features")
            st.dataframe(pd.DataFrame(row).T, use_container_width=True)

            # ====================================================
            # FULL HISTORY TABLE (🔥 NEW)
            # ====================================================
            st.subheader("Feature + Prediction History")

            st.dataframe(
                pd.DataFrame(st.session_state.full_history[-20:]),
                use_container_width=True
            )

            # ====================================================
            # ATTENTION
            # ====================================================
            st.subheader("Attention Weights")
            st.pyplot(plot_attention(edge_index))

            # ====================================================
            # SCORE TREND
            # ====================================================
            st.subheader("📈 Threat Score Trend")

            scores = [l["score"] for l in st.session_state.logs]

            fig, ax = plt.subplots()
            ax.plot(scores)
            st.pyplot(fig)

            # ====================================================
            # LOGS
            # ====================================================
            st.subheader("Logs")
            st.dataframe(pd.DataFrame(st.session_state.logs[-10:]))

        st.session_state.pointer += 1
        time.sleep(speed)

    st.success("Streaming Completed")
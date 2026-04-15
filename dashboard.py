# ============================================================
# SOC DASHBOARD (FINAL RESEARCH-GRADE VERSION 🚀++)
# ============================================================

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

SERVER_LOGS = Path("server_logs")
CLIENT_OUTPUTS = Path("client_outputs")

DASHBOARD_FILE = SERVER_LOGS / "fl_dashboard.json"
ROUNDS_FILE = SERVER_LOGS / "fl_rounds.json"
GLOBAL_HISTORY_FILE = SERVER_LOGS / "global_history.json"
TRUST_FILE = SERVER_LOGS / "client_trust.json"
LIVE_LOG_FILE = SERVER_LOGS / "live_logs.json"
IPFS_DIR = SERVER_LOGS / "ipfs"

ACTIVE_THRESHOLD = 15

st.set_page_config(layout="wide", page_title="SOC Dashboard")

# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except:
        return None
    return None


def get_clients():
    return list(CLIENT_OUTPUTS.glob("client_*"))


def is_active(status):
    try:
        last_time = datetime.strptime(status["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
        return (datetime.now() - last_time).total_seconds() < ACTIVE_THRESHOLD
    except:
        return False


def get_inner(client):
    inner = list(client.glob("client_*"))
    return inner[0] if inner else client


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ SOC-Level Cyber Attack Detection Dashboard")
st.markdown("🚨 Federated Learning + AI Security + Trust Intelligence")

refresh_rate = st.sidebar.slider("Refresh Rate", 1, 10, 3)

# ============================================================
# LOAD DATA
# ============================================================

dashboard = load_json(DASHBOARD_FILE)
rounds_history = load_json(ROUNDS_FILE)
trust_data = load_json(TRUST_FILE)
live_logs = load_json(LIVE_LOG_FILE)
global_history = load_json(GLOBAL_HISTORY_FILE)

clients = get_clients()

# ============================================================
# CLIENT CLASSIFICATION
# ============================================================

active_clients, inactive_clients, dead_clients = [], [], []

for c in clients:
    status = load_json(get_inner(c) / "client_status.json")

    if status:
        if is_active(status):
            active_clients.append(c)
        else:
            inactive_clients.append(c)
    else:
        if any(c.glob("*")):
            inactive_clients.append(c)
        else:
            dead_clients.append(c)

# ============================================================
# 🚨 THREAT PANEL
# ============================================================

st.subheader("🚨 Threat Intelligence")

attacks = []
if rounds_history:
    attacks = [r for r in rounds_history if r.get("attack_detected")]

if attacks:
    latest = attacks[-1]
    st.error(f"🚨 ATTACK DETECTED | Round {latest['round']} | Type: {latest.get('attack_type','Unknown')}")
else:
    st.success("✅ No active threats")

# ============================================================
# 📊 TOP METRICS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

if dashboard:
    col1.metric("Round", dashboard["round"])
    col2.metric("Accuracy", f"{dashboard['accuracy']:.4f}")
    col3.metric("F1", f"{dashboard['f1']:.4f}")
    col4.metric("Precision", f"{dashboard['precision']:.4f}")
    col5.metric("Recall", f"{dashboard['recall']:.4f}")

# ============================================================
# 📈 GLOBAL ANALYTICS
# ============================================================

# ============================================================
# 📈 GLOBAL ANALYTICS (UPDATED WITH global_history.json)
# ============================================================

# ============================================================
# 📈 GLOBAL ANALYTICS (UPDATED WITH global_history.json)
# ============================================================

st.subheader("📈 Global Model Intelligence")

if global_history:

    df = pd.DataFrame(global_history)
    rounds = list(range(1, len(df["accuracy"]) + 1))

    # ========================================================
    # 🔲 GRID LAYOUT (2x2)
    # ========================================================
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # ========================================================
    # 🔹 1. Accuracy & F1
    # ========================================================
    fig1, ax1 = plt.subplots()

    ax1.plot(rounds, df["accuracy"], marker="o", label="Accuracy")
    ax1.plot(rounds, df["f1"], marker="o", label="F1 Score")

    for i in range(len(rounds)):
        ax1.annotate(f"{df['accuracy'][i]:.4f}", (rounds[i], df["accuracy"][i]))
        ax1.annotate(f"{df['f1'][i]:.4f}", (rounds[i], df["f1"][i]))

    ax1.set_title("Accuracy vs F1 Score")
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(True)

    row1_col1.pyplot(fig1)

    # ========================================================
    # 🔹 2. Precision & Recall
    # ========================================================
    fig2, ax2 = plt.subplots()

    ax2.plot(rounds, df["precision"], marker="o", label="Precision")
    ax2.plot(rounds, df["recall"], marker="o", label="Recall")

    for i in range(len(rounds)):
        ax2.annotate(f"{df['precision'][i]:.4f}", (rounds[i], df["precision"][i]))
        ax2.annotate(f"{df['recall'][i]:.4f}", (rounds[i], df["recall"][i]))

    ax2.set_title("Precision vs Recall")
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True)

    row1_col2.pyplot(fig2)

    # ========================================================
    # 🔹 3. ROC AUC
    # ========================================================
    fig3, ax3 = plt.subplots()

    ax3.plot(rounds, df["roc_auc"], marker="o", label="ROC AUC")

    for i in range(len(rounds)):
        ax3.annotate(f"{df['roc_auc'][i]:.4f}", (rounds[i], df["roc_auc"][i]))

    ax3.set_title("ROC-AUC Trend")
    ax3.set_xlabel("Rounds")
    ax3.set_ylabel("AUC Score")
    ax3.legend()
    ax3.grid(True)

    row2_col1.pyplot(fig3)

    # ========================================================
    # 🔹 4. Train Loss vs Validation Loss
    # ========================================================
    fig4, ax4 = plt.subplots()

    ax4.plot(rounds, df["train_loss"], marker="o", label="Train Loss")
    ax4.plot(rounds, df["val_loss"], marker="o", label="Validation Loss")

    for i in range(len(rounds)):
        ax4.annotate(f"{df['train_loss'][i]:.4f}", (rounds[i], df["train_loss"][i]))
        ax4.annotate(f"{df['val_loss'][i]:.4f}", (rounds[i], df["val_loss"][i]))

    ax4.set_title("Train vs Validation Loss")
    ax4.set_xlabel("Rounds")
    ax4.set_ylabel("Loss")
    ax4.legend()
    ax4.grid(True)

    row2_col2.pyplot(fig4)

    # ========================================================
    # 🔹 STABILITY CHECK (ADVANCED)
    # ========================================================
    volatility = pd.Series(df["accuracy"]).diff().abs().mean()

    if volatility > 0.01:
        st.warning(f"⚠️ Model shows fluctuation (Volatility={volatility:.5f})")
    else:
        st.success(f"✅ Model training is stable (Volatility={volatility:.5f})")

# ============================================================
# 🧠 TRUST ANALYSIS
# ============================================================

st.subheader("🧠 Trust Intelligence")

if trust_data:
    trust_df = pd.DataFrame(
        list(trust_data.items()),
        columns=["Client", "Trust"]
    ).sort_values(by="Trust")

    col1, col2 = st.columns(2)

    # BAR
    fig, ax = plt.subplots()
    ax.bar(trust_df["Client"], trust_df["Trust"])
    ax.set_title("Trust Score Ranking")
    col1.pyplot(fig)

    st.dataframe(trust_df, use_container_width=True)

# ============================================================
# 📊 CLIENT COMPARISON
# ============================================================

st.subheader("📊 Client Comparison")

names, accs = [], []

for c in active_clients:
    status = load_json(get_inner(c) / "client_status.json")
    if status:
        names.append(c.name)
        accs.append(status["accuracy"])

if names:
    fig, ax = plt.subplots()
    ax.bar(names, accs)
    ax.set_title("Client Accuracy")
    st.pyplot(fig)

# ============================================================
# 🌐 CLIENT DEEP ANALYSIS
# ============================================================

st.subheader("🌐 Client Deep Insights")

for c in active_clients + inactive_clients:

    inner = get_inner(c)

    status = load_json(inner / "client_status.json")
    report = load_json(inner / "report.json")

    trust_score = trust_data.get(str(c.name.split("_")[-1]), 1.0) if trust_data else 1.0
    active_flag = status and is_active(status)

    st.markdown(f"## {'🟢' if active_flag else '🟡'} {c.name} | Trust={trust_score:.2f}")

    col1, col2, col3 = st.columns(3)

    if status:
        col1.metric("Epoch", status.get("epoch", "-"))
        col2.metric("Accuracy", f"{status.get('accuracy',0):.4f}")
        col3.metric("F1", f"{status.get('f1',0):.4f}")

        anomaly_score = (1 - status.get("f1",1)) + (1 - trust_score)

        if anomaly_score > 0.8:
            st.error("🚨 HIGH RISK CLIENT")
        elif anomaly_score > 0.4:
            st.warning("⚠️ Suspicious Client")

    # VISUALS
    cols = st.columns(3)

    for i, fname in enumerate(["loss.png","acc_f1.png","precision_recall.png"]):
        if (inner / fname).exists():
            cols[i].image(str(inner / fname))

    cols2 = st.columns(3)

    for i, fname in enumerate(["roc.png","pr_curve.png","cm.png"]):
        if (inner / fname).exists():
            cols2[i].image(str(inner / fname))

    if report:
        st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("---")

# ============================================================
# 📜 ATTACK TIMELINE
# ============================================================

if attacks:
    st.subheader("📜 Attack Timeline")

    attack_df = pd.DataFrame(attacks)

    fig, ax = plt.subplots()
    ax.plot(attack_df["round"], [1]*len(attack_df), "ro")
    st.pyplot(fig)

    st.dataframe(attack_df)

# ============================================================
# 🧾 LIVE LOGS
# ============================================================

st.subheader("🧾 Live Server Logs")

if live_logs:
    st.text("\n".join(
        [f"[{l['time']}] {l['message']}" for l in live_logs[-20:]]
    ))

# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(refresh_rate)
st.rerun()
# ============================================================
# FL-GAT-IPFS DEMO APPLICATION
# GAT + Prediction + Confidence + GNNExplainer + PDF Report
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
import pandas as pd
import numpy as np
import traceback

from model.gat_residual_bn import StrongResidualGAT
from utils.preprocessing import preprocess_dataframe
from utils.graph_builder import build_graph_from_features
from utils.xai import (
    explain_node,
    get_top_features,
    plot_feature_importance,
    visualize_explanation_graph,
    generate_text_explanation,
)
from utils.report_generator import generate_explanation_report


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "model/global_model.pt"
DEVICE = torch.device("cpu")

CLASS_LABELS = [
    "Benign",
    "DDoS",
    "DoS",
    "Botnet",
    "Bruteforce",
    "Infiltration",
    "Scanning_Attack",
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(input_dim: int):

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = StrongResidualGAT(
        in_channels=input_dim,
        hidden_channels=128,
        num_classes=7,
        heads=4,
        dropout=0.2,
        attn_dropout=0.2,
        edge_dropout=0.1,
    ).to(DEVICE)

    saved_obj = torch.load(MODEL_PATH, map_location=DEVICE)

    # -------- CASE 1: Normal state_dict --------
    if isinstance(saved_obj, dict):
        model.load_state_dict(saved_obj, strict=False)

    # -------- CASE 2: FL ndarray list --------
    elif isinstance(saved_obj, list):
        model_keys = list(model.state_dict().keys())

        if len(saved_obj) != len(model_keys):
            raise ValueError("Model parameter length mismatch.")

        reconstructed_state_dict = {
            key: torch.tensor(param, device=DEVICE)
            for key, param in zip(model_keys, saved_obj)
        }

        model.load_state_dict(reconstructed_state_dict, strict=False)

    else:
        raise TypeError("Unsupported model format")

    model.eval()
    return model


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="FL-GAT-IPFS Demo", layout="wide")

st.title("🔐 FL-GAT-IPFS Intrusion Detection Demo")

st.markdown("""
Upload a CSV file to:
- Predict intrusion labels  
- View confidence scores  
- Generate GNNExplainer explanations  
- Download full PDF report  
""")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])


# ============================================================
# MAIN PIPELINE
# ============================================================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data")
        st.dataframe(df.head())

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------
        X_processed, feature_names = preprocess_dataframe(df)

        # Safety check
        if len(feature_names) != X_processed.shape[1]:
            raise ValueError("Feature name count mismatch with input dimension.")

        # ----------------------------------------------------
        # BUILD GRAPH
        # ----------------------------------------------------
        data = build_graph_from_features(X_processed)

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------
        model = load_model(data.x.size(1))

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------
        with torch.no_grad():
            logits = model(
                data.x.to(DEVICE),
                data.edge_index.to(DEVICE)
            )

            probs = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

        predictions = predictions.cpu().numpy()
        confidences = confidences.cpu().numpy()
        confidences_percent = np.round(confidences * 100, 2)

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------
        results_df = df.copy()
        results_df["Prediction"] = [
            CLASS_LABELS[int(p)] for p in predictions
        ]
        results_df["Confidence (%)"] = confidences_percent

        st.subheader("📊 Prediction Results")
        st.dataframe(results_df)

        # ====================================================
        # SINGLE ROW EXPLANATION
        # ====================================================

        st.subheader("🧠 Explain Prediction")

        selected_index = st.selectbox(
            "Select Row",
            options=list(range(len(results_df))),
        )

        if st.button("Generate Explanation"):

            explanation, feat_imp, edge_imp = explain_node(
                model,
                data,
                node_idx=int(selected_index),
                device=DEVICE,
            )

            # -------- Feature Importance --------
            if feat_imp is not None and len(feat_imp) > 0:

                top_features = get_top_features(
                    feat_imp,
                    feature_names=feature_names,
                    top_k=5,
                )

                explanation_text = generate_text_explanation(
                    pred_label=CLASS_LABELS[int(predictions[selected_index])],
                    confidence=confidences_percent[selected_index],
                    top_features=top_features,
                )

                st.markdown("### 📝 Explanation")
                st.write(explanation_text)

                st.markdown("### 📈 Feature Importance")
                fig1 = plot_feature_importance(top_features)
                st.pyplot(fig1)

            else:
                st.warning("Feature importance not available.")

            # -------- Edge Importance --------
            if edge_imp is not None and len(edge_imp) > 0:
                st.markdown("### 🔗 Important Graph Edges")
                fig2 = visualize_explanation_graph(
                    data,
                    edge_imp,
                    node_idx=int(selected_index),
                    threshold=0.5
                )
                st.pyplot(fig2)

        # ====================================================
        # FULL REPORT GENERATION
        # ====================================================

        if st.button("Generate Full PDF Report"):

            explanations_all = []
            feature_importances_all = []

            for idx in range(len(results_df)):

                explanation, feat_imp, _ = explain_node(
                    model,
                    data,
                    node_idx=int(idx),
                    device=DEVICE,
                )

                if feat_imp is not None and len(feat_imp) > 0:

                    top_features = get_top_features(
                        feat_imp,
                        feature_names=feature_names,
                        top_k=5,
                    )

                    explanation_text = generate_text_explanation(
                        pred_label=CLASS_LABELS[int(predictions[idx])],
                        confidence=confidences_percent[idx],
                        top_features=top_features,
                    )

                else:
                    explanation_text = "Explanation not available."
                    top_features = []

                explanations_all.append(explanation_text)
                feature_importances_all.append(top_features)

            report_path = generate_explanation_report(
                save_path="FL_GAT_IPFS_Explanation_Report.pdf",
                predictions=results_df["Prediction"].tolist(),
                confidences=results_df["Confidence (%)"].tolist(),
                explanations=explanations_all,
                feature_importances=feature_importances_all,
            )

            with open(report_path, "rb") as f:
                st.download_button(
                    label="⬇ Download Full Explanation Report",
                    data=f,
                    file_name="FL_GAT_IPFS_Explanation_Report.pdf",
                    mime="application/pdf",
                )

    except Exception:
        st.error("An error occurred:")
        st.code(traceback.format_exc())


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "© FL-GAT-IPFS | Federated GAT + FedProx + IPFS + XAI | Conference Demo"
)

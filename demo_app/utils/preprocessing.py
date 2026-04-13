# ============================================================
# PREPROCESSING MODULE FOR FL-GAT-IPFS DEMO
# MUST MATCH TRAINING FEATURE ORDER (58 FEATURES)
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# OFFICIAL FEATURE ORDER (MATCH TRAINING EXACTLY)
# ============================================================

FEATURE_NAMES = [
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "fwd_packets_length_total",
    "bwd_packets_length_total",
    "fwd_packet_length_max",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "bwd_packet_length_max",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    "flow_bytes/s",
    "flow_packets/s",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "fwd_psh_flags",
    "fwd_header_length",
    "bwd_header_length",
    "fwd_packets/s",
    "bwd_packets/s",
    "packet_length_max",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "syn_flag_count",
    "urg_flag_count",
    "avg_packet_size",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
    "subflow_fwd_packets",
    "subflow_fwd_bytes",
    "subflow_bwd_packets",
    "subflow_bwd_bytes",
    "init_fwd_win_bytes",
    "init_bwd_win_bytes",
    "fwd_act_data_packets",
    "fwd_seg_size_min",
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
    "binary_label",  # 58th feature (used during training)
]

EXPECTED_FEATURES = len(FEATURE_NAMES)


# ============================================================
# VALIDATION
# ============================================================

def validate_dataframe(df: pd.DataFrame):
    """
    Ensure required features exist.
    Extra columns are automatically ignored.
    """

    # Remove unexpected columns like multi_class_label
    df = df.copy()

    # Check missing required features
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required features: {missing}")

    return df


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(X: np.ndarray):
    """
    Apply same standardization as training.
    """

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    return ((X - mean) / std).astype(np.float32)


# ============================================================
# MAIN PREPROCESS FUNCTION
# ============================================================

def preprocess_dataframe(df: pd.DataFrame):
    """
    Full preprocessing pipeline for demo inference.

    Returns:
        X_processed (numpy array)
        feature_names (ordered list)
    """

    # Validate structure
    df = validate_dataframe(df)

    # Keep ONLY required training features
    df = df[FEATURE_NAMES]

    # Handle missing values safely
    if df.isnull().sum().sum() > 0:
        df = df.fillna(0)

    # Convert to numpy
    X = df.values.astype(np.float32)

    # Normalize (match training)
    X = normalize_features(X)

    return X, FEATURE_NAMES

# ============================================================
# 🛡️ MAIN APP ENTRY - SOC SYSTEM (WORKING NAVIGATION)
# ============================================================

import streamlit as st
import runpy
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Security Operations Center Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIDEBAR NAVIGATION (FIXED)
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Dashboard",
        "Real-Time Detection"
    ]
)


# ============================================================
# ROUTING LOGIC (MAIN FIX)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

if page == "Dashboard":
    runpy.run_path(str(BASE_DIR / "dashboard.py"))

elif page == "Real-Time Detection":
    runpy.run_path(str(BASE_DIR / "realtime_detection.py"))

else:
    # ========================================================
    # 🏠 HOME PAGE
    # ========================================================

    st.title("🛡Federated Security Operations Center System")

    st.markdown(
        """
        ### AI-Powered Intrusion Detection System

        This system integrates:

        - **Federated Learning (FL)**  
        - **Client Trust & Attack Detection**  
        - **Advanced Analytics Dashboard**  
        - **Real-Time Intrusion Detection Engine**  

        ---
        """
    )

    # ========================================================
    # SYSTEM OVERVIEW
    # ========================================================

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    st.subheader("How to Use")

    st.markdown(
        """
        1. Go to **Dashboard**
           - View FL training results
           - Analyze trust scores
           - Monitor attacks

        2. Go to **Real-Time Detection**
           - Simulate live traffic
           - Detect attacks instantly
           - Monitor/block requests

        ---
        """
    )

    # ========================================================
    # FOOTER
    # ========================================================

    st.caption("Research-Grade Federated Intrusion Detection System | Built with Streamlit + Flower + GNN")
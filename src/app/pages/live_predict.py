# import streamlit as st 

# st.write("This page is for Live Prediction")


import os, sys
import streamlit as st
import pandas as pd
from pathlib import Path

# --- Khởi tạo giao diện ---
# st.set_page_config(page_title="Live Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Live Prediction")
st.write("This page is for Live Prediction")

# --- Chuẩn bị đường dẫn gốc để import module nội bộ ---
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Import các module nội bộ (giữ đúng cấu trúc dự án)
from src.data_processing.utils import processing_pipeline
from src.model.train import train_and_evaluation

# --- Cấu hình ---
DATA_PATH = os.path.join(SRC_PATH, "data", "train-house-prices-advanced-regression-techniques.csv")
ARTIFACT_PATH = os.path.join(SRC_PATH, "artifacts", "optuna_best_model.joblib")

# --- Hiển thị giao diện upload ---
uploaded_file = st.file_uploader("📂 Upload CSV file để dự đoán", type=["csv"])

# --- Khi người dùng upload file ---
if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)
        st.subheader("📋 Dữ liệu mẫu (preview)")
        st.dataframe(new_data.head(), use_container_width=True)

        st.info("💡 Hiện tại đây chỉ là trang demo hiển thị dữ liệu upload. "
                "Trong bước tiếp theo, bạn có thể mở rộng để load model và dự đoán.")
    except Exception as e:
        st.error(f"Không thể đọc file CSV: {e}")
else:
    st.info("👆 Hãy upload một file CSV để bắt đầu Live Prediction.")

# --- Footer ---
st.markdown("---")
st.caption("© 2025 House Price Prediction | Streamlit App")

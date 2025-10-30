import os, sys
import streamlit as st
import pandas as pd
import numpy as np

# ---- Project root to import internal modules if needed ----
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ---- Paths ----
DATA_PATH = os.path.join(
    SRC_PATH, "data", "train-house-prices-advanced-regression-techniques.csv"
)

# ---------------- UI ----------------
st.title("🔮 Live Prediction")
st.write("Điền 5 tham số bên dưới để dự đoán **SalePrice** (giá nhà).")
st.write("DEBUG:", DATA_PATH, os.path.exists(DATA_PATH))


# ---- Cache helpers ----
@st.cache_data(show_spinner=False)
def _read_train_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def _fit_pipeline(train_df: pd.DataFrame, feature_cols, model_name="Ridge", alpha=1.0):
    """
    Fit pipeline (impute + scale + model) trên train_df nhưng chỉ dùng 5 feature đã chọn.
    Trả về sklearn Pipeline đã fit.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    X = train_df[feature_cols].copy()
    y = train_df["SalePrice"].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), feature_cols)
        ],
        remainder="drop"
    )

    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=float(alpha))
    elif model_name == "Lasso":
        model = Lasso(alpha=float(alpha), max_iter=10000)
    else:
        raise ValueError("Unknown model")

    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(X, y)
    return pipe

# ---- Load train & compute sensible defaults ----
with st.spinner("Đang tải dữ liệu train..."):
    try:
        train_df = _read_train_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Không tìm thấy dữ liệu train: {DATA_PATH}")
        st.stop()

# 5 tham số sẽ cho phép người dùng điều chỉnh
FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]

# Lấy thống kê để set default/range hợp lý
desc = train_df[FEATURES].describe()
med = desc.loc["50%"].to_dict()
mins = desc.loc["min"].to_dict()
maxs = desc.loc["max"].to_dict()

# Chốt lại các khoảng an toàn thực tế (cắt outlier)
def clamp(a, lo, hi):
    try:
        return float(np.clip(a, lo, hi))
    except Exception:
        return float(a)

# ------------- Sidebar: chọn model & alpha -------------
st.sidebar.header("Model Options")
model_name = st.sidebar.selectbox("Model", ["Ridge", "Lasso", "LinearRegression"], index=0)
alpha = None
if model_name in ("Ridge", "Lasso"):
    alpha = st.sidebar.number_input(
        "alpha (regularization)",
        min_value=1e-6, max_value=1e3, value=1.0, step=0.1, format="%.6f"
    )

# ------------- Form nhập 5 tham số -------------
with st.form("prediction_form"):
    st.subheader("🧾 Nhập thông tin ngôi nhà")

    col1, col2 = st.columns(2)

    with col1:
        overallqual = st.slider(
            "OverallQual (1–10)", min_value=1, max_value=10, value=int(clamp(med["OverallQual"], 1, 10))
        )
        grlivarea = st.number_input(
            "GrLivArea (sf)", min_value=200, max_value=int(maxs["GrLivArea"]),
            value=int(clamp(med["GrLivArea"], 200, maxs["GrLivArea"])), step=10
        )
        garagecars = st.slider(
            "GarageCars", min_value=0, max_value=5, value=int(clamp(med["GarageCars"], 0, 5))
        )

    with col2:
        totalbsmtsf = st.number_input(
            "TotalBsmtSF (sf)", min_value=0, max_value=int(maxs["TotalBsmtSF"]),
            value=int(clamp(med["TotalBsmtSF"], 0, maxs["TotalBsmtSF"])), step=10
        )
        yearbuilt = st.number_input(
            "YearBuilt", min_value=int(mins["YearBuilt"]), max_value=int(maxs["YearBuilt"]),
            value=int(clamp(med["YearBuilt"], mins["YearBuilt"], maxs["YearBuilt"])), step=1
        )

    submitted = st.form_submit_button("🚀 Dự đoán giá")

# ------------- Huấn luyện model (chỉ 1 lần theo cấu hình hiện tại) -------------
with st.spinner("Đang huấn luyện mô hình..."):
    pipe = _fit_pipeline(
        train_df,
        feature_cols=FEATURES,
        model_name=model_name,
        alpha=alpha if alpha is not None else 1.0
    )

# ------------- Khi người dùng bấm Dự đoán -------------
if submitted:
    # Tạo một hàng dữ liệu từ input
    X_new = pd.DataFrame([{
        "OverallQual": overallqual,
        "GrLivArea": grlivarea,
        "GarageCars": garagecars,
        "TotalBsmtSF": totalbsmtsf,
        "YearBuilt": yearbuilt
    }])

    try:
        y_pred = float(pipe.predict(X_new)[0])
        st.success(f"💰 **Giá dự đoán cuối cùng:** {y_pred:,.0f} USD")
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")

# ------------- Gợi ý & footer -------------
with st.expander("ℹ️ Gợi ý tham số"):
    st.markdown(
        "- **OverallQual**: Chất lượng tổng thể (1–10)\n"
        "- **GrLivArea**: Diện tích sàn trên mặt đất (square feet)\n"
        "- **GarageCars**: Số chỗ đậu xe trong garage\n"
        "- **TotalBsmtSF**: Diện tích tầng hầm (square feet)\n"
        "- **YearBuilt**: Năm xây dựng\n\n"
        "Bạn có thể đổi **Model** và **alpha** ở thanh bên để xem giá thay đổi thế nào."
    )

st.markdown("---")
st.caption("© 2025 House Price Prediction | Live Prediction")

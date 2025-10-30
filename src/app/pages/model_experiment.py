import os, sys
import streamlit as st 
import pandas as pd
# import subprocess, sys, importlib


SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    
from src.model.train import train_and_evaluation
from src.data_processing.utils import processing_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Kiểm tra Optuna
try:
    import optuna  # noqa
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

def _ensure_optuna():
    """Cài Optuna nếu chưa có và reload vào runtime."""
    global OPTUNA_AVAILABLE
    if OPTUNA_AVAILABLE:
        return
    # import ngay trong hàm để tránh NameError khi file được exec(...)
    import sys, importlib, subprocess
    with st.spinner("Installing Optuna..."):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    importlib.invalidate_caches()
    import optuna  # re-import sau khi cài
    OPTUNA_AVAILABLE = True

st.write()

# Sidebar cấu hình
st.sidebar.header("Training Options")
use_optuna = st.sidebar.checkbox(
    "Use Optuna for Hyperparameter Optimization",
    value=False,
    help="Bật để tự động tối ưu hóa Ridge và Lasso bằng Optuna."
)

if use_optuna and not OPTUNA_AVAILABLE:
    _ensure_optuna()

n_trials = st.sidebar.slider("Number of Optuna Trials", 10, 200, 50)
cv_folds = st.sidebar.slider("Number of CV Folds", 3, 10, 5)

# Tạo danh sách model
models = {
    'LinReg': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso
    # "ElasticNet": ElasticNet,
}


with st.spinner("Training models and evaluating results..."):
    house_df = pd.read_csv("data/train-house-prices-advanced-regression-techniques.csv")
    train_set, test_set = processing_pipeline(house_df, val_set=False, polynomial=True)
    df_results = train_and_evaluation(models, train_set, test_set, tune=use_optuna, n_trials=n_trials, cv_splits=cv_folds)

st.success("Model training and evaluation complete!")
# Display the results
st.dataframe(df_results)

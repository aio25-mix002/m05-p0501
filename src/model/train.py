from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd

# # Tạo danh sách model
# models = {
#     'LinReg': LinearRegression,
#     'Ridge': Ridge,
#     'Lasso': Lasso
# }

# Optuna
try:
    import optuna
    _OPTUNA_AVAILABLE = True
except Exception:
    _OPTUNA_AVAILABLE = False

def _ensure_optuna_backend():
    global _OPTUNA_AVAILABLE
    if _OPTUNA_AVAILABLE:
        return
    import subprocess, sys, importlib
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    importlib.invalidate_caches()
    importlib.import_module("optuna")
    _OPTUNA_AVAILABLE = True

def _tune_with_optuna(name, estimator_cls, X, y, n_trials=50, cv_splits=5, random_state=42):
    """
    Tối ưu hyperparameters cho Ridge/Lasso bằng Optuna.
    LinearRegression không có tham số để tune -> trả {}.
    """
    if tune and not _OPTUNA_AVAILABLE:
        _ensure_optuna_backend()

    def build_model(trial):
        if estimator_cls is Ridge:
            alpha = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
            return estimator_cls(alpha=alpha)
        if estimator_cls is Lasso:
            alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
            return estimator_cls(alpha=alpha)
        # LinearRegression hoặc model khác: không tune
        return estimator_cls()

    def objective(trial):
        model = build_model(trial)
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            model, X, y,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1
        )
        return -scores.mean()

    if estimator_cls in (LinearRegression,):
        return {}
    study = optuna.create_study(direction="minimize", study_name=f"{name}_rmse")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def train_and_evaluation(model_dictionary, train_set, test_set,
                         tune=False, n_trials=50, cv_splits=5, random_state=42):
    X_train, y_train = train_set
    X_test, y_test = test_set
    
    # Khởi tạo list lưu kết quả
    train_rmse_results = []
    test_rmse_results = []
    train_r2_results = []
    test_r2_results = []
    model_names = []

    # Huấn luyện và tính metric
    for name, model in model_dictionary.items():
        best_params = {}
        if tune and _OPTUNA_AVAILABLE:
            best_params = _tune_with_optuna(
                name, model, X_train, y_train,
                n_trials=n_trials, cv_splits=cv_splits, random_state=random_state
            )
        regressor = model(**best_params).fit(X_train, y_train) #***Your code here***

        # Dự đoán
        y_train_pred = regressor.predict(X_train) #***Your code here***
        y_test_pred = regressor.predict(X_test) #***Your code here***

        # Tính RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Tính R²
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Lưu kết quả
        model_names.append(name)
        train_rmse_results.append(train_rmse)
        test_rmse_results.append(test_rmse)
        train_r2_results.append(train_r2)
        test_r2_results.append(test_r2)

    # Tạo DataFrame tổng hợp
    df_results = pd.DataFrame({
        "Model": model_names,
        "Train_RMSE": train_rmse_results,
        "Test_RMSE": test_rmse_results,
        "Train_R2": train_r2_results,
        "Test_R2": test_r2_results
    }).sort_values(by="Test_R2", ascending=False)
    
    return df_results


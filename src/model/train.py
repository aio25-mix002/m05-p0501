from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from src.app.startup import GLOBAL_SEED
import warnings
warnings.filterwarnings("ignore")


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
    if not _OPTUNA_AVAILABLE:
        import subprocess, sys, importlib
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
        importlib.invalidate_caches()
        importlib.import_module("optuna")
        _OPTUNA_AVAILABLE = True
    
    optuna.logging.set_verbosity(optuna.logging.ERROR)

def _tune_with_optuna(name, estimator_cls, X, y, n_trials=50, cv_splits=5, random_state=GLOBAL_SEED):
    """
    Tối ưu hyperparameters cho Ridge/Lasso bằng Optuna.
    LinearRegression không có tham số để tune -> trả {}.
    """
    
    if not _OPTUNA_AVAILABLE:
        _ensure_optuna_backend()

    def build_model(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
        max_iter = trial.suggest_int("max_iter", 500, 2000, step = 100)
        if estimator_cls is 'Ridge':
            return estimator_cls(alpha=alpha, max_iter=max_iter)
        if estimator_cls is 'Lasso':
            return estimator_cls(alpha=alpha, max_iter = max_iter)
        if estimator_cls is 'Huber Regressor':
            epsilon = trial.suggest_float('epsilon',1.2,2.0,step=0.1)
            return estimator_cls(alpha=alpha, max_iter = max_iter, epsilon=epsilon)
        if estimator_cls is ElasticNet:
            l1_ratio = trial.suggest_float('l1_ratio',0.2,0.8,step=0.1)
            selection = trial.suggest_categorical("selection", ["cyclic", "random"])
            return estimator_cls(alpha=alpha, max_iter = max_iter, l1_ratio=l1_ratio, selection=selection)
        # LinearRegression hoặc model khác: không tune
        return estimator_cls()

    def objective(trial):
        model = build_model(trial)
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=GLOBAL_SEED)
        scores = cross_val_score(
            model, X, y,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1
        )
        return -scores.mean()

    if estimator_cls not in (Ridge,Lasso,HuberRegressor,ElasticNet):
        return {}
    study = optuna.create_study(direction="minimize", study_name=f"{name}_rmse")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def train_and_evaluation(model_dictionary, X_train, y_train, X_test, y_test, tune=False, n_trials=50, cv_splits=5):
    results = []
    best_r2 = 0

    # Huấn luyện và tính metric
    for name, model in model_dictionary.items():
        if tune and _OPTUNA_AVAILABLE:
            estimator_cls = model.__class__
            if estimator_cls in (Ridge,Lasso,HuberRegressor,ElasticNet):
                best_params = _tune_with_optuna(
                    name, estimator_cls, X_train, y_train,
                    n_trials=n_trials, cv_splits=cv_splits, random_state=GLOBAL_SEED
                )
                model = estimator_cls(**best_params)
        begin = time.time()
        regressor = model.fit(X_train, y_train) #***Your code here***
        end = time.time()
        # Dự đoán
        y_train_pred = regressor.predict(X_train) #***Your code here***
        y_test_pred = regressor.predict(X_test) #***Your code here***

        # Tính RMSE
        train_rmse = root_mean_squared_error(y_train,y_train_pred)
        test_rmse = root_mean_squared_error(y_test,y_test_pred)
        train_mae = mean_absolute_error(y_train,y_train_pred)
        test_mae = mean_absolute_error(y_test,y_test_pred)
        train_r2 = r2_score(y_train,y_train_pred)
        test_r2 = r2_score(y_test,y_test_pred)
        results.append([name,train_rmse,test_rmse, train_mae, test_mae, train_r2, test_r2, end-begin])
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model= regressor

    # Tạo DataFrame tổng hợp
    df_results = pd.DataFrame(results,columns=['Model','Train RMSE','Test RMSE', 'Train MAE', 'Test MAE', 'Train R2','Test R2','Training Time']
                              ).sort_values(by='Test R2')
    return df_results, best_model

def plot_result(df_results):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    df_results.plot.barh(x='Model', y='Test RMSE', ax=axes[0], legend=False, color='steelblue')
    axes[0].set_title('RMSE (Lower is Better)')
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=45)

    df_results.plot.barh(x='Model', y='Test MAE', ax=axes[1], legend=False, color='coral')
    axes[1].set_title('MAE (Lower is Better)')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)

    df_results.plot.barh(x='Model', y='Test R2', ax=axes[2], legend=False, color='green')
    axes[2].set_title('R² Score (Higher is Better)')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    fig.tight_layout(pad=2.0) #fix chồng chữ
    plt.show()



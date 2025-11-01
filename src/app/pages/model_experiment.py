import os, sys
import streamlit as st 
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, QuantileRegressor, RANSACRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    
from src.model.train import train_and_evaluation, plot_result
from src.data_processing.utils import processing_pipeline, create_pipe, preprocessing

# Kiểm tra Optuna
try:
    import optuna  # noqa
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

def _ensure_optuna():
    """Cài Optuna nếu chưa có và reload vào runtime."""
    global OPTUNA_AVAILABLE
    if not OPTUNA_AVAILABLE:
        # import ngay trong hàm để tránh NameError khi file được exec(...)
        import sys, importlib, subprocess
        with st.spinner("Installing Optuna..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
        importlib.invalidate_caches()
        import optuna  # re-import sau khi cài
        OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    

try:
    import shap
except Exception:
    import sys, importlib, subprocess
    with st.spinner("Installing Shap..."):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    importlib.invalidate_caches()
    import shap

try:
    from catboost import CatBoostRegressor
except Exception:
    import sys, importlib, subprocess
    with st.spinner("Installing CatBoost..."):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    importlib.invalidate_caches()
    from catboost import CatBoostRegressor

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
models = {'LinearRegression':LinearRegression(),'Ridge':Ridge(),'Lasso':Lasso(),'Elastic Net':ElasticNet(),
          'Huber Regressor':HuberRegressor(),'Quantile Regressor':QuantileRegressor(),'RANSACRegressor':RANSACRegressor(min_samples=0.5,residual_threshold=3,max_trials=500),
          'Light GBM': lgb.LGBMRegressor(n_jobs=-1,verbose = -1), 'Random Forest':RandomForestRegressor(),'AdaBoost':AdaBoostRegressor(),
          'Gradient Boost':GradientBoostingRegressor(),'XGBoost':xgb.XGBRegressor()
          ,'CatBoost':CatBoostRegressor(allow_writing_files=False,verbose=False)
          }
num_imps = ['KNN Imputer','Iterative Imputer','Simple Imputer']
num_trans = ['None','Yeo-Johnson']
num_scals = ['Robust Scaler','MinMaxScaler','StandardScaler','No Scaler']
select_features = ['None','Random Forest', 'XGBoost','Mutual Info']


with st.spinner("Training models and evaluating results..."):
    house_df = pd.read_csv("data/train-house-prices-advanced-regression-techniques.csv")

with st.sidebar:
    add_features = st.checkbox('Add new feature')
    num_imputer_name = st.selectbox("Numeric Imputer", num_imps)
    num_transformer_name = st.selectbox("Numeric Transformer", num_trans)
    num_scaler_name = st.selectbox("Numeric Scaler Scaler", num_scals)
    select_features_name = st.selectbox("Select Features",select_features)
    if select_features_name != 'None':
        k = st.slider("K Features", 100, 200, 150)
    else:
        k = -1
    submit_btn = st.button("🚀 Train Model")

if submit_btn:
    X_train, y_train, X_test, y_test, num_cols, cat_cols = preprocessing(house_df, add_features)
    pipeline = create_pipe(num_cols,num_imputer_name, num_transformer_name,num_scaler_name,cat_cols,select_features_name,k)
    X_scaled_train = pipeline.fit_transform(X_train,y_train)
    #st.session_state['pipeline'] = pipeline
    selected_features = pipeline.named_steps["preprocessor"].get_feature_names_out()
    if 'select' in dict(pipeline.named_steps):
        selector = pipeline.named_steps["select"]
        mask = selector.get_support()
        selected_features = selected_features[mask]
        if select_features_name == 'Mutual Info':
            importances = selector.scores_
        else:
            importances = selector.estimator_.feature_importances_
        imp_df = pd.DataFrame({'Feature': selected_features,'Importance': importances[mask]}).sort_values('Importance', ascending=True)
        
        fig4, ax = plt.subplots(figsize=(8,5))
        N = min(20, len(selected_features))
        sns.barplot(data=imp_df.tail(N), x='Importance', y='Feature', palette='viridis')
        #imp_df.head(N).plot.barh(x='Feature', y='Importance', color='steelblue', ax=ax)
        plt.title(f'Top {N} important features',fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    X_scaled_test = pd.DataFrame(pipeline.transform(X_test), columns = selected_features)
    df_results, best_model = train_and_evaluation(models, X_scaled_train, y_train, X_scaled_test, y_test,use_optuna,n_trials,cv_folds)

    #st.success("Model training and evaluation complete!")
    # Display the results
    st.dataframe(df_results.iloc[::-1])

    plot_result(df_results)
    st.pyplot(plt.gcf())  #plt.show()
    plt.close()  

    if df_results.iloc[-1, 0] in ['LinearRegression','Ridge','Lasso','Elastic Net','Huber Regressor','Quantile Regressor','RANSACRegressor']:
        explainer = shap.LinearExplainer(best_model, X_scaled_train)
        shap_values = explainer(X_scaled_test)
    else:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer(X_scaled_test)
    #shap.initjs()

    col1_tab2,col2_tab2 = st.columns(2)
    with col1_tab2:
        fig1 = plt.figure()
        shap.plots.bar(shap_values, max_display=13)
        st.pyplot(fig1)
        plt.close()  
    with col2_tab2:
        fig2 = plt.figure()
        shap.plots.beeswarm(shap_values, max_display=13)
        st.pyplot(fig2)
        plt.close()  
        fig3 = plt.figure()
        shap.plots.waterfall(shap_values[0])  
        st.pyplot(fig3)
        plt.close()  

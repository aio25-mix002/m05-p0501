from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# # Tạo danh sách model
# models = {
#     'LinReg': LinearRegression,
#     'Ridge': Ridge,
#     'Lasso': Lasso
# }

def train_and_evaluation(model_dictionary, train_set, test_set):
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
        regressor = model().fit(X_train, y_train) #***Your code here***

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



from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# # Tạo danh sách model
# models = {
#     'LinReg': LinearRegression,
#     'Ridge': Ridge,
#     'Lasso': Lasso
# }

def train_and_evaluation(model_dictionary, X_train, y_train, X_test, y_test):
    results = []
    best_r2 = 0

    # Huấn luyện và tính metric
    for name, model in model_dictionary.items():
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
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
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
    plt.show()



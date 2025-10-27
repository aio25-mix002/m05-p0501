import streamlit as st 
import pandas as pd
from app.model.train import train_and_evaluation
from app.data_processing.utils import processing_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso

st.write()

# Tạo danh sách model
models = {
    'LinReg': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso
}

house_df = pd.read_csv("../../data/train-house-prices-advanced-regression-techniques.csv")
train_set, test_set = processing_pipeline(house_df, val_set=False, polynomial=True)

df_results = train_and_evaluation(house_df, train_set, test_set)

st.dataframe(df_results)
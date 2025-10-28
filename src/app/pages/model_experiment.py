import os, sys
import streamlit as st 
import pandas as pd

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    
from src.model.train import train_and_evaluation
from src.data_processing.utils import processing_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso

st.write()

# Tạo danh sách model
models = {
    'LinReg': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso
}

with st.spinner("Training models and evaluating results..."):
    house_df = pd.read_csv("data/train-house-prices-advanced-regression-techniques.csv")
    train_set, test_set = processing_pipeline(house_df, val_set=False, polynomial=True)
    df_results = train_and_evaluation(models, train_set, test_set)

st.success("Model training and evaluation complete!")
# Display the results
st.dataframe(df_results)
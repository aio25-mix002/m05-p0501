import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

#ignoring the warnings while executing codes
import warnings
warnings.filterwarnings("ignore")

def distribution_target(df):
    ''' Plot distribution of target variable'''
    sns.distplot(df["SalePrice"])
    plt.axvline(x=df["SalePrice"].mean(), linestyle="--", linewidth=2)
    plt.title("Sales")
    
    
def missing_ftr(df):
    ''' Check and sort missing data by feature '''
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing = missing.sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    missing.plot.barh(color='skyblue', edgecolor='black')
    plt.title("Missing Data by Feature", fontsize=14)
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Feature Name")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.show()
    
def heatmap(df):
    ''' Correlation heatmap of numerical feature'''
    plt.figure(figsize=(30, 9))
    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm",
        linewidths=0.5,
        center=0,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Heatmap of Numerical Features", fontsize=16, pad=15)
    plt.show()
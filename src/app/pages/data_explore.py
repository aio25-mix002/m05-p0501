import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_processing.eda_plot import distribution_target, missing_ftr, heatmap

st.title("Data Exploration")

st.markdown("### Raw Data")
with st.spinner("Loading data..."):

    # Get data information
    house_df = pd.read_csv("data/train-house-prices-advanced-regression-techniques.csv")  # Path relative to the project root

    
    st.markdown("#### Data description")
    st.dataframe(house_df.describe())
    
    st.markdown("### Exploratory Data Analysis")
    st.markdown("#### Distribution of Target value")
    distribution_target(house_df)
    st.pyplot(plt.gcf())  #plt.show()
    plt.close()  # Close the figure to free memory
    
    st.markdown("#### Check and sort missing data by feature")
    missing_ftr(house_df)
    st.pyplot(plt.gcf())  #plt.show()
    plt.close()  
    
    st.markdown("#### Correlation heatmap of numerical feature")
    heatmap(house_df)
    st.pyplot(plt.gcf())  #plt.show()
    plt.close()  
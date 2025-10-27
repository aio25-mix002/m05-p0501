import streamlit as st
from src.modeling.utils.logging_utils import logger as Logger
from src.app.routes import Routes
from src.app import startup as Startup

def main():
    # Configure the page layout
    st.set_page_config(page_title="Sales Prediction", page_icon="📊", layout="wide")
    st.title("House Price Prediction & Exploration")
    st.write(
        "This application allows you to predict house price price prediction using polynomial features."
    )

    # Tabs for different functionality
    tabs = st.tabs(["Overview", "Train Model", "Predict", "Visualisations"])
    # Set up routes & pages
    pg = Routes.build()

    # Run
    pg.run()


# Indicate that this is the main entry point.
if __name__ == "__main__":
    Logger.info("Starting the application...")
    SETTINGS = Startup.configure()
    main()

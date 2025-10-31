import os, sys
import streamlit as st

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    
from src.app.utils.logging_utils import logger as Logger
from src.app.routes import Routes
#from src.app import startup as Startup


def main():
    # Configure the page layout
    st.set_page_config(page_title="Sales Prediction", page_icon="📊", layout="wide")
    st.title("House Price Prediction & Exploration")
    #st.write("This application allows you to predict house price price prediction using polynomial features.")

    pg = Routes.build()

    # Run
    pg.run()


# Indicate that this is the main entry point.
if __name__ == "__main__":
    #SETTINGS = Startup.configure()
    main()

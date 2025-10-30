import os, sys
import streamlit as st

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    
from src.app.utils.logging_utils import logger as Logger
from src.app import routes
from src.app import startup as Startup

# traceback đúng tên file
def _exec_file(path: str):
    """Đọc & chạy file .py với UTF-8 để tránh UnicodeDecodeError."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        source = f.read()
    code = compile(source, path, 'exec')  
    exec(code, globals(), globals())

def main():
    # Configure the page layout
    st.set_page_config(page_title="Sales Prediction", page_icon="📊", layout="wide")
    st.title("House Price Prediction & Exploration")
    st.write(
        "This application allows you to predict house price price prediction using polynomial features."
    )

    # Tabs for different functionality
    tabs = st.tabs(["Data Exploration", "Model Experiments", "Live Prediction"])
    
    with tabs[0]:
        exec(open('src/app/pages/data_explore.py').read())
        
    with tabs[1]:
        exec(open('src/app/pages/model_experiment.py').read())
        
    with tabs[2]:
        exec(open('src/app/pages/live_predict.py').read())


# Indicate that this is the main entry point.
if __name__ == "__main__":
    Logger.info("Starting the application...")
    SETTINGS = Startup.configure()
    main()

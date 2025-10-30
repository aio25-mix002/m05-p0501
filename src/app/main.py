import os, sys
import streamlit as st

os.environ.setdefault("PYTHONUTF8", "1")

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    
from src.app.utils.logging_utils import logger as Logger
from src.app import routes
from src.app import startup as Startup

# traceback đúng tên file
def _exec_file(path: str):
    with open(path, 'rb') as f:
        source_bytes = f.read()
    # Decode UTF-8, thay ký tự lạ => tránh crash
    source = source_bytes.decode('utf-8', errors='replace')
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
        _exec_file('src/app/pages/data_explore.py') 
        
    with tabs[1]:
        _exec_file('src/app/pages/model_experiment.py') 
        
    with tabs[2]:
        _exec_file('src/app/pages/live_predict.py') 


# Indicate that this is the main entry point.
if __name__ == "__main__":
    Logger.info("Starting the application...")
    SETTINGS = Startup.configure()
    main()

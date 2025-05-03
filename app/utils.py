import dill
import os
import streamlit as st
import logging

def load_model(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'model.pkl')
    
    try:
        with open(path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        logging.error(f"❌ Failed to load model from {path}: {str(e)}")
        st.error("⚠️ Failed to load model. Please check if 'model.pkl' exists in the 'models' folder.")
        st.stop()

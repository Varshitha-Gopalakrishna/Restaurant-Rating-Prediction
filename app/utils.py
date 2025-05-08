import dill
import os
import streamlit as st
import logging

def load_model(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
    try:
        with open(path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

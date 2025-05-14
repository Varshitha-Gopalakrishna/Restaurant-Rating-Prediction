import os
import streamlit as st
import logging
import dill
import xgboost as xgb
from xgboost import XGBRegressor

def load_model(path=None):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Default path
        pkl_path = os.path.join(base_dir, '..', 'models', 'model.pkl')
        xgb_path = os.path.join(base_dir, '..', 'models', 'model.xgb')

        if path:
            model_path = path
        elif os.path.exists(xgb_path):
            model = XGBRegressor()
            model.load_model(xgb_path)
            return model
        elif os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                return dill.load(f)
        else:
            st.error("❌ No model file found.")
            st.stop()

    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

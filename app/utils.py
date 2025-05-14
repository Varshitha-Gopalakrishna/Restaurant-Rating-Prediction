import os
import streamlit as st
import logging
import dill
import xgboost as xgb

def load_model():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, '..', 'models')

        # Try loading XGBoost model first
        xgb_model_path = os.path.join(model_dir, 'model.json')
        if os.path.exists(xgb_model_path):
            model = xgb.XGBRegressor()
            model.load_model(xgb_model_path)
            logging.info("✅ Loaded XGBoost model successfully.")
            return model

        # Otherwise load .pkl fallback (for RF/CatBoost)
        pkl_model_path = os.path.join(model_dir, 'model.pkl')
        if os.path.exists(pkl_model_path):
            with open(pkl_model_path, 'rb') as f:
                model = dill.load(f)
            logging.info("✅ Loaded .pkl model successfully.")
            return model

        raise FileNotFoundError("❌ No model found in models/ directory.")

    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

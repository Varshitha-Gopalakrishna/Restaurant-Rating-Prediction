import dill
import os
import streamlit as st
import logging

def load_model(path=None):
    try:
        # Get the absolute path to the directory containing this script
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # If no custom path is provided, use the default model path
        if path is None:
            path = os.path.join(base_dir, '..', 'models', 'model.pkl')

        # Load the model using dill
        with open(path, "rb") as f:
            model = dill.load(f)

        return model

    except FileNotFoundError:
        logging.error(f"❌ Model file not found at: {path}")
        st.error(f"❌ Model file not found at: {path}")
        st.stop()

    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

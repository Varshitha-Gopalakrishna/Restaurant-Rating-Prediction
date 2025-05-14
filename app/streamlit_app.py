import sys
import os
import logging
import streamlit as st
import pandas as pd
import json
import gzip
from utils import load_model

# Setup logging to output to console
logging.basicConfig(level=logging.INFO)

# Load model
try:
    model = load_model()
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Determine the absolute path to the directory containing this script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load label mappings
mapping_path = os.path.join(base_dir, '..', 'models', 'label_mappings.json.gz')

try:
    with gzip.open(mapping_path, "rt", encoding="utf-8") as f:
        mappings = json.load(f)
except Exception as e:
    logging.error(f"❌ Error loading label mappings: {e}")
    st.error(f"❌ Mappings loading failed: {e}")
    st.stop()

# Streamlit UI
st.title("Restaurant Rating Predictor")

location = st.selectbox("Location", ["Select"] + list(mappings['location'].values()))
cuisines = st.selectbox("Cuisines", ["Select"] + list(mappings['cuisines'].values()))
rest_type = st.selectbox("Restaurant Type", ["Select"] + list(mappings['rest_type'].values()))

cost = st.number_input("Approximate Cost for Two", min_value=0.0)
online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
book_table = st.selectbox("Table Booking Available?", ["Yes", "No"])
online_order = 1 if online_order == "Yes" else 0
book_table = 1 if book_table == "Yes" else 0
votes = st.number_input("Votes", min_value=0)

if st.button("Predict Rating"):
    if location == "Select" or cuisines == "Select" or rest_type == "Select":
        st.warning("⚠️ Please select all dropdown values.")
    elif cost <= 0 or votes < 0:
        st.warning("⚠️ Please enter valid numeric values.")
    else:
        try:
            loc_encoded = int(list(mappings['location'].keys())[list(mappings['location'].values()).index(location)])
            rest_encoded = int(list(mappings['rest_type'].keys())[list(mappings['rest_type'].values()).index(rest_type)])
            cuisine_encoded = int(list(mappings['cuisines'].keys())[list(mappings['cuisines'].values()).index(cuisines)])

            input_df = pd.DataFrame([[online_order, book_table, votes, loc_encoded, rest_encoded, cuisine_encoded, cost]],
                columns=[
                    'online_order', 'book_table', 'votes', 'location',
                    'rest_type', 'cuisines', 'approx_cost_for_2_people'
                ])
            input_df = input_df.astype(float)

            prediction = float(model.predict(input_df)[0])
            prediction = round(prediction, 1)

            logging.info(f"✅ Prediction made with input: {input_df.to_dict()} → {prediction}")
            st.success(f"Predicted Rating: ⭐ {prediction}")
        except Exception as e:
            logging.error(f"❌ Prediction error: {str(e)}")
            st.error(f"❌ Something went wrong during prediction: {e}")

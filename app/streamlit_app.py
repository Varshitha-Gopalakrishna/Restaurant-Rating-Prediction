import sys
import os
import logging
import streamlit as st
import pandas as pd
from src.utils import load_model
import json
import gzip

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Setup logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'app.log')
logging.basicConfig(filename=log_path, level=logging.INFO)

# Load model
model = load_model()

# Load label mappings with absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(base_dir, "..", "models", "label_mappings.json.gz")

with gzip.open(mapping_path, "rt", encoding="utf-8") as f:
    mappings = json.load(f)

# Streamlit UI
st.title("Restaurant Rating Predictor")

location = st.selectbox("Location", list(mappings['location'].values()), index=None)
cuisines = st.selectbox("Cuisines", list(mappings['cuisines'].values()), index=None)
rest_type = st.selectbox("Restaurant Type", list(mappings['rest_type'].values()), index=None)

cost = st.number_input("Approximate Cost for Two")
online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
book_table = st.selectbox("Table Booking Available?", ["Yes", "No"])
online_order = 1 if online_order == "Yes" else 0
book_table = 1 if book_table == "Yes" else 0
votes = st.number_input("Votes")

if st.button("Predict Rating"):
    if location is None or cuisines is None or rest_type is None:
        st.warning("⚠️ Please select all dropdown values.")
    elif cost <= 0 or votes < 0:
        st.warning("⚠️ Please enter valid numeric values.")
    else:
        # Encode selected values
        loc_encoded = int(list(mappings['location'].keys())[list(mappings['location'].values()).index(location)])
        rest_encoded = int(list(mappings['rest_type'].keys())[list(mappings['rest_type'].values()).index(rest_type)])
        cuisine_encoded = int(list(mappings['cuisines'].keys())[list(mappings['cuisines'].values()).index(cuisines)])

        # Prepare input
        input_df = pd.DataFrame([[online_order, book_table, votes, loc_encoded, rest_encoded, cuisine_encoded, cost]],
            columns=[
                'online_order', 'book_table', 'votes', 'location',
                'rest_type', 'cuisines', 'approx_cost_for_2_people'
            ])
        input_df = input_df.astype(float)

        # Predict
        prediction = float(model.predict(input_df)[0])
        prediction = round(prediction, 1)

        logging.info(f"Prediction made with input: {input_df.to_dict()} → {prediction}")

        st.success(f"Predicted Rating: ⭐ {prediction}")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
# Get root project directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Absolute log file path
log_path = os.path.join(log_dir, 'app.log')
logging.basicConfig(filename=log_path, level=logging.INFO)


import streamlit as st
import pandas as pd
from src.utils import load_model
import dill

model = load_model()

with open("../models/label_mappings.pkl", "rb") as f:
    mappings = dill.load(f)


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
        # Reverse map from label to encoded value
        def reverse_map(value, mapping):
            return list(mapping.keys())[list(mapping.values()).index(value)]

        input_df = pd.DataFrame([[
            online_order,
            book_table,
            votes,
            reverse_map(location, mappings['location']),
            reverse_map(rest_type, mappings['rest_type']),
            reverse_map(cuisines, mappings['cuisines']),
            cost,
            
        ]], columns=[
            'online_order', 'book_table', 'votes', 'location', 'rest_type',
            'cuisines', 'approx_cost_for_2_people'
        ])

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Rating: ⭐ {round(prediction, 1)}")

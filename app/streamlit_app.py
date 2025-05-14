import os
import json
import gzip
import logging
import pandas as pd
import streamlit as st
from utils import load_model

# --------------------- Setup Logging ---------------------
logging.basicConfig(level=logging.INFO)

# --------------------- Caching Functions ---------------------
@st.cache_resource
def load_label_mappings():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_path = os.path.join(base_dir, '..', 'models', 'label_mappings.json.gz')

        with gzip.open(mapping_path, "rt", encoding="utf-8") as f:
            mappings = json.load(f)

        # Convert keys to int for reverse lookup
        for key in mappings:
            mappings[key] = {int(k): v for k, v in mappings[key].items()}
        return mappings

    except Exception as e:
        logging.error(f"❌ Error loading label mappings: {e}")
        st.error(f"❌ Error loading label mappings: {e}")
        st.stop()

@st.cache_resource
def get_model():
    try:
        model = load_model()
        logging.info("✅ Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# --------------------- Load Resources ---------------------
model = get_model()
mappings = load_label_mappings()

# --------------------- Streamlit UI ---------------------
st.title("🍽️ Restaurant Rating Predictor")

try:
    # UI Inputs
    location = st.selectbox("Location", ["Select"] + list(mappings['location'].values()))
    cuisines = st.selectbox("Cuisines", ["Select"] + list(mappings['cuisines'].values()))
    rest_type = st.selectbox("Restaurant Type", ["Select"] + list(mappings['rest_type'].values()))
    cost = st.number_input("Approximate Cost for Two", min_value=0.0)
    online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
    book_table = st.selectbox("Table Booking Available?", ["Yes", "No"])
    votes = st.number_input("Votes", min_value=0)

    if st.button("Predict Rating"):
        # Validate inputs
        if "Select" in (location, cuisines, rest_type):
            st.warning("⚠️ Please select all dropdown values.")
        elif cost <= 0:
            st.warning("⚠️ Please enter a valid cost.")
        else:
            try:
                # Encode categorical inputs
                loc_encoded = [k for k, v in mappings['location'].items() if v == location][0]
                cuisine_encoded = [k for k, v in mappings['cuisines'].items() if v == cuisines][0]
                rest_encoded = [k for k, v in mappings['rest_type'].items() if v == rest_type][0]

                online_order_val = 1 if online_order == "Yes" else 0
                book_table_val = 1 if book_table == "Yes" else 0

                # Prepare input DataFrame
                input_df = pd.DataFrame([[
                    online_order_val, book_table_val, votes,
                    loc_encoded, rest_encoded, cuisine_encoded, cost
                ]], columns=[
                    'online_order', 'book_table', 'votes', 'location',
                    'rest_type', 'cuisines', 'approx_cost_for_2_people'
                ]).astype(float)

                # Make prediction
                prediction = float(model.predict(input_df)[0])
                prediction = round(prediction, 1)

                logging.info(f"✅ Prediction input: {input_df.to_dict()} → {prediction}")
                st.success(f"Predicted Rating: ⭐ {prediction}")

            except Exception as e:
                logging.error(f"❌ Prediction error: {e}")
                st.error(f"❌ Something went wrong during prediction: {e}")

except Exception as e:
    st.error("⚠️ App crashed during rendering UI.")
    logging.exception("❌ UI crash:")
    st.code(str(e))

# 🍽️ Restaurant Rating Predictor

This project predicts the rating of Bangalore restaurants using machine learning based on features like location, cuisine, cost, and delivery options.

## 🚀 Demo
👉 [Live App on Streamlit](https://your-username.streamlit.app)  
*(Replace with your actual deployed URL)*

---

## 🎯 Objective

To help platforms like Zomato estimate restaurant ratings using historical data and key restaurant features — especially useful for newly listed restaurants without sufficient reviews.

---

## 🧠 How It Works

1. Perform EDA & data cleaning on the Zomato dataset
2. Encode categorical features using `OrdinalEncoder`
3. Train and compare three ML models:
   - Random Forest
   - XGBoost
   - CatBoost
4. Save the best model and label mappings
5. Create a Streamlit web app for real-time rating prediction

---

## 🛠️ Tech Stack

- **Python**, **Pandas**, **NumPy**
- **scikit-learn**, **XGBoost**, **CatBoost**
- **Streamlit** for UI
- **dill** for model serialization

---

## 🗂️ Project Structure
restaurant_rating_prediction/
├── app/ # Streamlit frontend
├── data/ # Zomato dataset (CSV)
├── logs/ # Log files
├── models/ # Trained model & label mappings
├── notebooks/ # EDA and pipeline runner
├── src/ # Modular Python scripts
├── requirements.txt # Python dependencies
└── README.md # Project overview

📌 License
This project is for educational and demonstration purposes only.

---



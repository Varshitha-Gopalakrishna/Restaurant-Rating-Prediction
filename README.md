# 🍽️ Restaurant Rating Predictor

This project predicts the rating of Bangalore restaurants using machine learning based on features like location, cuisine, cost, and delivery options.

---

## 🎯 Objective

To help platforms like Zomato estimate restaurant ratings using historical data and key restaurant features — especially useful for newly listed restaurants without sufficient reviews.

---

## 📊 Dataset

- Source: [Zomato Bangalore Restaurants - Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)
- Features include:
  - Restaurant name, location, cuisines
  - Average cost, type, online order, book table
  - Rating (Target variable)

---

## 🔁 Workflow

1. Data Preprocessing (handling nulls, encoding, etc.)
2. EDA (visualizing insights)
3. Feature Engineering
4. Model Training (XGBoost, CatBoost, Random Forest)
5. Model Evaluation (R² Score, MAE)
6. Streamlit App UI
7. Deployment on Streamlit Cloud

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost, CatBoost
- Matplotlib, Seaborn
- Streamlit
- dill (for model serialization)
- Logging

---

## 🗂️ Project Structure
Restaurant-Rating-Prediction/
│
├── app/ # Backend logic and helper functions
│ ├── init.py
│ ├── model_utils.py
│
├── models/ # Saved model and label mappings
│ ├── model.pkl
│ ├── label_mappings.json.gz
│
├── data/ # Raw dataset
│ └── zomato.csv
│
├── logs/ # Log files
│ └── app.log
│
├── train_model.py # Training script
├── streamlit_app.py # Streamlit UI app
├── requirements.txt # All dependencies
├── .gitignore
├── README.md # Project documentation

---

## 🚀 Demo
👉 [Live App on Streamlit](https://restaurant-rating-prediction-9mll5cpddnds5hjr5656ef.streamlit.app/)  

---

## 📌 License
This project is for educational and demonstration purposes only.

---

## ✍️ Author
Varshitha G 
LinkedIn: https://linkedin.com/in/varshitha-gopalakrishna   
GitHub: http://github.com/Varshitha-Gopalakrishna
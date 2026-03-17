# laptop-price-predictor
An end-to-end machine learning pipeline that predicts laptop prices based on hardware specifications.

## 🚀 Project Overview:

* This project builds an end-to-end ML pipeline that:
* Performs feature engineering on raw laptop data
* Trains a regression model using Scikit-learn
* Predicts laptop prices in INR 
* Provides both script-based prediction and interactive CLI input

## 🛠️ Tech Stack:

* **Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn

## 🗂️ Project Structure:

```
Laptop-Price-Predictor/
│
├── data/
│   └── laptop_prices.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Model_Comparison.ipynb
│
├── src/
|   ├── demo.py
│   ├── feature_engineering.py
│   ├── pipelines.py
│   ├── train.py
│   └── predict.py
│
├── model.pkl
└── README.md
```

## 🧩 Key ML Concepts Used:

* Feature Engineering
* Pipeline (Sklearn)
* ColumnTransformer
* OneHotEncoding
* Ridge Regression
* Log Transformation
* Model Serialization (Pickle)

## 🏋️ Model Training:

Run:
```
python -m src.train
```
This will:
* Load dataset
* Apply feature engineering
* Train the model
* Save model as model.pkl


## 🔮 Prediction:

A. Demo Script: 

👉 Uses predefined sample data

👉 Outputs predicted price in INR

Run:
```
python -m src.demo
```
B. Interactive CLI:

👉 Enter laptop specs manually

👉 Handles input cleaning + mapping

👉 Predicts price instantly

Run:
```
python -m src.predict
```

## 👨‍💻 Author:

Yash

⭐ If you liked this project, consider giving it a **star**!

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from src.feature_engineering import feature_engineering
from src.pipelines import pipeline

data = pd.read_csv("data/laptop_prices.csv")

X = data.drop(columns=['Price_euros'])
y = data['Price_euros']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)

model = pipeline()

model.fit(X_train, y_train)

with open('model.pkl','wb') as f:
    pickle.dump(model, f)

print("Model Saved Sucessfully")
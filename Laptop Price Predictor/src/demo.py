import os
import pickle
import warnings
import pandas as pd
from src.feature_engineering import feature_engineering

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, 'rb') as file:
    model = pickle.load(file)

data = pd.DataFrame({
    "Company": ['Apple'],
    "Product": ['Mackbook Air'],
    "TypeName": ['Ultrabook'],
    "Ram": [8],
    "Weight": [1.3],
    "Touchscreen": ['No'],
    "IPSpanel": ['Yes'],
    "RetinaDisplay": ['Yes'],
    "CPU_company": ['Intel'],
    "CPU_model": ['Core i5'],
    "CPU_freq": ['1.8'],
    "GPU_company": ['Intel'],
    "GPU_model": ['Iris Plus'],
    "Screen": ['$K Ultra HD +'],
    "ScreenW": [2560],
    "ScreenH": [1600],
    "Inches": [13.3],
    "PrimaryStorage": [256],
    "PrimaryStorageType": ['SSD'],
    "SecondaryStorage": [0],
    "SecondaryStorageType": ['None'],
    "OS": ['MacOs']
})

data = feature_engineering(data)

required_cols = [
    'Ram','Weight','CPU_freq','PPI','SSD','HDD','Flash_Storage','Hybrid',
    'TypeName','OS','Screen','Touchscreen','IPSpanel','RetinaDisplay',
    'CPU_company','GPU_company','CPU_tier','GPU_series','Brand'
]

data = data[required_cols]

prediction = model.predict(data)

EUR_TO_INR = 106
price = round(prediction[0] * EUR_TO_INR, 2)

print("\n" + "="*30)
print(f"Predicted Price: ₹{price}")
print("="*30 + "\n")

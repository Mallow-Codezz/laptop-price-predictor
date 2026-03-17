import os
import pickle
import warnings
import pandas as pd
from src.feature_engineering import feature_engineering

warnings.filterwarnings('ignore')

def clean_text(x):
    return x.strip().lower()

def yes_no_input(ans):
    val = clean_text(input(ans))
    return "Yes" if val in ['yes', 'y', '1'] else "No"

def map_os(ans):
    val = clean_text(ans)
    if 'win' in val:
        return 'Windows'
    elif 'mac' in val:
        return 'macOS'
    elif 'linux' in val:
        return 'Linux'
    else:
        return 'Other'

def map_storage(val):
    val = clean_text(val)
    if 'ssd' in val:
        return 'SSD'
    elif 'hdd' in val:
        return 'HDD'
    elif 'flash' in val:
        return 'Flash Storage'
    elif 'hybrid' in val:
        return 'Hybrid'
    else:
        return 'SSD'  

def safe_int(val):
    try:
        return int(input(val))
    except:
        return 0

def safe_float(val):
    try:
        return float(input(val))
    except:
        return 0.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, 'rb') as file:
    model = pickle.load(file)

brand = input("Brand: ").title()
product = input("Laptop Name: ").title()
typename = input("Type (Notebook/Ultrabook/Gaming/...): ").title()

ram = safe_int("RAM (GB): ")
weight = safe_float("Weight (kg): ")

touchscreen = yes_no_input("Touchscreen (Yes/No): ")
ips = yes_no_input("IPS Panel (Yes/No): ")
retina = yes_no_input("Retina (Yes/No): ")

cpu_company = input("CPU Company: ").title()
cpu_model = input("CPU Model: ").strip()
cpu_freq = safe_float("CPU Frequency: ")

gpu_company = input("GPU Company: ").title()
gpu_model = input("GPU Model: ").strip()

screen = input("Screen Type: ").title()

screenw = safe_int("Screen Width: ")
screenh = safe_int("Screen Height: ")
inches = safe_float("Screen Size: ")

primary_storage = safe_int("Primary Storage (GB): ")
primary_type = map_storage(input("Primary Type (SSD/HDD/...): "))

secondary_storage = safe_int("Secondary Storage (GB): ")
secondary_type = map_storage(input("Secondary Type: "))

os = map_os(input("Operating System: "))

data = pd.DataFrame({
    "Company": [brand],
    "Product": [product],
    "TypeName": [typename],
    "Ram": [ram],
    "Weight": [weight],
    "Touchscreen": [touchscreen],
    "IPSpanel": [ips],
    "RetinaDisplay": [retina],
    "CPU_company": [cpu_company],
    "CPU_model": [cpu_model],
    "CPU_freq": [cpu_freq],
    "GPU_company": [gpu_company],
    "GPU_model": [gpu_model],
    "Screen": [screen],
    "ScreenW": [screenw],
    "ScreenH": [screenh],
    "Inches": [inches],
    "PrimaryStorage": [primary_storage],
    "PrimaryStorageType": [primary_type],
    "SecondaryStorage": [secondary_storage],
    "SecondaryStorageType": [secondary_type],
    "OS": [os]
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

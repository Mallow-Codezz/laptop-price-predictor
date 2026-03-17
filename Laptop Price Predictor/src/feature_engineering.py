import numpy as np
import pandas as pd

def feature_engineering(data):

    # Calculate PPI:
    data['PPI'] = (np.sqrt(data['ScreenW']**2) + (data['ScreenH']**2)) / data['Inches']
    data.drop(columns=['ScreenW', 'ScreenH', 'Inches'], inplace=True)

    # Storage Features:
    data['SSD'] = 0
    data['HDD'] = 0
    data['Flash_Storage'] = 0
    data['Hybrid'] = 0

    data.loc[data['PrimaryStorageType'] == 'SSD', 'SSD'] = data['PrimaryStorage']
    data.loc[data['PrimaryStorageType'] == 'HDD', 'HDD'] = data['PrimaryStorage'] 
    data.loc[data['PrimaryStorageType'] == 'Flash Storage', 'Flash_Storage'] = data['PrimaryStorage']
    data.loc[data['PrimaryStorageType'] == 'Hybrid', 'Hybrid'] = data['PrimaryStorage']

    data.loc[data['SecondaryStorageType'] == 'SSD', 'SSD'] += data['SecondaryStorage']
    data.loc[data['SecondaryStorageType'] == 'HDD', 'HDD'] += data['SecondaryStorage'] 
    data.loc[data['SecondaryStorageType'] == 'Flash Storage', 'Flash_Storage'] += data['SecondaryStorage'] 
    data.loc[data['SecondaryStorageType'] == 'Hybrid', 'Hybrid'] += data['SecondaryStorage']

    data.drop(columns=['PrimaryStorage', 'SecondaryStorage', 'PrimaryStorageType', 'SecondaryStorageType'], inplace=True)

    # CPU Tier:
    data['CPU_tier'] = 'other'

    data.loc[data['CPU_model'].str.contains('i7', case=False, na=False),'CPU_tier'] = 'i7'
    data.loc[data['CPU_model'].str.contains('i5', case=False, na=False),'CPU_tier'] = 'i5'
    data.loc[data['CPU_model'].str.contains('i3', case=False, na=False),'CPU_tier'] = 'i3'
    data.loc[data['CPU_model'].str.contains('Pentium', case=False, na=False),'CPU_tier'] = 'Pentium'
    data.loc[data['CPU_model'].str.contains('Celeron', case=False, na=False),'CPU_tier'] = 'Celeron'
    data.loc[data['CPU_model'].str.contains('AMD', case=False, na=False),'CPU_tier'] = 'AMD'
    data.loc[data['CPU_model'].str.contains('Ryzen', case=False, na=False),'CPU_tier'] = 'AMD'

    data.drop(columns=['CPU_model'], inplace=True)

    # GPU Series:
    data['GPU_series'] = 'other'

    data.loc[data['GPU_model'].str.contains('GTX', case=False, na=False), 'GPU_series'] = 'GTX'
    data.loc[data['GPU_model'].str.contains('MX', case=False, na=False), 'GPU_series'] = 'MX'
    data.loc[data['GPU_model'].str.contains('GT', case=False, na=False), 'GPU_series'] = 'GT'
    data.loc[data['GPU_model'].str.contains('UHD', case=False, na=False), 'GPU_series'] = 'Intel UHD'
    data.loc[data['GPU_model'].str.contains('HD', case=False, na=False), 'GPU_series'] = 'Intel HD'
    data.loc[data['GPU_model'].str.contains('Iris', case=False, na=False), 'GPU_series'] = 'Intel Iris'
    data.loc[data['GPU_model'].str.contains('Radeon', case=False, na=False), 'GPU_series'] = 'Radeon'

    data.drop(columns=['GPU_model'], inplace=True)

    # Brand Features:
    data['Brand'] = 'other'
    brands = ['Dell', 'Lenovo', 'HP', 'Asus', 'Acer', 'MSI', 'Toshiba', 'Apple']
    
    data.loc[data['Company'].isin(brands), 'Brand'] = data['Company'] 

    data.drop(columns=['Company', 'Product'], inplace=True)

    return data 
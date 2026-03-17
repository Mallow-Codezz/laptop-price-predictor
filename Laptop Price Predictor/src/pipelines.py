import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

def pipeline():
    numerical_columns = ['Ram', 'Weight', 'CPU_freq', 'PPI', 'SSD', 'HDD', 'Flash_Storage', 'Hybrid']
    categorical_columns = ['TypeName', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'CPU_company', 'GPU_company', 'CPU_tier', 'GPU_series', 'Brand']

    handling_numerical_columns = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    handling_categorical_columns = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessing = ColumnTransformer(transformers=[
        ('numerical', handling_numerical_columns, numerical_columns),
        ('categorical', handling_categorical_columns, categorical_columns)
    ])

    pipelines = Pipeline(steps=[
        ('preprocessing', preprocessing),
        ('model', Ridge(alpha=0.1))
    ])

    model = TransformedTargetRegressor(
        regressor=pipelines,
        func=np.log,
        inverse_func=np.exp
    )

    return model 
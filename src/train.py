import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor

print("Loading data...")
df = pd.read_csv('/app/data/cars_cleaned.csv')

numerical_features = ['Year', 'Number of Doors', 'Mileage_numeric',
                       'Fiscal_Power_numeric', 'car_age', 'mileage_per_year']
categorical_features = ['Brand', 'Condition', 'Gearbox', 'Fuel', 'Origin',
                         'First Owner', 'Location_grouped', 'Model_grouped']
binary_features = [c for c in df.columns if c.startswith('has_')]

X = df.drop(columns=['Price', 'log_price'])
y = df['log_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1))
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features),
    ('bin', 'passthrough', binary_features)
])

print("Training LightGBM...")
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LGBMRegressor(n_estimators=300, random_state=42))
])
pipeline.fit(X_train, y_train)

print("Saving artifacts...")
joblib.dump(pipeline, '/app/src/model.pkl', protocol=4)
joblib.dump(preprocessor, '/app/src/preprocessor.pkl', protocol=4)

print("Building SHAP explainer...")
lgbm_model = pipeline.named_steps['model']
explainer = shap.TreeExplainer(lgbm_model)
joblib.dump(explainer, '/app/src/explainer.pkl', protocol=4)

feature_names = numerical_features + categorical_features + binary_features
with open('/app/src/feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))

print("All artifacts saved successfully.")
import sys
sys.path.append('.')
from main import to_model_columns, CarInput, pipeline, explainer
import pandas as pd
import numpy as np

sample = CarInput(
    Brand="Dacia",
    Model_grouped="Logan",
    Year=2019,
    Condition="Good",
    Gearbox="Manual",
    Fuel="Diesel",
    Number_of_Doors=5.0,
    Origin="WW in Morocco",
    First_Owner="Yes",
    Mileage_numeric=85000,
    Fiscal_Power_numeric=6,
    Location_grouped="Casablanca",
)

df = to_model_columns(sample)
df['car_age'] = 2024 - df['Year']
df['mileage_per_year'] = (df['Mileage_numeric'] / df['car_age'].replace(0, 1)).clip(upper=50000)

log_pred = pipeline.predict(df)[0]
price = float(np.expm1(log_pred))

print(f"Predicted price: {price:,.0f} MAD")


transformed = pipeline.named_steps['preprocessor'].transform(df)
shap_vals = explainer.shap_values(transformed)[0]

with open('../src/feature_names.txt') as f:
    feature_names = f.read().splitlines()

contributions = {
    name: round(float(val) * price)
    for name, val in zip(feature_names, shap_vals)
}
sorted_contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

for k, v in list(sorted_contributions.items())[:8]:
    print(f"{k}: {v:+,} MAD")
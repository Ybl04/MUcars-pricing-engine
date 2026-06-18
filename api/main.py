from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="MUCars Pricing Intelligence Engine")

# Load everything once at startup, not per-request — loading a model from disk
# on every API call would be slow and pointless since the model never changes between requests
pipeline = joblib.load("../src/model.pkl")
import shap
lgbm_model = pipeline.named_steps['model']
explainer = shap.TreeExplainer(lgbm_model)

with open("../src/feature_names.txt") as f:
    feature_names = f.read().splitlines()


class CarInput(BaseModel):
    Brand: str
    Model_grouped: str
    Year: int
    Condition: str
    Gearbox: str
    Fuel: str
    Number_of_Doors: float
    Origin: str
    First_Owner: str
    Mileage_numeric: float
    Fiscal_Power_numeric: float
    Location_grouped: str
    has_ABS: int = 0
    has_Air_Conditioning: int = 0
    has_Airbags: int = 0
    has_Alloy_Wheels: int = 0
    has_CD_MP3_Bluetooth: int = 0
    has_Central_Locking: int = 0
    has_Cruise_Control: int = 0
    has_ESP: int = 0
    has_Electric_Windows: int = 0
    has_Leather_Seats: int = 0
    has_Navigation_System_GPS: int = 0
    has_Onboard_Computer: int = 0
    has_Parking_Sensors: int = 0
    has_Rear_Camera: int = 0
    has_Speed_Limiter: int = 0
    has_Sunroof: int = 0



def to_model_columns(car: CarInput) -> pd.DataFrame:
    """Translate API field names (underscores) into the exact column names
    the trained pipeline expects (some with spaces), since Pydantic models
    can't have spaces in field names but the pipeline was fit on the originals."""
    data = car.dict()
    data['Number of Doors'] = data.pop('Number_of_Doors')
    data['First Owner'] = data.pop('First_Owner')
    return pd.DataFrame([data])


@app.post("/predict")
def predict(car: CarInput):
    df = to_model_columns(car)

    # Recreate the engineered features exactly as done during training —
    # the API receives raw specs, not pre-engineered columns
    df['car_age'] = 2024 - df['Year']
    df['mileage_per_year'] = (df['Mileage_numeric'] / df['car_age'].replace(0, 1)).clip(upper=50000)

    log_pred = pipeline.predict(df)[0]
    price = float(np.expm1(log_pred))

    return {
        "predicted_price_mad": round(price),
        "confidence_interval_mad": [round(price * 0.88), round(price * 1.12)]
    }


@app.post("/explain")
def explain(car: CarInput):
    df = to_model_columns(car)
    df['car_age'] = 2024 - df['Year']
    df['mileage_per_year'] = (df['Mileage_numeric'] / df['car_age'].replace(0, 1)).clip(upper=50000)

    log_pred = pipeline.predict(df)[0]
    price = float(np.expm1(log_pred))

    transformed = pipeline.named_steps['preprocessor'].transform(df)
    shap_vals = explainer.shap_values(transformed)[0]

    contributions = {
        name: round(float(val) * price)
        for name, val in zip(feature_names, shap_vals)
    }
    sorted_contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

    return {
        "predicted_price_mad": round(price),
        "shap_contributions_mad": sorted_contributions
    }


df_full = pd.read_csv("../data/cars_cleaned.csv")

@app.get("/compare")
def compare(brand: str, model_grouped: str, year: int, mileage: float):
    similar = df_full[
        (df_full['Brand'] == brand) &
        (abs(df_full['Year'] - year) <= 2) &
        (abs(df_full['Mileage_numeric'] - mileage) <= 20000)
    ].head(5)

    return similar[['Brand', 'Model_grouped', 'Year', 'Price', 'Mileage_numeric', 'Location_grouped']].to_dict(orient='records')
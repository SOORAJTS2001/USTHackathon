import os

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()

# Globals (for lifespan)
models = {}
feature_names = []


# Define the request schema
class PredictionRequest(BaseModel):
    date: str
    temperature: float
    humidity: float
    is_raining: int
    is_sunny: int
    is_holiday: int
    holiday_name: str


# Lifespan event: load & train model on app startup
@app.on_event("startup")
def load_and_train():
    global models, feature_names

    # STEP 1: Load multiple CSV files
    base_path = "/Users/254428/PycharmProjects/USTHackathon/The Crown & Cutlery"
    dfs = [pd.read_csv(os.path.join(base_path, file)) for file in os.listdir(base_path) if file.endswith(".csv")]
    df = pd.concat(dfs, ignore_index=True)

    # STEP 2: Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df.drop(columns=['Date'], inplace=True)

    # One-hot encode Holiday_Name
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    holiday_encoded = encoder.fit_transform(df[['Holiday_Name']])
    holiday_encoded_df = pd.DataFrame(holiday_encoded, columns=encoder.get_feature_names_out(['Holiday_Name']))

    data = pd.concat([df.drop(columns=['Holiday_Name']), holiday_encoded_df], axis=1)

    # STEP 3: Train models
    targets = [
        'Dessert_Waste_kg', 'Soup_Waste_kg', 'Main Course_Waste_kg',
        'Appetizer_Waste_kg', 'Salad_Waste_kg', 'Beverage_Waste_kg'
    ]
    X = data.drop(columns=targets)
    y = data[targets]
    feature_names = X.columns.tolist()

    for target in targets:
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y[target])
        models[target] = model


# Utility to build input vector
def build_input_vector(req: PredictionRequest) -> pd.DataFrame:
    date = pd.to_datetime(req.date)
    dow = date.dayofweek
    month = date.month
    is_weekend = int(dow in [5, 6])

    base_features = {
        'Temperature': req.temperature,
        'Humidity': req.humidity,
        'Is_Raining': req.is_raining,
        'Is_Sunny': req.is_sunny,
        'Is_Holiday': req.is_holiday,
        'DayOfWeek': dow,
        'Month': month,
        'Is_Weekend': is_weekend,
    }

    holiday_one_hot = {col: 0 for col in feature_names if col.startswith("Holiday_Name_")}
    holiday_col = f"Holiday_Name_{req.holiday_name}"
    if holiday_col in holiday_one_hot:
        holiday_one_hot[holiday_col] = 1
    else:
        print(f"⚠️ Warning: Unknown holiday '{req.holiday_name}', using zeroed one-hot vector.")

    input_data = {**base_features, **holiday_one_hot}
    return pd.DataFrame([input_data], columns=feature_names)


# Prediction endpoint
@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        input_vector = build_input_vector(req)
        predictions = {target: round(model.predict(input_vector)[0], 2) for target, model in models.items()}
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

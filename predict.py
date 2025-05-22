import os
from datetime import datetime, timedelta
from functools import lru_cache

import holidays
import httpx
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()

# Globals
models = {}
feature_names = []
importances = {}
accuracies = {}
encoder = None

# US holidays (change to "IN" for India, "GB" for UK, etc.)
in_holidays = holidays.country_holidays("IN", years=range(2020, 2031))


# Request schemas
class PredictionRequest(BaseModel):
    date: str  # "YYYY-MM-DD"
    latitude: float
    longitude: float
    filter: str | None = None
    temperature: float | None = None
    humidity: float | None = None
    is_raining: int | None = None
    is_sunny: int | None = None
    is_holiday: bool | None = None
    holiday_name: str | None = None

async def get_weather_forecast_day(req: PredictionRequest) -> PredictionRequest:
    async with httpx.AsyncClient() as client:
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "cloud_cover"],
            "forecast_days": 1
        }
        resp = await client.get(forecast_url, params=params)
        resp.raise_for_status()
        data = resp.json()

        temperature = data["current"]["temperature_2m"]
        humidity = data["current"]["relative_humidity_2m"]
        precipitation = data["current"]["precipitation"]
        cloudcover = data["current"]["cloud_cover"]

        is_raining = 1 if precipitation > 0 else 0
        is_sunny = 1 if cloudcover < 40 else 0

        return req.model_copy(update={
            "temperature": temperature,
            "humidity": humidity,
            "is_raining": is_raining,
            "is_sunny": is_sunny,
        })


async def get_weather_forecast_week(lat: float, lon: float):
    async with httpx.AsyncClient() as client:
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "precipitation_sum", "rain_sum"],
            "forecast_days": 16

        }
        resp = await client.get(forecast_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        temperatures = daily.get("temperature_2m_max", [])
        rain_sums = daily.get("rain_sum", [])
        dates = daily.get("time", [])

        weather_data = {}

        for i in range(len(dates)):
            temp = temperatures[i] or 25
            rain = rain_sums[i] or 0
            # Estimate humidity
            if rain > 0:
                humidity = 85 + (10 if temp < 10 else 0)
            else:
                humidity = 60 if temp > 20 else 70

            # Determine rain/sun status
            is_raining = 1 if rain > 0 else 0
            is_sunny = 1 if rain == 0 and temp > 15 else 0

            weather_data[dates[i]] = {
                "date": dates[i],
                "temperature": temp,
                "humidity": humidity,
                "is_raining": is_raining,
                "is_sunny": is_sunny,
            }
        return weather_data


@app.on_event("startup")
def load_and_train():
    global models, feature_names, importances, accuracies, encoder

    base_path = "/Users/254428/PycharmProjects/USTHackathon/Plav"
    dfs = [pd.read_csv(os.path.join(base_path, file)) for file in os.listdir(base_path) if file.endswith(".csv")]
    df = pd.concat(dfs, ignore_index=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df.drop(columns=['Date'], inplace=True)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    holiday_encoded = encoder.fit_transform(df[['Holiday_Name']])
    holiday_encoded_df = pd.DataFrame(holiday_encoded, columns=encoder.get_feature_names_out(['Holiday_Name']))

    data = pd.concat([df.drop(columns=['Holiday_Name']), holiday_encoded_df], axis=1)

    targets = [
        'Dessert_Waste_kg', 'Soup_Waste_kg', 'Main_Course_Waste_kg',
        'Appetizer_Waste_kg', 'Salad_Waste_kg', 'Beverage_Waste_kg'
    ]
    X = data.drop(columns=targets)
    y = data[targets]
    feature_names.extend(X.columns.tolist())

    for target in targets:
        X_train, X_test, y_train, y_test = train_test_split(X, y[target], test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        models[target] = model

        y_pred = model.predict(X_test)
        accuracies[target] = r2_score(y_test, y_pred)

        feat_importance = dict(zip(X.columns, model.feature_importances_))
        importances[target] = feat_importance


# Build input vector
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

    input_data = {**base_features, **holiday_one_hot}
    return pd.DataFrame([input_data], columns=feature_names)


# Enrich request with holiday info
async def enrich_request(req: PredictionRequest) -> PredictionRequest:
    date_obj = pd.to_datetime(req.date)
    iso_date = date_obj.date()

    is_holiday = 1 if iso_date in in_holidays else 0
    holiday_name = in_holidays.get(iso_date, "None")
    req.is_holiday = is_holiday
    req.holiday_name = holiday_name
    return req

@app.post("/predict")
async def predict(req: PredictionRequest):
    try:
        req = await get_weather_forecast_day(req=req)
        req = await enrich_request(req)
        input_vector = build_input_vector(req)

        predictions = {
            target: round(model.predict(input_vector)[0], 2)
            for target, model in models.items()
        }

        insights = {}
        for target in models.keys():
            feat_dict = importances.get(target, {})
            total = sum(feat_dict.values())
            contrib_percent = {
                feat: round((imp / total) * 100, 2)
                for feat, imp in feat_dict.items()
            }
            sorted_contrib = dict(sorted(contrib_percent.items(), key=lambda x: x[1], reverse=True))

            insights[target] = {
                "accuracy": round(accuracies.get(target, 0), 3),
                "feature_contributions_percent": sorted_contrib
            }
        if req.filter and predictions.get(req.filter):
            predictions = {req.filter: predictions.get(req.filter)}
        return JSONResponse(content={
            "date": req.date,
            "is_weekend": int(pd.to_datetime(req.date).dayofweek in [5, 6]),
            "is_holiday": req.is_holiday,
            "holiday_name": req.holiday_name,
            "predictions": predictions,
            # "model_insights": insights
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict-week")
async def predict_week(req: PredictionRequest):
    try:
        start_date = datetime.strptime(req.date, "%Y-%m-%d")
        weather_data = await get_weather_forecast_week(lat=req.latitude, lon=req.longitude)
        results = []

        for i in range(7):
            current_date = start_date + timedelta(days=i)
            iso_date = current_date.date()
            is_holiday = 1 if iso_date in in_holidays else 0
            holiday_name = in_holidays.get(iso_date, "None")
            is_weekend = int(current_date.weekday() in [5, 6])
            data_ = weather_data[str(iso_date)]
            req.temperature = data_["temperature"]
            req.humidity = data_["humidity"]
            req.is_raining = data_["is_raining"]
            req.is_sunny = data_["is_sunny"]
            enriched_req = await enrich_request(req)
            input_vector = build_input_vector(enriched_req)
            predictions = {
                target: round(model.predict(input_vector)[0], 2)
                for target, model in models.items()
            }
            insights = {}
            for target in models.keys():
                feat_dict = importances.get(target, {})
                total = sum(feat_dict.values())
                contrib_percent = {
                    feat: round((imp / total) * 100, 2)
                    for feat, imp in feat_dict.items()
                    if not feat.startswith("Holiday_Name_")
                }
                sorted_contrib = dict(sorted(contrib_percent.items(), key=lambda x: x[1], reverse=True))

                insights[target] = {
                    "accuracy": round(accuracies.get(target, 0), 3),
                    "feature_contributions_percent": sorted_contrib
                }
            if req.filter and predictions.get(req.filter) and insights.get(req.filter):
                predictions = {req.filter: predictions.get(req.filter)}
                insights = {req.filter: insights.get(req.filter)}
            results.append({
                "date": str(iso_date),
                "temperature": req.temperature,
                "humidity": req.humidity,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "is_raining": req.is_raining,
                "is_sunny": req.is_sunny,
                "holiday_name": holiday_name,
                "predictions": predictions,
                "model_insights": insights

            })

        return JSONResponse(content={"weekly_predictions": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

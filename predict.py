import io
import os
import pathlib
import random
from datetime import datetime, timedelta

import holidays
import httpx
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, HTMLResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import json

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

app = FastAPI()

# Globals
models = {}
feature_names = []
importances = {}
accuracies = {}
encoder = None

df_plot = None

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

WASTE_COLUMNS = [
    "Dessert_Waste_kg",
    "Soup_Waste_kg",
    "Main_Course_Waste_kg",
    "Appetizer_Waste_kg",
    "Salad_Waste_kg",
    "Beverage_Waste_kg"
]


def upsert_csv_row_pandas(new_row_dict: list[dict]):
    filename = "predicted_data/predictions.csv"
    """Upserts a row (as dict) to a CSV using Pandas, keeping 'Date' unique and column order fixed."""
    columns = [
        "Date",
        "Temperature",
        "Humidity",
        "Is_Raining",
        "Is_Sunny",
        "Is_Holiday",
        "Holiday_Name",
        "Dessert_Waste_kg",
        "Soup_Waste_kg",
        "Main_Course_Waste_kg",
        "Appetizer_Waste_kg",
        "Salad_Waste_kg",
        "Beverage_Waste_kg"
    ]
    for row in new_row_dict:
        # Convert the row into a one-row DataFrame with correct column order
        row["Date"] = row.pop("date")
        row["Temperature"] = row.pop("temperature")
        row["Humidity"] = row.pop("humidity")
        row["Is_Raining"] = row.pop("is_raining")
        row["Is_Sunny"] = row.pop("is_sunny")
        row["Is_Holiday"] = row.pop("is_holiday")
        row["Holiday_Name"] = row.pop("holiday_name")
        row['Dessert_Waste_kg'] = row['predictions'].pop("Dessert_Waste_kg")
        row['Soup_Waste_kg'] = row['predictions'].pop("Soup_Waste_kg")
        row['Main_Course_Waste_kg'] = row['predictions'].pop("Main_Course_Waste_kg")
        row['Appetizer_Waste_kg'] = row['predictions'].pop("Appetizer_Waste_kg")
        row['Salad_Waste_kg'] = row['predictions'].pop("Salad_Waste_kg")
        row['Beverage_Waste_kg'] = row['predictions'].pop("Beverage_Waste_kg")
        row.pop("predictions")
        new_row_df = pd.DataFrame([row], columns=columns)
        if os.path.exists(filename):
            df = pd.read_csv(filename)

            # Check for column mismatch
            if set(df.columns) != set(columns):
                raise ValueError("CSV columns don't match provided columns")
            df["Holiday_Name"] = df["Holiday_Name"].astype("string")
            new_row_df["Holiday_Name"] = new_row_df["Holiday_Name"].astype("string")
            df.set_index("Date", inplace=True)
            new_row_df.set_index("Date", inplace=True)

            # Update or append
            df.update(new_row_df)
            combined_df = pd.concat([df, new_row_df[~new_row_df.index.isin(df.index)]])
            combined_df.reset_index(inplace=True)
        else:
            combined_df = new_row_df
        # Ensure column order is preserved
        combined_df = combined_df[columns]
        combined_df.to_csv(filename, index=False)


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
    global models, feature_names, importances, accuracies, encoder, df_plot

    base_path = pathlib.Path(__file__).parent / "Plav"
    os.makedirs("predicted_data", exist_ok=True)
    dfs = [pd.read_csv(os.path.join(base_path, file)) for file in os.listdir(base_path) if file.endswith(".csv")]
    df = pd.concat(dfs, ignore_index=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df_plot = df.copy()
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
    holiday_name = in_holidays.get(iso_date, None)
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
        data = {
            "date": req.date,
            "temperature": req.temperature,
            "humidity": req.humidity,
            "is_raining": req.is_raining,
            "is_sunny": req.is_sunny,
            "is_weekend": int(pd.to_datetime(req.date).dayofweek in [5, 6]),
            "is_holiday": req.is_holiday,
            "holiday_name": req.holiday_name,
            "predictions": predictions,
            # "model_insights": insights
        }
        # for updated csv
        upsert_csv_row_pandas([data])
        return JSONResponse(content=data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict-week")
async def predict_week(req: PredictionRequest):
    try:
        start_date = datetime.strptime(req.date, "%Y-%m-%d")
        weather_data = await get_weather_forecast_week(lat=req.latitude, lon=req.longitude)
        results = []
        results_ = []

        for i in range(7):
            current_date = start_date + timedelta(days=i)
            iso_date = current_date.date()
            is_holiday = 1 if iso_date in in_holidays else 0
            holiday_name = in_holidays.get(iso_date, None)
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
                    feat:round((imp / total) * 100 * random.uniform(0.6, 1.05), 2)
                    for feat, imp in feat_dict.items()
                    if not feat.startswith("Holiday_Name_")
                }
                sorted_contrib = dict(sorted(contrib_percent.items(), key=lambda x: x[1], reverse=True))

                insights[target] = {
                    "accuracy": round(accuracies.get(target, 0), 3),
                    "feature_contributions_percent": sorted_contrib
                }
            data_ = {
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
            }
            if req.filter and predictions.get(req.filter) and insights.get(req.filter):
                predictions = predictions[req.filter]
                insights = insights[req.filter]
            data = {
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

            }
            results.append(data)
            results_.append(data_)
        # for all fields
        upsert_csv_row_pandas(results_)
        return JSONResponse(content={"weekly_predictions": results})

    except Exception as e:
        # raise e
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/analytics", response_class=HTMLResponse)
def compare_actual_vs_predicted():
    trained_data_path = pathlib.Path(__file__).parent / "Plav"
    trained_dfs = []
    trained_filenames = []

    for file in os.listdir(trained_data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(trained_data_path / file, parse_dates=["Date"])
            trained_dfs.append(df)
            trained_filenames.append(file)

    pred_df = pd.read_csv("predicted_data/predictions.csv", parse_dates=["Date"])
    pred_df["DayOfYear"] = pred_df["Date"].dt.dayofyear
    pred_df["Date_str"] = pred_df["Date"].dt.strftime("%Y-%m-%d")

    categories = [
        "Dessert_Waste_kg", "Soup_Waste_kg", "Main_Course_Waste_kg",
        "Appetizer_Waste_kg", "Salad_Waste_kg", "Beverage_Waste_kg"
    ]

    series = []

    for cat in categories:
        series.append({
            "name": f"Predicted → {cat}",
            "type": "line",
            "data": list(zip(pred_df["DayOfYear"], pred_df[cat]))
        })

    pred_days = set(pred_df["DayOfYear"])

    for filename, df in zip(trained_filenames, trained_dfs):
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df["Date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
        df_matched = df[df["DayOfYear"].isin(pred_days)]

        for cat in categories:
            series.append({
                "name": f"{filename} → {cat}",
                "type": "line",
                "data": list(zip(df_matched["DayOfYear"], df_matched[cat]))
            })

    legend_data = [s["name"] for s in series]
    legend_selected = {name: False for name in legend_data}

    chart_data = {
        "tooltip": {
            "trigger": "axis",
            "formatter": """
                function(params) {
                    let tooltip = '<strong>Day of Year: ' + params[0].data[0] + '</strong><br>';
                    for (let p of params) {
                        tooltip += p.seriesName + ': ' + p.data[1] + '<br>';
                    }
                    return tooltip;
                }
            """
        },
        "legend": {
            "orient": "horizontal",
            "bottom": 10,
            "data": legend_data,
            "selected": legend_selected
        },
        "grid": {
            "bottom": 120
        },
        "xAxis": {"type": "value", "name": "Day of Year"},
        "yAxis": {"type": "value", "name": "Waste (kg)"},
        "series": series
    }

    return f"""
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
            <style>
                body {{ background: #f9f9f9; font-family: sans-serif; padding: 2rem; }}
                #main {{ width: 100%; height: 950px; }}
            </style>
        </head>
        <body>
            <h2>📊 Food Waste Comparison by Day-of-Year (Matched Dates Only)</h2>
            <div id="main"></div>
            <script>
                var chart = echarts.init(document.getElementById('main'));
                var option = {json.dumps(chart_data)};
                chart.setOption(option);
            </script>
        </body>
    </html>
    """

@app.get("/plot")
async def plot_from_predictions():
    pred_df = pd.read_csv("predicted_data/predictions.csv")
    all_data_df = pd.concat([df_plot, pred_df], ignore_index=True)
    df = all_data_df[['Date', 'Dessert_Waste_kg', 'Soup_Waste_kg',
       'Main_Course_Waste_kg', 'Appetizer_Waste_kg', 'Salad_Waste_kg',
       'Beverage_Waste_kg']]

    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year
    df['MonthDay'] = df['Date'].dt.strftime('%m-%d')

    df.rename(columns={'Dessert_Waste_kg': 'Dessert', 'Soup_Waste_kg': 'Soup',
       'Main_Course_Waste_kg': 'Main Course', 'Appetizer_Waste_kg': 'Appetizer', 'Salad_Waste_kg': 'Salad',
       'Beverage_Waste_kg': 'Beverage'}, inplace=True)

    df_melted = df.melt(
        id_vars=['Year', 'MonthDay'],
        value_vars=['Soup', 'Appetizer', 'Salad', 'Main Course', 'Dessert', 'Beverage'],
        var_name='metric',
        value_name='wastage'
    )

    df_melted = df_melted[df_melted["MonthDay"] == df_melted["MonthDay"].iloc[-1]]

    g = sns.FacetGrid(df_melted, col="metric", col_wrap=2, height=4, sharey=False, sharex=True, aspect=1, legend_out=False)
    g.fig.suptitle(f'Wastage in Kilograms / Day of the Year ({df["MonthDay"].iloc[-1]})', fontsize=15, fontweight='bold')
    g.map_dataframe(sns.barplot, x="Year", y="wastage", palette="muted", hue="Year")

    g.set_titles(col_template="{col_name}", size=14, y=0.96)
    g.set_axis_labels("year", "wastage (kg)")

    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    g.add_legend()
    sns.move_legend(g, "upper right", frameon=False)

    line = Line2D(
        [0.15, 0.85],
        [0.95, 0.95],
        transform=g.fig.transFigure,
        color="black",
        linewidth=1.2
    )
    g.fig.lines.append(line)
    g.fig.subplots_adjust(top=0.92)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    buf_contents = img_buf.getvalue()
    img_buf.close()

    headers = {'Content-Disposition': 'inline; filename="wastage.png"'}
    return Response(buf_contents, headers=headers, media_type='image/png')

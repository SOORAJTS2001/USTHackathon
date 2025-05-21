import csv
import os
import random
from datetime import datetime, timedelta

import holidays

food_categories = ['Dessert', 'Soup', 'Main_Course', 'Appetizer', 'Salad', 'Beverage']


def generate(hotel_name: str, start_date: datetime, scale_factor: float = 5.0):
    days_in_year = 365
    year = start_date.year

    indian_holidays = holidays.India(years=year)
    holiday_dict = {day.strftime("%Y-%m-%d"): name for day, name in indian_holidays.items()}

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5, 6]:
            return 'Summer'
        elif month in [7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Autumn'

    def seasonal_weather(season):
        if season == 'Winter':
            temp = round(random.uniform(10.0, 25.0), 1)
            humidity = random.randint(40, 70)
        elif season == 'Summer':
            temp = round(random.uniform(30.0, 45.0), 1)
            humidity = random.randint(30, 60)
        elif season == 'Monsoon':
            temp = round(random.uniform(25.0, 35.0), 1)
            humidity = random.randint(70, 95)
        else:  # Autumn
            temp = round(random.uniform(25.0, 35.0), 1)
            humidity = random.randint(50, 80)
        return temp, humidity

    def is_long_weekend(date):
        date_str = date.strftime("%Y-%m-%d")
        prev_day = (date - timedelta(days=1)).strftime("%Y-%m-%d")
        next_day = (date + timedelta(days=1)).strftime("%Y-%m-%d")
        return (
            (date.weekday() == 0 and prev_day in holiday_dict) or
            (date.weekday() == 4 and next_day in holiday_dict) or
            (date.weekday() in [5, 6] and date_str in holiday_dict)
        )

    data = []
    for i in range(days_in_year):
        date = start_date + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        weekday = date.weekday()
        month = date.month
        season = get_season(month)

        temperature, humidity = seasonal_weather(season)

        # Rainfall logic for India
        if season == 'Monsoon':
            is_raining = 1 if random.random() < 0.6 else 0
        else:
            is_raining = random.choices([0, 1], weights=[0.7, 0.3])[0]

        is_sunny = 1 if not is_raining and random.random() > 0.5 else 0

        if date_str in holiday_dict:
            is_holiday = 1
            holiday_name = holiday_dict[date_str]
        elif weekday in [5, 6]:
            is_holiday = 1
            holiday_name = 'Weekend'
        else:
            is_holiday = 0
            holiday_name = 'None'

        long_weekend = is_long_weekend(date)

        waste = {}
        for category in food_categories:
            base = 0

            if category == 'Dessert':
                if temperature > 30 and is_sunny:
                    base = 0.5
                elif is_raining:
                    base = 5
                else:
                    base = 3

            elif category == 'Soup':
                if temperature < 20 and is_raining:
                    base = 1
                elif temperature > 30:
                    base = 5
                else:
                    base = 2

            elif category == 'Main_Course':
                base = 8
                if is_holiday or long_weekend:
                    base *= 1.2
                if weekday in [0, 2]:
                    base *= 1.1

            elif category == 'Appetizer':
                base = 4
                if weekday in [5, 6]:
                    base *= 1.3
                if is_raining:
                    base *= 0.8

            elif category == 'Salad':
                if temperature > 30 and is_sunny:
                    base = 2
                elif is_raining:
                    base = 4
                elif weekday in [5, 6]:
                    base = 5
                else:
                    base = 3

            elif category == 'Beverage':
                if temperature > 35 and is_sunny:
                    base = 1.5
                elif temperature < 15:
                    base = 4
                else:
                    base = 2.5

            # Adjustments
            if long_weekend:
                base *= 0.5
            elif is_holiday:
                base *= 0.7
            elif weekday in [0, 2]:
                base *= 1.3

            waste[category] = round(random.uniform(0.9, 1) * base * scale_factor, 1)

        row = [
            date_str, temperature, humidity, is_raining, is_sunny,
            is_holiday, holiday_name
        ] + [waste[cat] for cat in food_categories]

        data.append(row)

    # Write CSV
    csv_filename = f"{start_date.year}.csv"
    os.makedirs(hotel_name, exist_ok=True)
    with open(f"{hotel_name}/{csv_filename}", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date", "Temperature", "Humidity", "Is_Raining", "Is_Sunny",
            "Is_Holiday", "Holiday_Name"
        ] + [f"{cat}_Waste_kg" for cat in food_categories])
        writer.writerows(data)

    print(f"✅ CSV file '{csv_filename}' generated in folder '{hotel_name}' with Indian context.")


# Example execution
hotels = ['Supreme Bakery', 'Saranva Bhavan', 'Mamta Rasooi', 'Kaffehaus', 'Plav', 'Souther Delight']
years = [2022, 2023, 2024]
for hotel in hotels:
    for year in years:
        generate(hotel_name=hotel, start_date=datetime(year, 1, 1), scale_factor=5.0)

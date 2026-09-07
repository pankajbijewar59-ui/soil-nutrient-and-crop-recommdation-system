"""
data_generator.py
------------------
Generates a realistic synthetic soil-nutrient & climate dataset for
crop recommendation, modeled on typical agronomic ranges (similar in
spirit to the well-known Kaggle "Crop Recommendation" dataset).

Columns: N, P, K, temperature, humidity, ph, rainfall, label
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Approximate ideal growing ranges (min, max) per crop for:
# N (kg/ha), P (kg/ha), K (kg/ha), temperature (C), humidity (%), ph, rainfall (mm)
CROP_PROFILES = {
    "rice":        {"N": (60, 100), "P": (35, 60),  "K": (35, 45),  "temp": (20, 27), "hum": (75, 90), "ph": (5.5, 7.0), "rain": (180, 300)},
    "maize":       {"N": (60, 100), "P": (35, 60),  "K": (15, 25),  "temp": (18, 27), "hum": (55, 75), "ph": (5.5, 7.5), "rain": (60, 110)},
    "chickpea":    {"N": (20, 45),  "P": (55, 80),  "K": (75, 100), "temp": (17, 25), "hum": (14, 25), "ph": (6.0, 8.0), "rain": (60, 100)},
    "kidneybeans": {"N": (15, 40),  "P": (55, 80),  "K": (15, 25),  "temp": (15, 25), "hum": (18, 25), "ph": (5.5, 6.5), "rain": (60, 150)},
    "pigeonpeas":  {"N": (15, 40),  "P": (55, 80),  "K": (15, 25),  "temp": (18, 37), "hum": (30, 70), "ph": (4.5, 7.5), "rain": (90, 200)},
    "mothbeans":   {"N": (15, 40),  "P": (35, 60),  "K": (15, 25),  "temp": (24, 32), "hum": (35, 65), "ph": (3.5, 9.9), "rain": (30, 70)},
    "mungbean":    {"N": (15, 40),  "P": (35, 60),  "K": (15, 25),  "temp": (25, 35), "hum": (60, 90), "ph": (6.2, 7.2), "rain": (40, 65)},
    "blackgram":   {"N": (20, 45),  "P": (55, 80),  "K": (15, 25),  "temp": (25, 35), "hum": (60, 75), "ph": (6.0, 7.5), "rain": (60, 80)},
    "lentil":      {"N": (15, 30),  "P": (55, 80),  "K": (15, 25),  "temp": (15, 27), "hum": (60, 70), "ph": (5.5, 7.5), "rain": (35, 55)},
    "pomegranate": {"N": (15, 40),  "P": (10, 30),  "K": (35, 45),  "temp": (18, 25), "hum": (85, 95), "ph": (5.5, 7.0), "rain": (100, 130)},
    "banana":      {"N": (90, 120), "P": (70, 100), "K": (45, 55),  "temp": (23, 30), "hum": (75, 90), "ph": (5.5, 7.0), "rain": (95, 130)},
    "mango":       {"N": (15, 40),  "P": (15, 40),  "K": (25, 35),  "temp": (27, 37), "hum": (45, 55), "ph": (4.5, 7.0), "rain": (30, 100)},
    "grapes":      {"N": (15, 40),  "P": (120,145), "K": (195,205), "temp": (8, 42),  "hum": (80, 90), "ph": (5.5, 7.0), "rain": (60, 75)},
    "watermelon":  {"N": (90, 110), "P": (10, 25),  "K": (45, 55),  "temp": (24, 27), "hum": (80, 90), "ph": (6.0, 6.8), "rain": (40, 50)},
    "muskmelon":   {"N": (90, 110), "P": (10, 25),  "K": (45, 55),  "temp": (27, 30), "hum": (90, 95), "ph": (6.0, 6.8), "rain": (20, 30)},
    "apple":       {"N": (15, 40),  "P": (120,145), "K": (195,205), "temp": (21, 24), "hum": (90, 95), "ph": (5.5, 6.5), "rain": (100, 125)},
    "orange":      {"N": (15, 40),  "P": (10, 30),  "K": (10, 15),  "temp": (10, 34), "hum": (90, 95), "ph": (6.0, 7.5), "rain": (100, 120)},
    "papaya":      {"N": (45, 70),  "P": (55, 80),  "K": (45, 55),  "temp": (23, 43), "hum": (85, 95), "ph": (6.0, 7.0), "rain": (40, 250)},
    "coconut":     {"N": (15, 40),  "P": (10, 30),  "K": (25, 35),  "temp": (25, 30), "hum": (90, 100),"ph": (5.5, 7.0), "rain": (140, 230)},
    "cotton":      {"N": (100,140), "P": (35, 60),  "K": (15, 25),  "temp": (22, 27), "hum": (70, 85), "ph": (5.5, 7.5), "rain": (60, 100)},
    "jute":        {"N": (60, 100), "P": (35, 60),  "K": (35, 45),  "temp": (23, 27), "hum": (70, 90), "ph": (6.0, 7.5), "rain": (150, 200)},
    "coffee":      {"N": (80, 120), "P": (15, 40),  "K": (25, 35),  "temp": (23, 28), "hum": (50, 70), "ph": (6.0, 7.5), "rain": (150, 200)},
}

SAMPLES_PER_CROP = 120


def sample_range(low, high, n):
    return np.round(np.random.uniform(low, high, n), 2)


def generate_dataset():
    rows = []
    for crop, prof in CROP_PROFILES.items():
        n = SAMPLES_PER_CROP
        N = sample_range(*prof["N"], n)
        P = sample_range(*prof["P"], n)
        K = sample_range(*prof["K"], n)
        temp = sample_range(*prof["temp"], n)
        hum = sample_range(*prof["hum"], n)
        ph = sample_range(*prof["ph"], n)
        rain = sample_range(*prof["rain"], n)
        for i in range(n):
            rows.append([N[i], P[i], K[i], temp[i], hum[i], ph[i], rain[i], crop])

    df = pd.DataFrame(rows, columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("dataset.csv", index=False)
    print(f"Dataset generated: {df.shape[0]} rows, {df['label'].nunique()} crop classes")
    print(df.head())

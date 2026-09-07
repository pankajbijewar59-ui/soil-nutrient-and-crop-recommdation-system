# 🌾 AI Soil Nutrient Analysis & Crop Recommendation System

Ek Machine Learning based project jo soil ke nutrient values (N, P, K),
climate data (temperature, humidity, rainfall) aur pH ke basis par:

1. **Best crop recommend karta hai** (22 crops mein se) — Random Forest classifier (~99% accuracy)
2. **Soil nutrient deficiency analysis** deta hai (kya kam hai, kya zyada hai, kya khaad/fertilizer daalein)
3. Ek interactive **Streamlit web app** ke through use hota hai

---

## 📁 Project Structure

```
soil_crop_recommendation/
├── data_generator.py     # Realistic synthetic soil-crop dataset generator
├── train_model.py        # Trains Random Forest model, saves .pkl files
├── app.py                # Streamlit web app (main UI)
├── requirements.txt      # Python dependencies
├── dataset.csv           # Generated dataset (22 crops x 120 samples)
├── crop_model.pkl        # Trained model (already included)
├── scaler.pkl            # Feature scaler (already included)
└── README.md
```

## 🌱 Supported Crops (22)

rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean,
blackgram, lentil, pomegranate, banana, mango, grapes, watermelon,
muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

## ⚙️ Input Features

| Feature | Description | Typical Range |
|---|---|---|
| N | Nitrogen content | 0–150 kg/ha |
| P | Phosphorus content | 0–150 kg/ha |
| K | Potassium content | 0–210 kg/ha |
| temperature | Air temperature | 0–45 °C |
| humidity | Relative humidity | 0–100 % |
| ph | Soil pH value | 0–14 |
| rainfall | Rainfall | 0–300 mm |

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate dataset & retrain model
Model files are already included, but you can rebuild them:
```bash
python data_generator.py     # creates dataset.csv
python train_model.py        # trains model -> crop_model.pkl, scaler.pkl
```

### 3. Launch the web app
```bash
streamlit run app.py
```
Browser mein `http://localhost:8501` khul jayega.

## 🧠 How It Works

- **Model**: RandomForestClassifier (200 trees), trained on 7 soil/climate features → crop label
- **Preprocessing**: StandardScaler for feature normalization
- **Nutrient Analysis**: N, P, K, pH ko general agronomic reference ranges se compare karke
  low/optimal/high status + Hindi-English fertilizer advice diya jata hai
- **Confidence Scores**: Top-5 crop matches with probability shown as a bar chart

## 📊 Model Performance

Test set par ~99% accuracy (synthetic dataset par). Real-world use ke liye,
`dataset.csv` ko apne actual field/lab soil-test data se replace karke
`train_model.py` dobara run karein — better real-world accuracy milegi.

## 🔧 Customization Ideas

- Real soil-test lab data (ICAR / state agriculture dept datasets) se retrain karein
- Fertilizer dosage calculator add karein (quantity in kg based on deficiency %)
- Weather API integrate karke temperature/humidity/rainfall auto-fetch karein
- Multi-language support (Hindi/regional languages) UI ke liye
- Location-based soil database se NPK values auto-fill karein

---
Made with Python, scikit-learn & Streamlit.

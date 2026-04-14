import streamlit as st
import pandas as pd
import joblib
import datetime
import requests
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Electricity Demand Forecaster", layout="wide")
MODEL_PATH = 'electricity_xgb_prediction_model.pkl'


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

OPENWEATHER_API_KEY =st.secrets.get("OPENWEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY")


def get_live_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url).json()
        if response.get("cod") == 200:
            return {
                "temp": response['main']['temp'],
                "hum": response['main']['humidity']
            }
        else:
            st.sidebar.error(f"Error: {response.get('message', 'City not found')}")
            return None
    except Exception as e:
        st.sidebar.error(f"Connection error: {e}")
        return None



st.title(" Smart Electricity Demand Forecasting")
st.markdown("This app predicts grid load using **XGBoost** and real-time weather data.")


st.sidebar.header("Step 1: Location & Time")
city = st.sidebar.text_input("Enter City for Live Weather", "Nagpur")
selected_date = st.sidebar.date_input("Select Date", datetime.date.today())
selected_time = st.sidebar.slider("Select Hour", 0, 23, 12)


weather_data = None
if st.sidebar.button("Fetch Live Weather"):
    weather_data = get_live_weather(city)

st.sidebar.markdown("---")
st.sidebar.header("Step 2: Environmental & Historical Data")


default_temp = weather_data['temp'] if weather_data else 25.0
default_hum = weather_data['hum'] if weather_data else 50.0

temp = st.sidebar.number_input("Temperature (°C)", value=float(default_temp))
hum = st.sidebar.number_input("Humidity (%)", value=float(default_hum))


st.sidebar.subheader("Time-Series Features")
lag_24 = st.sidebar.number_input("Demand 24h ago (MW)", value=2500.0)
lag_168 = st.sidebar.number_input("Demand 7 days ago (MW)", value=2550.0)
roll_mean = st.sidebar.number_input("24h Rolling Mean (MW)", value=2400.0)
roll_std = st.sidebar.number_input("24h Rolling Std (MW)", value=100.0)


day_of_week = selected_date.weekday()
month = selected_date.month
year = selected_date.year
day_of_year = selected_date.timetuple().tm_yday
week_of_year = selected_date.isocalendar()[1]
quarter = (month - 1) // 3 + 1
is_weekend = 1 if day_of_week >= 5 else 0


feature_columns = [
    'hour', 'dayofweek', 'month', 'year', 'dayofyear', 'weekofyear',
    'quarter', 'is_weekend', 'Temperature', 'Humidity',
    'Demand_lag_24hr', 'demand_lag_168hr', 'demand_rolling_mean_24hr', 'demand_rolling_std_24hr'
]

input_dict = {
    'hour': [selected_time],
    'dayofweek': [day_of_week],
    'month': [month],
    'year': [year],
    'dayofyear': [day_of_year],
    'weekofyear': [week_of_year],
    'quarter': [quarter],
    'is_weekend': [is_weekend],
    'Temperature': [temp],
    'Humidity': [hum],
    'Demand_lag_24hr': [lag_24],
    'demand_lag_168hr': [lag_168],
    'demand_rolling_mean_24hr': [roll_mean],
    'demand_rolling_std_24hr': [roll_std]
}

input_df = pd.DataFrame(input_dict)[feature_columns]


col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Input Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), height=400)

with col2:
    st.subheader("Prediction Result")
    if st.button("Generate Forecast", type="primary"):
        prediction = model.predict(input_df)[0]
        st.metric(label="Predicted Demand", value=f"{prediction:.2f} MW")

        # Visual indicator
        st.write("Demand Level Visualization")
        st.progress(min(int(prediction / 5000 * 100), 100))

        if prediction > 3500:
            st.warning("⚠️ High Demand Detected: Consider grid balancing measures.")
        else:
            st.success("✅ Demand is within normal operating range.")

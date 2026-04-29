import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Weather App", layout="wide")

st.title("🌦️ AI Weather Dashboard")

# 🔍 CITY INPUT
city = st.text_input("Enter City Name", "Hyderabad")

# 🌍 Convert city → latitude, longitude
geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
geo_data = requests.get(geo_url).json()

if "results" not in geo_data:
    st.error("City not found")
    st.stop()

lat = geo_data['results'][0]['latitude']
lon = geo_data['results'][0]['longitude']

# 🌦️ Weather API (7 days forecast)
url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m&daily=temperature_2m_max,temperature_2m_min&current_weather=true"
data = requests.get(url).json()

# 📊 DATAFRAME
hourly = data['hourly']

df = pd.DataFrame({
    'time': hourly['time'],
    'temperature': hourly['temperature_2m'],
    'humidity': hourly['relativehumidity_2m']
})

df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour

# 🤖 ML MODEL
X = df[['hour', 'humidity']]
y = df['temperature']

model = RandomForestRegressor()
model.fit(X, y)

# 🔮 PREDICTION
current = data['current_weather']
current_hour = pd.to_datetime(current['time']).hour
current_humidity = df['humidity'].iloc[-1]

prediction = model.predict([[current_hour, current_humidity]])

# 📊 UI LAYOUT
col1, col2, col3 = st.columns(3)

col1.metric("🌡️ Current Temp", f"{current['temperature']}°C")
col2.metric("🤖 Predicted Temp", f"{round(prediction[0],2)}°C")
col3.metric("💨 Wind Speed", f"{current['windspeed']} km/h")

# 📈 CHART
st.subheader("📈 Temperature Trend")
st.line_chart(df[['temperature']])

# 📅 7 DAY FORECAST
st.subheader("📅 7-Day Forecast")

daily = data['daily']

forecast_df = pd.DataFrame({
    'date': daily['time'],
    'max_temp': daily['temperature_2m_max'],
    'min_temp': daily['temperature_2m_min']
})

st.dataframe(forecast_df)

st.bar_chart(forecast_df[['max_temp', 'min_temp']])

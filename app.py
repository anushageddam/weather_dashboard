import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("🌦️ Weather AI Dashboard")

url = "https://api.open-meteo.com/v1/forecast?latitude=17.3850&longitude=78.4867&hourly=temperature_2m,relativehumidity_2m&current_weather=true"
data = requests.get(url).json()

hourly = data['hourly']

df = pd.DataFrame({
    'time': hourly['time'],
    'temperature': hourly['temperature_2m'],
    'humidity': hourly['relativehumidity_2m']
})

df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour

X = df[['hour', 'humidity']]
y = df['temperature']

model = RandomForestRegressor()
model.fit(X, y)

current = data['current_weather']
current_hour = pd.to_datetime(current['time']).hour
current_humidity = df['humidity'].iloc[-1]

prediction = model.predict([[current_hour, current_humidity]])

st.metric("Current Temperature", current['temperature'])
st.metric("Predicted Temperature", round(prediction[0], 2))

st.line_chart(df[['temperature']])

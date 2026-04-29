import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE ----------------
st.set_page_config(page_title="Weather Pro", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather Pro")
st.caption("Real-time + AI Prediction")

# ---------------- SEARCH ----------------
city = st.text_input("🔍 Search City", "Hyderabad")

# ---------------- GEO ----------------
geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}").json()

if "results" not in geo:
    st.error("City not found")
    st.stop()

lat = geo['results'][0]['latitude']
lon = geo['results'][0]['longitude']

# ---------------- WEATHER ----------------
data = requests.get(
    f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m,precipitation&daily=temperature_2m_max,temperature_2m_min&current_weather=true"
).json()

hourly = data['hourly']

df = pd.DataFrame({
    "time": pd.to_datetime(hourly['time']),
    "temp": hourly['temperature_2m'],
    "humidity": hourly['relativehumidity_2m'],
    "rain": hourly['precipitation']
})

df['hour'] = df['time'].dt.hour

# ---------------- ML ----------------
X = df[['hour', 'humidity']]
y = df['temp']

model = RandomForestRegressor()
model.fit(X, y)

current = data['current_weather']
pred = model.predict([[pd.to_datetime(current['time']).hour, df['humidity'].iloc[-1]]])

# ---------------- MAIN ----------------
col1, col2 = st.columns([2,1])

with col1:
    st.metric("📍 City", city)
    st.metric("🌡️ Temperature", f"{current['temperature']}°C")
    st.metric("🤖 Prediction", f"{round(pred[0],2)}°C")

with col2:
    st.metric("💧 Humidity", f"{df['humidity'].iloc[-1]}%")
    st.metric("💨 Wind", f"{current['windspeed']} km/h")

# ---------------- HOURLY (FIXED) ----------------
st.subheader("⏰ Hourly Forecast")

hour_cols = st.columns(6)

for i in range(6):
    with hour_cols[i]:
        st.write(df['time'][i].strftime('%H:%M'))
        st.write(f"🌡️ {df['temp'][i]}°")
        st.write(f"🌧️ {df['rain'][i]}")

# ---------------- GRAPH ----------------
st.subheader("📈 Temperature Trend")
st.line_chart(df.set_index("time")['temp'])

# ---------------- 7 DAY ----------------
st.subheader("📅 7-Day Forecast")

daily = data['daily']
cols = st.columns(7)

for i in range(7):
    with cols[i]:
        st.write(pd.to_datetime(daily['time'][i]).strftime('%a'))
        st.write(f"{daily['temperature_2m_max'][i]}°")
        st.write(f"{daily['temperature_2m_min'][i]}°")

# ---------------- MAP (WORKING) ----------------
st.subheader("📍 Map")

st.map(pd.DataFrame({
    'lat': [lat],
    'lon': [lon]
}), zoom=10)

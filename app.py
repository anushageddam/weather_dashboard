import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Weather Pro", layout="wide")

# ---------------- SIMPLE CLEAN CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🌦️ Weather Pro")
st.caption("AI-powered weather dashboard")

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

# ---------------- MAIN CARD ----------------
col1, col2 = st.columns([2,1])

with col1:
    st.markdown(f"""
    <div class="card">
        <h2>📍 {city}</h2>
        <h1>{current['temperature']}°C</h1>
        <p>🤖 Predicted: {round(pred[0],2)}°C</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        💧 Humidity: {df['humidity'].iloc[-1]}% <br>
        💨 Wind: {current['windspeed']} km/h
    </div>
    """, unsafe_allow_html=True)

# ---------------- HOURLY (FIXED — NO HTML BUG) ----------------
st.subheader("⏰ Hourly Forecast")

hour_cols = st.columns(6)

for i in range(6):
    with hour_cols[i]:
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <b>{df['time'][i].strftime('%H:%M')}</b><br>
            🌡️ {df['temp'][i]}°<br>
            🌧️ {df['rain'][i]}
        </div>
        """, unsafe_allow_html=True)

# ---------------- GRAPH ----------------
st.subheader("📈 Temperature Trend")
st.line_chart(df.set_index("time")['temp'])

# ---------------- 7 DAY ----------------
st.subheader("📅 7-Day Forecast")

daily = data['daily']
cols = st.columns(7)

for i in range(7):
    with cols[i]:
        st.markdown(f"""
        <div class="card" style="text-align:center">
            {pd.to_datetime(daily['time'][i]).strftime('%a')}<br>
            <b>{daily['temperature_2m_max'][i]}°</b><br>
            <small>{daily['temperature_2m_min'][i]}°</small>
        </div>
        """, unsafe_allow_html=True)

# ---------------- MAP (STABLE) ----------------
st.subheader("📍 Map")

st.map(pd.DataFrame({'lat':[lat],'lon':[lon]}), zoom=12)

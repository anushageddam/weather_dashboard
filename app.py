import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import st_folium

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Weather Pro", layout="wide")

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    margin-bottom: 15px;
}

.big-temp {
    font-size: 60px;
    font-weight: bold;
}

.small-text {
    opacity: 0.8;
}

.hourly-box {
    display: flex;
    overflow-x: auto;
    gap: 15px;
    padding: 10px;
}

.hour-card {
    min-width: 90px;
    background: rgba(255,255,255,0.1);
    padding: 10px;
    border-radius: 15px;
    text-align: center;
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
st.markdown(f"""
<div class="card">
    <h2>📍 {city}</h2>
    <div class="big-temp">☀️ {current['temperature']}°C</div>
    <p class="small-text">Predicted: {round(pred[0],2)}°C</p>
</div>
""", unsafe_allow_html=True)

# ---------------- HOURLY SCROLL ----------------
st.markdown("### ⏰ Hourly Forecast")

hour_html = '<div class="hourly-box">'
for i in range(12):
    hour = df.iloc[i]
    hour_html += f"""
    <div class="hour-card">
        <div>{hour['time'].strftime('%H:%M')}</div>
        <div>🌡️ {hour['temp']}°</div>
        <div>🌧️ {hour['rain']}</div>
    </div>
    """
hour_html += "</div>"

st.markdown(hour_html, unsafe_allow_html=True)

# ---------------- GRAPH ----------------
st.markdown("### 📈 Temperature Trend")
st.line_chart(df.set_index("time")['temp'])

# ---------------- 7 DAY ----------------
st.markdown("### 📅 7-Day Forecast")

daily = data['daily']

cols = st.columns(7)
for i in range(7):
    with cols[i]:
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <b>{pd.to_datetime(daily['time'][i]).strftime('%a')}</b>
            <h3>{daily['temperature_2m_max'][i]}°</h3>
            <small>{daily['temperature_2m_min'][i]}°</small>
        </div>
        """, unsafe_allow_html=True)

# ---------------- MAP ----------------
st.markdown("### 📍 Map")

m = folium.Map(location=[lat, lon], zoom_start=12, tiles="CartoDB positron")

folium.Marker(
    [lat, lon],
    tooltip=city,
    popup=f"{city} {current['temperature']}°C"
).add_to(m)

st_folium(m, use_container_width=True, height=500)

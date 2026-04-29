import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import st_folium

# ---------------- PAGE ----------------
st.set_page_config(page_title="Weather Pro", layout="wide")

# ---------------- CSS FIXED ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
}

/* MAIN CARDS */
.metric-card {
    background: rgba(255,255,255,0.25);
    color: white;
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

/* TEXT FIX */
.metric-card h1, .metric-card h2, .metric-card p {
    color: white !important;
}

/* HOURLY SCROLL */
.hour-box {
    display: flex;
    overflow-x: auto;
    gap: 10px;
}

.hour-card {
    min-width: 90px;
    background: rgba(255,255,255,0.3);
    color: white;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
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
    st.markdown(f"""
    <div class="metric-card">
        <h2>📍 {city}</h2>
        <h1 style="font-size:60px">{current['temperature']}°C</h1>
        <p>🤖 Prediction: {round(pred[0],2)}°C</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        💧 Humidity: {df['humidity'].iloc[-1]}%<br>
        💨 Wind: {current['windspeed']} km/h
    </div>
    """, unsafe_allow_html=True)

# ---------------- HOURLY ----------------
st.markdown("### ⏰ Hourly Forecast")

hour_html = '<div class="hour-box">'
for i in range(12):
    hour = df.iloc[i]
    hour_html += f"""
    <div class="hour-card">
        {hour['time'].strftime('%H:%M')}<br>
        🌡️ {hour['temp']}°<br>
        🌧️ {hour['rain']}
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
        <div class="metric-card" style="text-align:center">
            {pd.to_datetime(daily['time'][i]).strftime('%a')}<br>
            <b>{daily['temperature_2m_max'][i]}°</b><br>
            <small>{daily['temperature_2m_min'][i]}°</small>
        </div>
        """, unsafe_allow_html=True)

# ---------------- MAP ----------------
st.markdown("### 📍 Map")

m = folium.Map(location=[lat, lon], zoom_start=12, tiles="CartoDB positron")

folium.Marker(
    [lat, lon],
    popup=f"{city} - {current['temperature']}°C"
).add_to(m)

st_folium(m, use_container_width=True, height=500)

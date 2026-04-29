import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import st_folium

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Weather App", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🌦️ Weather App")
st.markdown("### 🌤️ Real-Time Weather + AI Prediction")
st.caption("Live data • AI prediction • 7-day forecast")

# ---------------- SEARCH ----------------
col_search, _ = st.columns([2,5])
with col_search:
    city = st.text_input("🔍 Search City", "Hyderabad")

# ---------------- GEO API ----------------
geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
geo_data = requests.get(geo_url).json()

if "results" not in geo_data:
    st.error("City not found")
    st.stop()

lat = geo_data['results'][0]['latitude']
lon = geo_data['results'][0]['longitude']

# ---------------- WEATHER API ----------------
url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m,precipitation&daily=temperature_2m_max,temperature_2m_min&current_weather=true"
data = requests.get(url).json()

# ---------------- DATA ----------------
hourly = data['hourly']

df = pd.DataFrame({
    'time': hourly['time'],
    'temperature': hourly['temperature_2m'],
    'humidity': hourly['relativehumidity_2m'],
    'rain': hourly['precipitation']
})

df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour

# ---------------- ML MODEL ----------------
X = df[['hour', 'humidity']]
y = df['temperature']

model = RandomForestRegressor()
model.fit(X, y)

# ---------------- PREDICTION ----------------
current = data['current_weather']
current_hour = pd.to_datetime(current['time']).hour
current_humidity = df['humidity'].iloc[-1]

prediction = model.predict([[current_hour, current_humidity]])
accuracy = model.score(X, y)

# ---------------- MAIN UI ----------------
col1, col2 = st.columns([3,1])

with col1:
    st.markdown(f"""
    <div class="card">
        <h2>📍 {city}</h2>
        <h1 style="font-size:50px">☀️ {current['temperature']}°C</h1>
        <p>🤖 Predicted: {round(prediction[0],2)}°C</p>
        <p>📊 Model Confidence: {round(accuracy*100,2)}%</p>
        <p>🌡️ Feels like: {current['temperature'] + 2}°C</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <h3>Details</h3>
        <p>💧 Humidity: {df['humidity'].iloc[-1]}%</p>
        <p>💨 Wind: {current['windspeed']} km/h</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- GRAPH ----------------
st.markdown("## 📈 Temperature Trend")
st.line_chart(df.set_index('time')['temperature'])

# ---------------- RAIN ----------------
st.markdown("## 🌧️ Rain Prediction (Next 24 Hours)")
st.bar_chart(df.set_index('time')['rain'].head(24))

# ---------------- 7 DAY FORECAST ----------------
st.markdown("## 📅 7-Day Forecast")

daily = data['daily']
forecast_df = pd.DataFrame({
    'date': daily['time'],
    'max': daily['temperature_2m_max'],
    'min': daily['temperature_2m_min']
})

cols = st.columns(7)

for i in range(7):
    with cols[i]:
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <h4>{pd.to_datetime(forecast_df['date'][i]).strftime('%a')}</h4>
            <h2>☀️ {forecast_df['max'][i]}°</h2>
            <p>{forecast_df['min'][i]}°</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FINAL MAP ----------------
st.markdown(f"## 📍 Map View: {city}")

m = folium.Map(
    location=[lat, lon],
    zoom_start=12,
    tiles="CartoDB positron"   # clean, mostly English
)

folium.Marker(
    [lat, lon],
    tooltip=city,
    popup=f"{city} - {current['temperature']}°C"
).add_to(m)

st_folium(m, use_container_width=True, height=500)

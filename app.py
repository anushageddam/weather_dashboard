import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pydeck as pdk

# PAGE CONFIG
st.set_page_config(page_title="Weather App", layout="wide")

# DARK UI
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stToolbar"] {
    display: none;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.title("🌦️ Weather App")
st.markdown("### 🌤️ Real-Time Weather + AI Prediction")
st.caption("Live data • AI prediction • 7-day forecast")

# 🔍 SEARCH BAR
col_search, _ = st.columns([2,5])
with col_search:
    city = st.text_input("🔍 Search City", "Hyderabad")

# 🌍 GET LOCATION
geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
geo_data = requests.get(geo_url).json()

if "results" not in geo_data:
    st.error("City not found")
    st.stop()

lat = geo_data['results'][0]['latitude']
lon = geo_data['results'][0]['longitude']

# 🌦️ WEATHER DATA
url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m,precipitation&daily=temperature_2m_max,temperature_2m_min&current_weather=true"
data = requests.get(url).json()

# 📊 DATAFRAME
hourly = data['hourly']
df = pd.DataFrame({
    'time': hourly['time'],
    'temperature': hourly['temperature_2m'],
    'humidity': hourly['relativehumidity_2m'],
    'rain': hourly['precipitation']
})

df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour

# 🤖 MODEL
X = df[['hour', 'humidity']]
y = df['temperature']

model = RandomForestRegressor()
model.fit(X, y)

# 🔮 PREDICTION
current = data['current_weather']
current_hour = pd.to_datetime(current['time']).hour
current_humidity = df['humidity'].iloc[-1]

prediction = model.predict([[current_hour, current_humidity]])

# 🌡️ MAIN UI
col1, col2 = st.columns([3,1])

with col1:
    st.markdown(f"""
    <div class="card">
        <h2>📍 {city}</h2>
        <h1 style="font-size:50px">{current['temperature']}°C</h1>
        <p>🤖 Predicted: {round(prediction[0],2)}°C</p>
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

# 📈 TEMPERATURE GRAPH
st.subheader("📈 Temperature Trend")
st.line_chart(df[['temperature']])

# 🌧️ RAIN
st.subheader("🌧️ Rain Prediction (Next 24 Hours)")
st.bar_chart(df[['rain']].head(24))

# 📅 7 DAY FORECAST
st.subheader("📅 7-Day Forecast")

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

# 🗺️ MAP (FIXED & WORKING)

st.subheader(f"📍 Map View: {city}")

map_df = pd.DataFrame({
    'lat': [lat],
    'lon': [lon]
})

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=10,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_color='[255, 0, 0]',
            get_radius=300,
        )
    ]
))

import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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

# ---------------- SEARCH ----------------
city = st.text_input("🔍 Search City", "Hyderabad")

# ---------------- GEO ----------------
geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
geo_data = requests.get(geo_url).json()

if "results" not in geo_data:
    st.error("City not found")
    st.stop()

lat = geo_data['results'][0]['latitude']
lon = geo_data['results'][0]['longitude']

# ---------------- WEATHER ----------------
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

# ---------------- ML ----------------
X = df[['hour', 'humidity']]
y = df['temperature']

model = RandomForestRegressor()
model.fit(X, y)

current = data['current_weather']
prediction = model.predict([[pd.to_datetime(current['time']).hour, df['humidity'].iloc[-1]]])

# ---------------- UI ----------------
st.markdown(f"## 📍 {city}")
st.write(f"🌡️ Current: {current['temperature']}°C")
st.write(f"🤖 Predicted: {round(prediction[0],2)}°C")

# ---------------- MAP (FINAL FIX) ----------------
st.markdown("## 📍 Map View")

# TRY FOLIUM FIRST
try:
    import folium
    from streamlit_folium import st_folium

    m = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker([lat, lon], tooltip=city).add_to(m)

    st_folium(m, use_container_width=True, height=500)

except:
    # FALLBACK (ALWAYS WORKS)
    st.warning("Advanced map not loading, showing basic map instead")
    st.map(pd.DataFrame({'lat':[lat],'lon':[lon]}), zoom=12)

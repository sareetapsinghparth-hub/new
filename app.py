import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import textwrap


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🌤️",
    layout="wide"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>
body {
    background-color: #05080d;
}

.stApp {
    background-color: #05080d;
    color: white;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

/* Header */
.skyguard-header {
    background: linear-gradient(135deg, #09131e, #050a10);
    border: 1px solid #1d3040;
    border-radius: 15px;
    padding: 20px 25px;
    margin-bottom: 25px;
}

.logo {
    font-size: 28px;
    font-weight: bold;
    letter-spacing: 2px;
}

.logo span {
    color: #45d9ff;
}

.status {
    color: #54e38e;
    font-size: 13px;
    margin-top: 5px;
}

/* Hero */
.hero-small {
    color: #45d9ff;
    font-size: 12px;
    letter-spacing: 2px;
}

.hero-title {
    font-size: 45px;
    font-weight: bold;
    line-height: 1.1;
    margin-top: 5px;
}

.hero-title span {
    color: #45d9ff;
}

.hero-text {
    color: #7d8c9b;
    font-size: 16px;
    margin-top: 10px;
    margin-bottom: 25px;
}

/* Sensor cards */
.sensor-card {
    background: linear-gradient(145deg, #0b141e, #070d14);
    border: 1px solid #1c2d3d;
    border-radius: 12px;
    padding: 20px;
}

.sensor-name {
    color: #718292;
    font-size: 12px;
    letter-spacing: 1.5px;
}

.sensor-value {
    font-size: 34px;
    margin-top: 10px;
    font-weight: bold;
}

.sensor-unit {
    color: #738494;
    font-size: 14px;
}

.normal {
    color: #54e38e;
    font-size: 12px;
    margin-top: 8px;
}

/* Panels */
.panel {
    background: #0a1018;
    border: 1px solid #192a39;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}

.panel-title {
    color: #7c8d9d;
    font-size: 12px;
    letter-spacing: 2px;
    margin-bottom: 15px;
}

/* Alert */
.alert {
    background: #170d11;
    border-left: 4px solid #ff5263;
    padding: 15px;
    border-radius: 7px;
    margin-bottom: 10px;
}

.alert-title {
    color: #ff6877;
    font-weight: bold;
}

.alert-text {
    color: #8796a4;
    font-size: 13px;
    margin-top: 5px;
}

/* AI box */
.ai-box {
    background: #0b1119;
    border: 1px solid #273747;
    border-radius: 10px;
    padding: 20px;
}

.ai-score {
    font-size: 42px;
    color: #ff5c6c;
    font-weight: bold;
}

.ai-label {
    color: #ff5c6c;
    font-size: 12px;
    letter-spacing: 1px;
}

/* Footer */
.footer {
    text-align: center;
    color: #526272;
    font-size: 11px;
    margin-top: 40px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# GENERATE SIMULATED AWS DATA
# ============================================================

np.random.seed(42)

number_of_points = 100

timestamps = [
    datetime.now() - timedelta(minutes=(number_of_points - i) * 5)
    for i in range(number_of_points)
]

x = np.linspace(0, 10, number_of_points)

temperature = (
    29
    + 3 * np.sin(x)
    + np.random.normal(0, 0.4, number_of_points)
)

pressure = (
    1008
    + 4 * np.cos(x / 2)
    + np.random.normal(0, 0.7, number_of_points)
)

humidity = (
    65
    - 7 * np.sin(x)
    + np.random.normal(0, 1.2, number_of_points)
)


# Inject an artificial anomaly near the end
temperature[-10:-6] += 20
humidity[-10:-6] += 18
pressure[-10:-6] -= 12


data = pd.DataFrame({
    "Time": timestamps,
    "Temperature": temperature,
    "Pressure": pressure,
    "Humidity": humidity
})


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="skyguard-header">
    <div class="logo">SKY<span>GUARD</span> AI</div>
    <div class="status">● OBSERVATION NETWORK ONLINE</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero-small">INTELLIGENT WEATHER OBSERVATION</div>
<div class="hero-title">Trust every <span>weather reading.</span></div>
<div class="hero-text">
    AI-powered anomaly detection for Automatic Weather Stations.
    Monitor sensor health and identify suspicious observations in real time.
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("SKYGUARD CONTROL")

station = st.sidebar.selectbox(
    "Select AWS Station",
    [
        "AWS-DEL-042",
        "AWS-MUM-017",
        "AWS-BLR-031",
        "AWS-HYD-022"
    ]
)

sensitivity = st.sidebar.slider(
    "Anomaly Sensitivity",
    0.50,
    0.95,
    0.72
)

st.sidebar.markdown("---")

st.sidebar.write("SYSTEM")

st.sidebar.success("ML ENGINE ONLINE")

st.sidebar.write("Model: Isolation Forest")
st.sidebar.write("Mode: Multivariate")
st.sidebar.write("Input: T / P / RH")


# ============================================================
# SENSOR VALUES
# ============================================================

latest = data.iloc[-1]

c1, c2, c3, c4 = st.columns(4)


with c1:
    st.markdown(f"""
    <div class="sensor-card">
        <div class="sensor-name">TEMPERATURE</div>
        <div class="sensor-value">{latest["Temperature"]:.1f} <span class="sensor-unit">°C</span></div>
        <div class="normal">● SENSOR ONLINE</div>
    </div>
    """, unsafe_allow_html=True)


with c2:
    st.markdown(f"""
    <div class="sensor-card">
        <div class="sensor-name">ATMOSPHERIC PRESSURE</div>
        <div class="sensor-value">{latest["Pressure"]:.1f} <span class="sensor-unit">hPa</span></div>
        <div class="normal">● SENSOR ONLINE</div>
    </div>
    """, unsafe_allow_html=True)


with c3:
    st.markdown(f"""
    <div class="sensor-card">
        <div class="sensor-name">RELATIVE HUMIDITY</div>
        <div class="sensor-value">{latest["Humidity"]:.1f} <span class="sensor-unit">%</span></div>
        <div class="normal">● SENSOR ONLINE</div>
    </div>
    """, unsafe_allow_html=True)


with c4:
    st.markdown("""
    <div class="sensor-card">
        <div class="sensor-name">SENSOR HEALTH</div>
        <div class="sensor-value">96%</div>
        <div class="normal">● HEALTHY</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# GRAPH
# ============================================================

st.markdown("""
<div class="panel">
    <div class="panel-title">LIVE TELEMETRY</div>
</div>
""", unsafe_allow_html=True)


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data["Time"],
        y=data["Temperature"],
        name="Temperature",
        mode="lines"
    )
)

fig.add_trace(
    go.Scatter(
        x=data["Time"],
        y=data["Humidity"],
        name="Humidity",
        mode="lines",
        yaxis="y2"
    )
)

# Highlight anomaly region

fig.add_vrect(
    x0=data["Time"].iloc[-10],
    x1=data["Time"].iloc[-6],
    fillcolor="red",
    opacity=0.12,
    line_width=0
)

fig.update_layout(
    height=420,
    paper_bgcolor="#0a1018",
    plot_bgcolor="#0a1018",
    font=dict(color="#8a9aaa"),
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis=dict(gridcolor="#172532"),
    yaxis=dict(title="Temperature °C", gridcolor="#172532"),
    yaxis2=dict(title="Humidity %", overlaying="y", side="right"),
    legend=dict(orientation="h")
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# AI ANALYSIS + ALERTS
# ============================================================

left, right = st.columns([1, 1])


with left:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">AI ANOMALY ANALYSIS</div>
        <div class="ai-box">
            <div class="ai-score">91%</div>
            <div class="ai-label">HIGH ANOMALY CONFIDENCE</div>
            <br>
            <b>Likely cause</b>
            <p style="color:#7d8d9c;">Temperature sensor malfunction</p>
            <b>Why was this flagged?</b>
            <p style="color:#7d8d9c;">
                Temperature changed significantly from its
                learned temporal pattern while humidity and
                pressure also became inconsistent.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


with right:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">ANOMALY EVENT STREAM</div>
        <div class="alert">
            <div class="alert-title">🔴 HIGH — Temperature spike</div>
            <div class="alert-text">Temperature exceeded expected range. Confidence: 91%</div>
        </div>
        <div class="alert">
            <div class="alert-title">🔴 HIGH — Multivariate inconsistency</div>
            <div class="alert-text">Temperature / humidity relationship differs from learned behaviour.</div>
        </div>
        <div class="alert">
            <div class="alert-title">🟡 MEDIUM — Pressure deviation</div>
            <div class="alert-text">Sudden pressure change detected.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SENSOR HEALTH
# ============================================================

st.markdown("""
<div class="panel">
    <div class="panel-title">SENSOR HEALTH MATRIX</div>
</div>
""", unsafe_allow_html=True)


health = pd.DataFrame({
    "Sensor": [
        "Temperature",
        "Pressure",
        "Humidity",
        "Communication"
    ],
    "Health": [
        91,
        98,
        95,
        99
    ]
})

st.dataframe(
    health,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RAW DATA
# ============================================================

with st.expander("VIEW RAW AWS DATA"):
    st.dataframe(
        data.tail(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    SKYGUARD AI<br>
    SIH 2026 • Intelligent AWS Anomaly Detection
</div>
""", unsafe_allow_html=True)
import html
import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import inference as inf
from edge import EDGE_MODEL_ID
from features import SENSOR_COLS, STATION_ORDER
from live_meteo import fetch_open_meteo_safe

attach_xai = inf.attach_xai
run_inference = inf.run_inference
run_inference_batch = inf.run_inference_batch

# Schematic NCR cluster for the live map (zone layout, not surveyed siting).
STATION_GEO = {
    "AWS-N01": {"lat": 28.704, "lon": 77.209, "zone": "North"},
    "AWS-S01": {"lat": 28.527, "lon": 77.209, "zone": "South"},
    "AWS-E01": {"lat": 28.613, "lon": 77.359, "zone": "East"},
    "AWS-W01": {"lat": 28.613, "lon": 77.078, "zone": "West"},
}

ROOT = Path(__file__).resolve().parent
TEST_PATH = ROOT / "data" / "aws_test_injected.csv"


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

.warn {
    color: #ff6877;
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
    font-weight: bold;
}

.ai-label {
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

/* Sensor health matrix */
.health-matrix {
    background: #0a1018;
    border: 1px solid #2a4a5e;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}

.health-matrix-title {
    color: #45d9ff;
    font-size: 13px;
    letter-spacing: 2px;
    margin-bottom: 6px;
}

.health-matrix-sub {
    color: #7d8c9b;
    font-size: 13px;
    margin-bottom: 16px;
}

.health-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

.health-card {
    background: #0b141e;
    border-radius: 10px;
    padding: 14px 16px;
}

.health-card-name {
    color: #8a9aaa;
    font-size: 11px;
    letter-spacing: 1.5px;
}

.health-card-value {
    font-size: 28px;
    font-weight: bold;
    margin-top: 8px;
}

.health-card-status {
    font-size: 11px;
    letter-spacing: 1px;
    margin-top: 6px;
}

.health-bar-wrap {
    background: #172532;
    height: 6px;
    border-radius: 4px;
    margin-top: 10px;
    overflow: hidden;
}

.health-bar {
    height: 6px;
    border-radius: 4px;
}

@media (max-width: 900px) {
    .health-grid {
        grid-template-columns: 1fr 1fr;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD REAL AWS DATA
# ============================================================

@st.cache_data
def load_test_frame() -> pd.DataFrame:
    df = pd.read_csv(TEST_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "station" not in df.columns and "station_id" in df.columns:
        df["station"] = df["station_id"]
    return df.sort_values(["station", "timestamp"]).reset_index(drop=True)


@st.cache_data(show_spinner="Scoring observation network…")
def score_network(decision_threshold: float) -> dict[str, tuple[pd.DataFrame, list[dict]]]:
    full = load_test_frame()
    try:
        all_results = run_inference_batch(
            full,
            decision_threshold=decision_threshold,
            network=full,
        )
    except TypeError:
        all_results = run_inference_batch(full, decision_threshold=decision_threshold)
    packed: dict[str, tuple[pd.DataFrame, list[dict]]] = {}
    for stn in STATION_ORDER:
        data = full.loc[full["station"] == stn].copy().reset_index(drop=True)
        results = [r for r in all_results if str(r.get("station_id")) == stn]
        packed[stn] = (data, results)
    return packed


def fmt_num(val, unit: str = "") -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{float(val):.1f}{unit}"


def result_at_time(results: list[dict], ts: pd.Timestamp) -> dict:
    ts = pd.Timestamp(ts)
    for rec in reversed(results):
        if pd.Timestamp(rec["timestamp"]) <= ts:
            return rec
    return results[0]


def live_map_figure(status_by_station: dict[str, dict], selected: str) -> go.Figure:
    lats, lons, colors, sizes, texts, names = [], [], [], [], [], []
    for stn, geo in STATION_GEO.items():
        rec = status_by_station[stn]
        flag = bool(rec.get("anomaly"))
        colors.append("#ff5263" if flag else "#1a9f5c")
        sizes.append(18 if stn == selected else 12)
        names.append(f"{stn} · {geo['zone']}")
        texts.append(
            f"{stn} ({geo['zone']}) · Delhi–NCR<br>"
            f"{rec.get('anomaly_type', 'none')} · health {rec.get('sensor_health', 0):.0f}%"
        )
        lats.append(geo["lat"])
        lons.append(geo["lon"])
    fig = go.Figure(
        go.Scattermap(
            lat=lats,
            lon=lons,
            text=names,
            mode="markers+text",
            textposition="top right",
            textfont=dict(size=11, color="#0b1a28"),
            marker=dict(size=sizes, color=colors, opacity=0.95),
            hovertext=texts,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=28.613, lon=77.209),
            zoom=10.4,
        ),
        height=420,
        paper_bgcolor="#0a1018",
        plot_bgcolor="#0a1018",
        margin=dict(l=0, r=0, t=8, b=0),
        font=dict(color="#1a1a1a"),
        uirevision="delhi-ncr-osm",
    )
    return fig


def sensor_status_html(ok: bool) -> str:
    if ok:
        return '<div class="normal">● SENSOR ONLINE</div>'
    return '<div class="warn">● ANOMALY</div>'


def health_status_html(health: float, anomaly: bool) -> str:
    if anomaly or health < 70:
        return '<div class="warn">● DEGRADED</div>'
    return '<div class="normal">● HEALTHY</div>'


def per_sensor_health(latest: dict, latest_row: pd.Series) -> pd.DataFrame:
    base = float(latest["sensor_health"])
    atype = latest["anomaly_type"]
    t_h, p_h, rh_h = base, base, base
    if atype == "missing":
        t_h = p_h = rh_h = 18.0
        comm = 35.0
    else:
        comm = 99.0
        if atype in {"spike", "false_spike", "frozen", "drift"}:
            t_h = max(12.0, base - 8)
        if atype == "multivariate":
            rh_h = max(12.0, base - 10)
            t_h = min(98.0, base + 8)
    if pd.isna(latest_row.get("temperature_c")):
        t_h = 10.0
    if pd.isna(latest_row.get("pressure_hpa")):
        p_h = 10.0
    if pd.isna(latest_row.get("humidity_pct")):
        rh_h = 10.0
    return pd.DataFrame(
        {
            "Sensor": ["Temperature", "Pressure", "Humidity", "Communication"],
            "Health": [round(t_h), round(p_h), round(rh_h), round(comm)],
        }
    )


def health_tone(score: float) -> tuple[str, str, str]:
    if score >= 85:
        return "#54e38e", "#14301f", "HEALTHY"
    if score >= 70:
        return "#e3c454", "#2a2814", "WATCH"
    return "#ff6877", "#2a1218", "DEGRADED"


def health_matrix_html(health: pd.DataFrame) -> str:
    cards = []
    for _, row in health.iterrows():
        name = html.escape(str(row["Sensor"]))
        score = int(row["Health"])
        color, bg, label = health_tone(score)
        width = max(4, min(100, score))
        cards.append(
            f'<div class="health-card" style="border-left:4px solid {color};background:{bg};">'
            f'<div class="health-card-name">{name.upper()}</div>'
            f'<div class="health-card-value" style="color:{color};">{score}%</div>'
            f'<div class="health-card-status" style="color:{color};">● {label}</div>'
            f'<div class="health-bar-wrap"><div class="health-bar" style="width:{width}%;background:{color};"></div></div>'
            f"</div>"
        )
    inner = "".join(cards)
    return (
        '<div class="health-matrix">'
        '<div class="health-matrix-title">SENSOR HEALTH MATRIX</div>'
        '<div class="health-matrix-sub">Per-sensor fitness from the latest observation — green healthy, amber watch, red degraded.</div>'
        f'<div class="health-grid">{inner}</div>'
        "</div>"
    )


def anomaly_spans(times: pd.Series, flags: list[bool]) -> list[tuple]:
    spans = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        if (not flag or i == len(flags) - 1) and start is not None:
            end = i if flag and i == len(flags) - 1 else i - 1
            spans.append((times.iloc[start], times.iloc[max(start, end)]))
            start = None
    return spans


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
    STATION_ORDER,
)

sensitivity = st.sidebar.slider(
    "Anomaly Sensitivity",
    0.50,
    0.95,
    0.72,
)

# Slider default 0.72 matches IsolationForest.predict (decision_function < 0).
# Higher sensitivity raises the threshold so more points are flagged.
decision_threshold = float(sensitivity - 0.72)

network = score_network(decision_threshold)
data_full, results_full = network[station]
n = len(data_full)

st.sidebar.markdown("---")
st.sidebar.write("DATA SOURCE")
source = st.sidebar.radio(
    "What the AI is using",
    ["History data", "Live Open-Meteo"],
    help="History data is the station archive the model learned from. Live Open-Meteo is current weather for Delhi–NCR.",
)
st.sidebar.caption(
    "History data is past AWS readings the AI was trained to understand. "
    "Live Open-Meteo is current Delhi–NCR weather scored against that history."
)

st.sidebar.markdown("---")
st.sidebar.write("LIVE MAP")
live = False
idx = n - 1
if source == "History data":
    live = st.sidebar.toggle("Play through history", value=False)
    if live:
        if st.session_state.get("_live_armed") is not True:
            st.session_state.play_idx = 24 if n > 24 else 0
            st.session_state._live_armed = True
        idx = int(st.session_state.get("play_idx", 0))
        idx = min(n - 1, idx + 8)
        st.session_state.play_idx = idx
        st.sidebar.caption(str(data_full["timestamp"].iloc[idx]))
    else:
        st.session_state._live_armed = False
        idx = st.sidebar.slider("Point in history", 0, max(n - 1, 0), n - 1)

data = data_full.iloc[: idx + 1].copy()
results = results_full[: idx + 1]
latest_row = data.iloc[-1]
latest = results[-1]
clock = pd.Timestamp(latest_row["timestamp"])
status_by_station = {
    stn: result_at_time(network[stn][1], clock) for stn in STATION_ORDER
}

if source == "Live Open-Meteo":
    live_df, live_err = fetch_open_meteo_safe()
    if live_df is None:
        st.sidebar.error(live_err)
    else:
        st.sidebar.success("Open-Meteo current weather")
        status_by_station = {}
        for stn in STATION_ORDER:
            hist = network[stn][0].tail(80).copy()
            row = live_df.loc[live_df["station"] == stn].iloc[0]
            rec = run_inference(row, hist, network=live_df)
            if stn == station:
                rec = attach_xai(rec, pd.concat([hist, pd.DataFrame([row])], ignore_index=True))
            else:
                rec["lime_reason"] = "Select this station to compute LIME."
            status_by_station[stn] = rec
        latest_row = live_df.loc[live_df["station"] == station].iloc[0]
        latest = status_by_station[station]
        data = pd.concat([data, pd.DataFrame([latest_row])], ignore_index=True)
        results = list(results) + [latest]
        clock = pd.Timestamp(latest_row["timestamp"])
else:
    latest = attach_xai(latest, data)

anomaly_flags = [bool(r["anomaly"]) for r in results]
events = [r for r in results if r["anomaly"]][-5:][::-1]

st.sidebar.markdown("---")
st.sidebar.write("SYSTEM")
st.sidebar.success("ML ENGINE ONLINE")
st.sidebar.write("Model: Isolation Forest")
st.sidebar.write(f"Edge: {EDGE_MODEL_ID}")
st.sidebar.write("Mode: Multivariate")
st.sidebar.write("Input: T / P / RH")
st.sidebar.write(f"Scored rows: {len(results)}")
st.sidebar.write(f"Flagged: {sum(anomaly_flags)}")
if live and idx < n - 1:
    time.sleep(0.35)
    st.rerun()


# ============================================================
# SENSOR VALUES
# ============================================================

c1, c2, c3, c4 = st.columns(4)
temp_ok = not latest["anomaly"] or latest["anomaly_type"] not in {"spike", "false_spike", "frozen", "missing"}
pres_ok = not latest["anomaly"] or latest["anomaly_type"] not in {"missing"}
hum_ok = not latest["anomaly"] or latest["anomaly_type"] not in {"multivariate", "missing"}
t_val = latest_row["temperature_c"]
p_val = latest_row["pressure_hpa"]
h_val = latest_row["humidity_pct"]
t_txt = "—" if pd.isna(t_val) else f"{float(t_val):.1f}"
p_txt = "—" if pd.isna(p_val) else f"{float(p_val):.1f}"
h_txt = "—" if pd.isna(h_val) else f"{float(h_val):.1f}"
health_txt = f"{latest['sensor_health']:.0f}%"
maint_msg = latest.get("maintenance_message", "stable - no maintenance trend")

with c1:
    st.markdown(f"""
    <div class="sensor-card">
        <div class="sensor-name">TEMPERATURE</div>
        <div class="sensor-value">{t_txt} <span class="sensor-unit">°C</span></div>
        {sensor_status_html(temp_ok and pd.notna(t_val))}
    </div>
    """, unsafe_allow_html=True)


with c2:
    st.markdown(f"""
    <div class="sensor-card">
        <div class="sensor-name">ATMOSPHERIC PRESSURE</div>
        <div class="sensor-value">{p_txt} <span class="sensor-unit">hPa</span></div>
        {sensor_status_html(pres_ok and pd.notna(p_val))}
    </div>
    """, unsafe_allow_html=True)


with c3:
    st.markdown(f"""
    <div class="sensor-card">
        <div class="sensor-name">RELATIVE HUMIDITY</div>
        <div class="sensor-value">{h_txt} <span class="sensor-unit">%</span></div>
        {sensor_status_html(hum_ok and pd.notna(h_val))}
    </div>
    """, unsafe_allow_html=True)


with c4:
    st.markdown(f"""
    <div class="sensor-card">
        <div class="sensor-name">SENSOR HEALTH</div>
        <div class="sensor-value">{health_txt}</div>
        {health_status_html(latest["sensor_health"], latest["anomaly"])}
        <div class="sensor-name" style="margin-top:10px;">{html.escape(maint_msg)}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# GRAPH
# ============================================================

st.markdown("""
<div class="panel">
    <div class="panel-title">LOCATION</div>
    <div style="font-size:26px;font-weight:bold;letter-spacing:0.5px;margin-top:6px;">Delhi–NCR, India</div>
    <div style="color:#7d8c9b;font-size:15px;margin-top:8px;">
        Demo Automatic Weather Station cluster (North · South · East · West).
        Map and live Open-Meteo weather use these Delhi-region coordinates — not Bangalore.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="panel">
    <div class="panel-title">LIVE MAP · OBSERVATION NETWORK · DELHI–NCR</div>
</div>
""", unsafe_allow_html=True)

map_col, telem_col = st.columns([1, 1.35])
with map_col:
    st.plotly_chart(live_map_figure(status_by_station, station), width="stretch")
    st.caption("OpenStreetMap · Delhi–NCR · Green = healthy · Red = flagged · Larger marker = selected station")

with telem_col:
    st.markdown(
        '<div class="panel-title">HISTORY DATA · WHAT THE AI LEARNED FROM · DELHI–NCR</div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["temperature_c"],
            name="Temperature",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["humidity_pct"],
            name="Humidity",
            mode="lines",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["pressure_hpa"],
            name="Pressure",
            mode="lines",
            yaxis="y3",
            line=dict(dash="dot"),
        )
    )
    anom_times = data.loc[anomaly_flags, "timestamp"]
    if len(anom_times):
        fig.add_trace(
            go.Scatter(
                x=anom_times,
                y=data.loc[anomaly_flags, "temperature_c"],
                name="Anomaly",
                mode="markers",
                marker=dict(color="#ff5263", size=8, symbol="x"),
            )
        )
    for x0, x1 in anomaly_spans(data["timestamp"], anomaly_flags):
        fig.add_vrect(x0=x0, x1=x1, fillcolor="red", opacity=0.12, line_width=0)
    fig.update_layout(
        height=380,
        paper_bgcolor="#0a1018",
        plot_bgcolor="#0a1018",
        font=dict(color="#8a9aaa"),
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(gridcolor="#172532"),
        yaxis=dict(title="Temperature °C", gridcolor="#172532"),
        yaxis2=dict(title="Humidity %", overlaying="y", side="right"),
        yaxis3=dict(
            title="Pressure hPa",
            overlaying="y",
            side="right",
            position=0.95,
            showgrid=False,
        ),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "History data: past station readings the AI used to learn normal vs faulty behaviour. "
        "Each prediction is explained from this history (SHAP + LIME)."
    )


# ============================================================
# AI ANALYSIS + ALERTS
# ============================================================

left, right = st.columns([1, 1])

conf = latest["confidence"]
score_color = "#ff5c6c" if latest["anomaly"] else "#54e38e"
label = (
    f"{latest['severity']} ANOMALY CONFIDENCE"
    if latest["anomaly"]
    else "INLIER — LOW ANOMALY CONFIDENCE"
)
cause = latest["anomaly_type"].replace("_", " ") if latest["anomaly"] else "none"
reason = latest["reason"]
if latest["anomaly"] and latest.get("impute_method") not in {None, "none", "unavailable"}:
    heal_line = (
        f"Reported {fmt_num(latest.get('reported_temperature_c'))} °C → "
        f"suggested {fmt_num(latest.get('suggested_temperature_c'))} °C "
        f"({latest.get('impute_method')}). "
        f"P {fmt_num(latest.get('suggested_pressure_hpa'))} hPa · "
        f"RH {fmt_num(latest.get('suggested_humidity_pct'))} %."
    )
else:
    heal_line = "No correction — reading treated as usable."
edge_line = (
    f"Edge ({EDGE_MODEL_ID}): "
    f"{'flag' if latest.get('edge_anomaly') else 'clear'} / "
    f"{latest.get('edge_anomaly_type', 'none')}"
)

with left:
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">AI ANOMALY ANALYSIS</div>
        <div class="ai-box">
            <div class="ai-score" style="color:{score_color};">{conf:.0f}%</div>
            <div class="ai-label" style="color:{score_color};">{label}</div>
            <br>
            <b>Likely cause</b>
            <p style="color:#7d8d9c;">{cause}</p>
            <b>Why was this flagged? (SHAP)</b>
            <p style="color:#7d8d9c;">{html.escape(str(reason))}</p>
            <b>LIME</b>
            <p style="color:#7d8d9c;">{html.escape(str(latest.get('lime_reason', '')))}</p>
            <b>Corrected value</b>
            <p style="color:#7d8d9c;">{html.escape(heal_line)}</p>
            <b>Maintenance</b>
            <p style="color:#7d8d9c;">{html.escape(str(latest.get('maintenance_message', '')))}</p>
            <b>Edge profile</b>
            <p style="color:#7d8d9c;">{html.escape(edge_line)}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def alert_card(title: str, text: str) -> str:
    # Keep this compact: indented HTML after a blank line becomes a Markdown code block.
    return (
        f'<div class="alert">'
        f'<div class="alert-title">{html.escape(title)}</div>'
        f'<div class="alert-text">{html.escape(text)}</div>'
        f"</div>"
    )


if not events:
    alert_html = alert_card(
        "🟢 CLEAR — No recent anomalies",
        "IsolationForest has not flagged the latest window at this sensitivity.",
    )
else:
    cards = []
    for ev in events[:3]:
        icon = "🔴" if ev["severity"] == "HIGH" else ("🟡" if ev["severity"] == "MEDIUM" else "🟠")
        title = f"{icon} {ev['severity']} — {ev['anomaly_type'].replace('_', ' ')}"
        text = f"{ev['timestamp']} · Confidence: {ev['confidence']:.0f}% · {ev['reason']}"
        cards.append(alert_card(title, text))
    alert_html = "".join(cards)

with right:
    st.markdown(
        '<div class="panel"><div class="panel-title">ANOMALY EVENT STREAM</div>'
        f"{alert_html}</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# SENSOR HEALTH
# ============================================================

health = per_sensor_health(latest, latest_row)
st.markdown(health_matrix_html(health), unsafe_allow_html=True)


# ============================================================
# RAW DATA
# ============================================================

raw_tail = data.tail(20)[["timestamp", "station", *SENSOR_COLS]].copy()
if "is_anomaly" in data.columns:
    raw_tail["is_anomaly"] = data.tail(20)["is_anomaly"].values
raw_tail["detected"] = anomaly_flags[-20:]
raw_tail["anomaly_type"] = [r["anomaly_type"] for r in results[-20:]]

with st.expander("VIEW HISTORY DATA"):
    st.dataframe(
        raw_tail,
        width="stretch",
        hide_index=True,
    )
    st.caption("Latest inference JSON")
    st.code(json.dumps(latest, indent=2), language="json")


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    SKYGUARD AI<br>
    SIH 2026 • Intelligent AWS Anomaly Detection
</div>
""", unsafe_allow_html=True)

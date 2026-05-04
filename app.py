"""
CNN-GRU Temperature Forecasting — Streamlit App
Using ONNX Runtime (works on all Python versions including 3.14)
================================================
Files needed in same folder:
    cnn_gru_model.onnx   ← converted from .keras using convert_model.py
    scaler_X.pkl
    scaler_y.pkl
    model_config.json
    Weather.csv

Run with:
    streamlit run app.py
"""

import json, pickle, datetime, warnings, requests
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CNN-GRU Temperature Forecast",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

C_NAVY   = "#1F4E79"
C_BLUE   = "#2E75B6"
C_GREEN  = "#70AD47"
C_ORANGE = "#ED7D31"

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1F4E79; }
    [data-testid="stSidebar"] * { color: white !important; }
    .horizon-card {
        background: #f7faff;
        border: 2px solid #2E75B6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 6px;
    }
    .horizon-temp  { font-size: 2rem;  font-weight: 700; color: #1F4E79; }
    .horizon-label { font-size: 0.9rem; color: #555; margin-top: 6px; }
    .section-hdr {
        background: linear-gradient(90deg, #1F4E79, #2E75B6);
        color: white; padding: 10px 18px; border-radius: 8px;
        font-size: 1.05rem; font-weight: 700; margin: 18px 0 10px 0;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white; border: none; border-radius: 8px;
        padding: 12px 32px; font-size: 1.05rem; font-weight: 700;
        width: 100%;
    }
    .fetch-success {
        background: #f0fff4; border: 2px solid #70AD47;
        border-radius: 8px; padding: 12px; margin: 8px 0;
        color: #375623; font-weight: 600;
    }
    .fetch-error {
        background: #fff5f5; border: 2px solid #e74c3c;
        border-radius: 8px; padding: 12px; margin: 8px 0;
        color: #c0392b;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# OPEN-METEO WEATHER FETCHER
# ════════════════════════════════════════════════════════════
def fetch_weather_open_meteo(date, hour, lat=-17.8252, lon=31.0335):
    date_str = date.strftime("%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": date_str,
        "end_date":   date_str,
        "hourly": ",".join([
            "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
            "windspeed_10m", "surface_pressure", "shortwave_radiation",
            "precipitation", "cloudcover",
        ]),
        "timezone": "Africa/Harare",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        h = resp.json().get("hourly", {})
        return {
            "Temperature_C":       round(h["temperature_2m"][hour], 2),
            "Humidity_pct":        round(h["relativehumidity_2m"][hour], 1),
            "Dew_Point_C":         round(h["dewpoint_2m"][hour], 2),
            "Wind_Speed_ms":       round(h["windspeed_10m"][hour] / 3.6, 2),
            "Pressure_hPa":        round(h["surface_pressure"][hour], 2),
            "Solar_Radiation_Wm2": round(h["shortwave_radiation"][hour], 1),
            "Precipitation_mm":    round(h["precipitation"][hour], 1),
            "Cloud_Cover_pct":     int(h["cloudcover"][hour]),
        }, None
    except requests.exceptions.ConnectionError:
        return None, "No internet connection."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Try again."
    except Exception as e:
        return None, f"Error: {e}"


# ════════════════════════════════════════════════════════════
# ONNX MODEL LOADER — no TensorFlow needed on server
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading CNN-GRU model…")
def load_onnx_model():
    import onnxruntime as ort
    sess = ort.InferenceSession(
        "cnn_gru_model.onnx",
        providers=["CPUExecutionProvider"]
    )
    return sess

@st.cache_resource(show_spinner=False)
def load_scalers():
    with open("scaler_X.pkl", "rb") as f: sx = pickle.load(f)
    with open("scaler_y.pkl", "rb") as f: sy = pickle.load(f)
    return sx, sy

@st.cache_resource(show_spinner=False)
def load_config():
    with open("model_config.json") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_history():
    df = pd.read_csv("Weather.csv", parse_dates=["Timestamp"])
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
INPUT_FEATURES = [
    "Temperature_C", "Humidity_pct", "Wind_Speed_ms", "Pressure_hPa",
    "Solar_Radiation_Wm2", "Dew_Point_C", "Precipitation_mm", "Cloud_Cover_pct",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"
]

def make_feature_row(ts, temp, hum, wind, pres, solar, dew, rain, cloud):
    return {
        "Temperature_C":       temp,
        "Humidity_pct":        hum,
        "Wind_Speed_ms":       wind,
        "Pressure_hPa":        pres,
        "Solar_Radiation_Wm2": solar,
        "Dew_Point_C":         dew,
        "Precipitation_mm":    rain,
        "Cloud_Cover_pct":     cloud,
        "hour_sin":  np.sin(2*np.pi*ts.hour/24),
        "hour_cos":  np.cos(2*np.pi*ts.hour/24),
        "month_sin": np.sin(2*np.pi*ts.month/12),
        "month_cos": np.cos(2*np.pi*ts.month/12),
        "dow_sin":   np.sin(2*np.pi*ts.weekday()/7),
        "dow_cos":   np.cos(2*np.pi*ts.weekday()/7),
    }

def build_sequence(df_hist, current_row, scaler_X, lookback=168):
    hist = df_hist.tail(lookback - 1).copy()
    hist["hour_sin"]  = np.sin(2*np.pi*hist["Hour"]/24)
    hist["hour_cos"]  = np.cos(2*np.pi*hist["Hour"]/24)
    hist["month_sin"] = np.sin(2*np.pi*hist["Month"]/12)
    hist["month_cos"] = np.cos(2*np.pi*hist["Month"]/12)
    hist["dow_sin"]   = np.sin(2*np.pi*hist["Day_of_Week"]/7)
    hist["dow_cos"]   = np.cos(2*np.pi*hist["Day_of_Week"]/7)
    rows = hist[INPUT_FEATURES].values.tolist()
    rows.append([current_row[f] for f in INPUT_FEATURES])
    arr    = np.array(rows, dtype=np.float32)
    arr_sc = scaler_X.transform(arr)
    return arr_sc[np.newaxis, ...].astype(np.float32)

def run_inference(seq, sess, scaler_y):
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_sc = sess.run([output_name], {input_name: seq})[0]
    pred    = scaler_y.inverse_transform(pred_sc)
    return float(pred[0,0]), float(pred[0,1]), float(pred[0,2])


# ════════════════════════════════════════════════════════════
# PLOT HELPERS
# ════════════════════════════════════════════════════════════
def plot_recent(df_hist):
    recent = df_hist.tail(7*24)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(recent["Timestamp"], recent["Temperature_C"],
            lw=1.0, color=C_NAVY, alpha=0.85)
    ax.fill_between(recent["Timestamp"], recent["Temperature_C"],
                    alpha=0.10, color=C_BLUE)
    ax.set_title("Recent 7-Day Temperature History",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Date"); ax.set_ylabel("Temperature (°C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=25)
    fig.tight_layout(); return fig

def plot_forecast_bar(t1h, t3h, t6h, t_input):
    labels = ["Yesterday\n(Input)", "+1 Hour", "+3 Hours", "+6 Hours"]
    values = [t_input, t1h, t3h, t6h]
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_NAVY]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, values, color=colors, width=0.55,
                  edgecolor="white", zorder=3)
    ax.set_ylim(min(values)-3, max(values)+4)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title("Forecast Summary", fontsize=13, fontweight="bold")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.3, f"{v:.1f}°C",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(t_input, color="grey", ls="--", lw=1.2, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout(); return fig

def plot_diurnal(df_hist):
    h = df_hist.groupby("Hour")["Temperature_C"].agg(["mean","std"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(h.index, h["mean"]-h["std"], h["mean"]+h["std"],
                    alpha=0.15, color=C_BLUE)
    ax.plot(h.index, h["mean"], color=C_NAVY, lw=2.2, marker="o", ms=4)
    ax.set_title("Average Diurnal Pattern", fontweight="bold", fontsize=11)
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Temperature (°C)")
    ax.set_xticks(range(0, 24, 3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(); return fig

def plot_monthly(df_hist):
    mnames = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
    data = [df_hist[df_hist["Month"]==m]["Temperature_C"].values
            for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(10, 4))
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="white", lw=2))
    for p in bp["boxes"]:
        p.set_facecolor(C_NAVY); p.set_alpha(0.8)
    ax.set_xticklabels(mnames, fontsize=9)
    ax.set_title("Monthly Temperature Distribution",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Month"); ax.set_ylabel("Temperature (°C)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(); return fig


# ════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════
def main():
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{C_NAVY},{C_BLUE});
                padding:24px 30px;border-radius:14px;margin-bottom:20px;'>
        <h1 style='color:white;margin:0;font-size:1.9rem;'>
            🌡️ CNN-GRU Temperature Forecasting
        </h1>
        <p style='color:#D6E4F0;margin:6px 0 0 0;font-size:1rem;'>
            Short-term temperature forecast &nbsp;|&nbsp;
            Horizons: +1h · +3h · +6h &nbsp;|&nbsp;
            Real weather data via Open-Meteo
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load artefacts ───────────────────────────────────────
    model_ok = True
    try:
        sess               = load_onnx_model()
        scaler_X, scaler_y = load_scalers()
        config             = load_config()
        df_hist            = load_history()
        LOOKBACK           = config.get("lookback", 168)
    except FileNotFoundError as e:
        st.warning(
            f"⚠️ **Demo Mode** — file not found: `{e}`\n\n"
            "Ensure `cnn_gru_model.onnx`, `scaler_X.pkl`, `scaler_y.pkl`, "
            "`model_config.json`, `Weather.csv` are all present."
        )
        model_ok = False
        LOOKBACK = 168
        try:    df_hist = load_history()
        except: df_hist = None

    # ── Session state ────────────────────────────────────────
    if "weather_fetched" not in st.session_state:
        st.session_state.weather_fetched = False
    if "fetched_data" not in st.session_state:
        st.session_state.fetched_data = {}
    if "fetch_status" not in st.session_state:
        st.session_state.fetch_status = ""

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📅 Select Date & Time")
        obs_date = st.date_input(
            "Date to forecast from",
            value=datetime.date.today() - datetime.timedelta(days=1),
            max_value=datetime.date.today() - datetime.timedelta(days=1),
        )
        obs_hour = st.slider("Observation Hour", 0, 23, 15)

        st.markdown("---")
        st.markdown("### 🌐 Auto-Fetch Weather")
        st.markdown(
            "<small style='color:#ccc;'>Fetches real weather from "
            "Open-Meteo — free, no login</small>",
            unsafe_allow_html=True
        )

        location = st.selectbox(
            "Location",
            ["Harare, Zimbabwe", "Bulawayo, Zimbabwe", "Custom coordinates"],
        )
        coords = {
            "Harare, Zimbabwe":   (-17.8252, 31.0335),
            "Bulawayo, Zimbabwe": (-20.1325, 28.6264),
        }
        if location == "Custom coordinates":
            lat = st.number_input("Latitude",  -90.0,  90.0, -17.8252, 0.0001)
            lon = st.number_input("Longitude", -180.0, 180.0, 31.0335, 0.0001)
        else:
            lat, lon = coords[location]

        fetch_btn = st.button("🌐 Fetch Weather Automatically")
        if fetch_btn:
            with st.spinner(f"Fetching weather for {obs_date} {obs_hour}:00…"):
                data, err = fetch_weather_open_meteo(obs_date, obs_hour, lat, lon)
            if data:
                st.session_state.fetched_data    = data
                st.session_state.weather_fetched = True
                st.session_state.fetch_status    = "success"
            else:
                st.session_state.weather_fetched = False
                st.session_state.fetch_status    = err

        if st.session_state.fetch_status == "success":
            st.markdown(
                "<div class='fetch-success'>✅ Weather fetched! "
                "Fields auto-filled below.</div>",
                unsafe_allow_html=True
            )
        elif st.session_state.fetch_status:
            st.markdown(
                f"<div class='fetch-error'>❌ {st.session_state.fetch_status}"
                "</div>", unsafe_allow_html=True
            )

        st.markdown("---")
        fd = st.session_state.fetched_data
        st.markdown("### 🌡️ Weather Conditions")

        temp  = st.number_input("Temperature (°C)",      -5.0,  50.0, float(fd.get("Temperature_C",       22.5)), 0.1)
        hum   = st.number_input("Relative Humidity (%)", 10.0, 100.0, float(fd.get("Humidity_pct",        65.0)), 0.5)
        dew   = st.number_input("Dew Point (°C)",       -10.0,  40.0, float(fd.get("Dew_Point_C",         15.0)), 0.1)
        wind  = st.number_input("Wind Speed (m/s)",       0.0,  30.0, float(fd.get("Wind_Speed_ms",         3.5)), 0.1)
        pres  = st.number_input("Pressure (hPa)",       950.0, 1050.0, float(fd.get("Pressure_hPa",      1013.0)), 0.5)
        solar = st.number_input("Solar Radiation (W/m²)", 0.0, 1000.0, float(fd.get("Solar_Radiation_Wm2", 350.0)), 5.0)
        cloud = st.slider("Cloud Cover (%)", 0, 100, int(fd.get("Cloud_Cover_pct", 40)))
        rain  = st.number_input("Precipitation (mm)",    0.0,  100.0, float(fd.get("Precipitation_mm",     0.0)), 0.1)

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Temperature")

    # ── Tabs ─────────────────────────────────────────────────
    tab_pred, tab_eda, tab_model = st.tabs(
        ["🔮 Forecast", "📊 Data Explorer", "📈 Model Performance"]
    )

    # ════════════════════════════════════════════════════════
    # TAB 1 — FORECAST
    # ════════════════════════════════════════════════════════
    with tab_pred:
        col_info, col_plot = st.columns([1, 1.6])

        with col_info:
            st.markdown('<div class="section-hdr">📍 How It Works</div>',
                        unsafe_allow_html=True)
            st.markdown("""
1. Select a **date** in the sidebar
2. Click **🌐 Fetch Weather Automatically**
3. Click **🔮 Predict Temperature**
4. Get forecasts at **+1h**, **+3h**, **+6h**
            """)
            st.markdown('<div class="section-hdr">⚙️ Model Status</div>',
                        unsafe_allow_html=True)
            if model_ok:
                st.success("✅ CNN-GRU model loaded (ONNX)")
                st.markdown(f"**Lookback:** {LOOKBACK}h (7 days)")
                st.markdown("**Features:** 14 (8 met + 6 cyclic)")
                st.markdown("**Architecture:** Conv1D×2 → MaxPool → GRU×2 → Dense")
            else:
                st.error("⚠️ Demo Mode")

            if st.session_state.weather_fetched:
                st.markdown('<div class="section-hdr">🌐 Fetched Data</div>',
                            unsafe_allow_html=True)
                for k, v in st.session_state.fetched_data.items():
                    st.markdown(f"**{k.replace('_',' ')}:** {v}")

        with col_plot:
            st.markdown('<div class="section-hdr">📉 Recent 7-Day History</div>',
                        unsafe_allow_html=True)
            if df_hist is not None:
                st.pyplot(plot_recent(df_hist), use_container_width=True)

        if predict_btn:
            ts_obs = pd.Timestamp(
                datetime.datetime.combine(obs_date, datetime.time(obs_hour))
            )
            current_row = make_feature_row(
                ts_obs, temp, hum, wind, pres, solar, dew, rain, cloud
            )
            src = "Open-Meteo" if st.session_state.weather_fetched \
                  else "manual input"

            with st.spinner("Running CNN-GRU inference…"):
                if model_ok and df_hist is not None:
                    seq = build_sequence(df_hist, current_row,
                                         scaler_X, LOOKBACK)
                    t1h, t3h, t6h = run_inference(seq, sess, scaler_y)
                else:
                    np.random.seed(42)
                    t1h = temp + np.random.normal(0.4, 0.5)
                    t3h = temp + np.random.normal(1.2, 0.8)
                    t6h = temp + np.random.normal(2.5, 1.2)

            st.success(f"✅ Forecast complete! (Source: {src})")

            st.markdown('<div class="section-hdr">🔮 Forecast Results</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            for col, label, val in [
                (c1, "+1 Hour Ahead",    t1h),
                (c2, "+3 Hours Ahead",   t3h),
                (c3, "+6 Hours Ahead",   t6h),
            ]:
                delta  = val - temp
                arrow  = "▲" if delta >= 0 else "▼"
                dcolor = "#27ae60" if delta >= 0 else "#e74c3c"
                with col:
                    st.markdown(f"""
<div class='horizon-card'>
  <div class='horizon-temp'>{val:.1f}°C</div>
  <div class='horizon-label'>📍 {label}</div>
  <div style='color:{dcolor};font-weight:700;margin-top:8px;'>
    {arrow} {abs(delta):.1f}°C vs input
  </div>
</div>""", unsafe_allow_html=True)

            st.pyplot(plot_forecast_bar(t1h, t3h, t6h, temp),
                      use_container_width=True)

            trend = "RISING"  if t6h > temp + 0.5 else \
                    "FALLING" if t6h < temp - 0.5 else "STABLE"
            st.info(
                f"🌡️ From **{temp:.1f}°C** on {obs_date} at {obs_hour}:00 — "
                f"**{t1h:.1f}°C** (+1h) → **{t3h:.1f}°C** (+3h) → "
                f"**{t6h:.1f}°C** (+6h). Trend: **{trend}**."
            )
        else:
            if not st.session_state.weather_fetched:
                st.markdown(
                    "<br><div style='text-align:center;color:#888;padding:30px;'>"
                    "← Click <b>🌐 Fetch Weather Automatically</b> "
                    "then <b>🔮 Predict Temperature</b></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<br><div style='text-align:center;color:#375623;"
                    "padding:30px;font-weight:600;'>"
                    "✅ Weather fetched! Now click "
                    "<b>🔮 Predict Temperature</b></div>",
                    unsafe_allow_html=True
                )

    # ════════════════════════════════════════════════════════
    # TAB 2 — DATA EXPLORER
    # ════════════════════════════════════════════════════════
    with tab_eda:
        if df_hist is None:
            st.warning("Place Weather.csv in the same folder as app.py.")
        else:
            st.markdown('<div class="section-hdr">📊 Dataset Overview</div>',
                        unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Records",    f"{len(df_hist):,}")
            m2.metric("Mean Temperature", f"{df_hist['Temperature_C'].mean():.1f}°C")
            m3.metric("Min Temperature",  f"{df_hist['Temperature_C'].min():.1f}°C")
            m4.metric("Max Temperature",  f"{df_hist['Temperature_C'].max():.1f}°C")

            ca, cb = st.columns(2)
            with ca: st.pyplot(plot_monthly(df_hist), use_container_width=True)
            with cb: st.pyplot(plot_diurnal(df_hist), use_container_width=True)

            yr = st.selectbox(
                "Filter by Year",
                ["All"] + sorted(df_hist["Year"].unique().tolist(), reverse=True)
            )
            show = df_hist if yr == "All" else df_hist[df_hist["Year"]==int(yr)]
            st.dataframe(show.tail(200), use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 3 — MODEL PERFORMANCE
    # ════════════════════════════════════════════════════════
    with tab_model:
        st.markdown('<div class="section-hdr">📈 Test-Set Metrics</div>',
                    unsafe_allow_html=True)
        results = {
            "1h Ahead": {"RMSE": 1.8472, "MAE": 1.4380, "R²": 0.9611},
            "3h Ahead": {"RMSE": 1.9305, "MAE": 1.5121, "R²": 0.9575},
            "6h Ahead": {"RMSE": 2.0366, "MAE": 1.5970, "R²": 0.9527},
        }
        for horizon, m in results.items():
            st.markdown(f"""
<div style='background:#f7faff;border:2px solid {C_BLUE};
            border-radius:12px;padding:16px;margin:8px 0;'>
  <b style='color:{C_NAVY};font-size:1.05rem;'>{horizon}</b>&nbsp;&nbsp;
  RMSE: <b>{m["RMSE"]}°C</b> &nbsp;|&nbsp;
  MAE: <b>{m["MAE"]}°C</b> &nbsp;|&nbsp;
  R²: <b>{m["R²"]}</b>
</div>""", unsafe_allow_html=True)

        arch = pd.DataFrame({
            "Layer": ["Input","Conv1D(64)","BatchNorm","Conv1D(128)","BatchNorm",
                      "MaxPooling1D","Dropout(0.2)","GRU(128)","Dropout(0.2)",
                      "GRU(64)","Dropout(0.2)","Dense(64)","Dense(32)","Output(3)"],
            "Output Shape": ["(None,168,14)","(None,168,64)","(None,168,64)",
                             "(None,168,128)","(None,168,128)","(None,84,128)",
                             "(None,84,128)","(None,84,128)","(None,84,128)",
                             "(None,64)","(None,64)","(None,64)","(None,32)","(None,3)"],
        })
        st.dataframe(arch, use_container_width=True, hide_index=True)

    st.markdown("""<hr>
<div style='text-align:center;color:#888;font-size:0.85rem;padding:8px 0;'>
  CNN-GRU Temperature Forecasting &nbsp;|&nbsp;
  University of Zimbabwe &nbsp;|&nbsp; 2024 &nbsp;|&nbsp;
  Weather: <a href='https://open-meteo.com' target='_blank'>Open-Meteo</a>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

"""
CNN-GRU Temperature Forecasting — Professional Portable App
Files: cnn_gru_model.onnx, scaler_X.pkl, scaler_y.pkl,
       model_config.json, Weather.csv
Run:  streamlit run app.py
"""
import json, pickle, datetime, warnings, requests
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(
    page_title="CNN-GRU Temperature Forecast",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

C_NAVY  = "#1F4E79"
C_BLUE  = "#2E75B6"
C_GREEN = "#70AD47"
C_ORANGE= "#ED7D31"

st.markdown(f"""<style>
/* ── Global ─────────────────────────────────── */
html, body, [class*="css"] {{ font-size: 14px; }}
.block-container {{
    padding: 1rem 1.5rem 1rem 1.5rem !important;
    max-width: 1200px !important;
}}

/* ── Sidebar ─────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {C_NAVY};
    min-width: 260px !important;
    max-width: 260px !important;
}}
[data-testid="stSidebar"] * {{ color: white !important; }}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {{
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] input {{
    font-size: 0.82rem !important;
    padding: 4px 8px !important;
}}

/* ── Header banner ───────────────────────────── */
.banner {{
    background: linear-gradient(135deg, {C_NAVY}, {C_BLUE});
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 12px;
}}
.banner h2 {{
    color: white; margin: 0;
    font-size: 1.35rem; font-weight: 700;
}}
.banner p {{
    color: #D6E4F0; margin: 3px 0 0;
    font-size: 0.78rem;
}}

/* ── Section header ──────────────────────────── */
.sec {{
    background: linear-gradient(90deg, {C_NAVY}, {C_BLUE});
    color: white; padding: 5px 12px;
    border-radius: 6px; font-size: 0.82rem;
    font-weight: 700; margin: 10px 0 6px;
}}

/* ── Forecast cards ──────────────────────────── */
.fc {{
    background: #f0f6ff;
    border: 2px solid {C_BLUE};
    border-radius: 10px;
    padding: 12px 8px;
    text-align: center;
}}
.fc-temp  {{ font-size: 1.5rem; font-weight: 700; color: {C_NAVY}; }}
.fc-label {{ font-size: 0.75rem; color: #444; margin-top: 2px; }}
.fc-delta {{ font-size: 0.78rem; font-weight: 700; margin-top: 4px; }}

/* ── Metric cards (performance tab) ─────────── */
.mc {{
    background: #f0f6ff;
    border: 2px solid {C_BLUE};
    border-radius: 10px;
    padding: 10px 6px;
    text-align: center;
}}
.mc-h {{ font-size: 0.82rem; font-weight: 700; color: {C_NAVY}; }}
.mc-v {{ font-size: 1.2rem; font-weight: 700; color: {C_NAVY}; margin: 3px 0; }}
.mc-s {{ font-size: 0.7rem; color: #555; }}

/* ── Status badge ────────────────────────────── */
.ok {{
    background: #eafaf1; border: 1.5px solid {C_GREEN};
    border-radius: 7px; padding: 5px 10px;
    color: #1e7e34; font-size: 0.8rem;
    font-weight: 600; margin: 4px 0 8px;
}}

/* ── Fetched weather rows ────────────────────── */
.fw {{
    display: flex; justify-content: space-between;
    align-items: center; padding: 3px 0;
    border-bottom: 1px solid #e8eef5;
    font-size: 0.78rem;
}}
.fw-k {{ color: #666; }}
.fw-v {{ font-weight: 600; color: {C_NAVY}; }}

/* ── Predict button ──────────────────────────── */
div[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, {C_NAVY}, {C_BLUE});
    color: white !important; border: none;
    border-radius: 8px; padding: 9px 18px;
    font-size: 0.9rem; font-weight: 700;
    width: 100%; margin-top: 6px;
    transition: opacity 0.2s;
}}
div[data-testid="stButton"] > button:hover {{ opacity: 0.88; }}

/* ── Tabs ────────────────────────────────────── */
[data-testid="stTabs"] button {{
    font-size: 0.82rem !important;
    padding: 6px 14px !important;
    font-weight: 600 !important;
}}

/* ── Metric widget ───────────────────────────── */
[data-testid="stMetricLabel"] {{ font-size: 0.72rem !important; }}
[data-testid="stMetricValue"] {{ font-size: 1rem !important; }}

/* ── Alert boxes ─────────────────────────────── */
[data-testid="stAlert"] {{ font-size: 0.8rem !important; padding: 8px 12px !important; }}

/* ── Dataframe ───────────────────────────────── */
[data-testid="stDataFrame"] {{ font-size: 0.78rem !important; }}

/* ── Mobile responsive ───────────────────────── */

/* ── SIDEBAR GAPS — target all 3 arrow locations ─────── */
/* Arrow 1: push to very top */
section[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
    margin-top: 0 !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 0.15rem !important;
    padding-bottom: 0 !important;
}}
/* Kill ALL vertical spacing between sidebar widgets */
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
    gap: 0rem !important;
}}
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {{
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}}
[data-testid="stSidebar"] .element-container {{
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
}}
/* Arrow 2: below slider */
[data-testid="stSidebar"] .stSlider {{
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}}
/* Divider lines */
[data-testid="stSidebar"] hr {{
    margin-top: 5px !important;
    margin-bottom: 5px !important;
    border-color: rgba(255,255,255,0.2) !important;
}}
/* Arrow 3: below success alert */
[data-testid="stSidebar"] [data-testid="stAlert"] {{
    padding: 3px 8px !important;
    font-size: 0.74rem !important;
    margin: 2px 0 !important;
    line-height: 1.2 !important;
}}
/* Section headings */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    margin: 3px 0 1px !important;
    padding: 0 !important;
}}
/* Labels */
[data-testid="stSidebar"] label {{
    font-size: 0.78rem !important;
    margin-bottom: 0 !important;
    line-height: 1.2 !important;
}}
/* Input height */
[data-testid="stSidebar"] input {{
    height: 28px !important;
    font-size: 0.8rem !important;
    padding: 2px 7px !important;
}}
[data-testid="stSidebar"] [data-baseweb="input"] {{
    height: 28px !important;
}}
/* Buttons */
[data-testid="stSidebar"] button {{
    padding: 5px 10px !important;
    font-size: 0.82rem !important;
    margin: 2px 0 !important;
}}

@media (max-width: 768px) {{
    .block-container {{ padding: 0.5rem 0.6rem !important; }}
    .banner h2 {{ font-size: 1.05rem !important; }}
    .banner p  {{ font-size: 0.7rem !important; }}
    .fc-temp   {{ font-size: 1.2rem !important; }}
    [data-testid="stSidebar"] {{
        min-width: 100% !important;
        max-width: 100% !important;
    }}
}}
</style>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# OPEN-METEO FETCHER
# ════════════════════════════════════════════════════════════
def fetch_weather(date, hour, lat=-17.8252, lon=31.0335):
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": date.strftime("%Y-%m-%d"),
                "end_date":   date.strftime("%Y-%m-%d"),
                "hourly": ("temperature_2m,relativehumidity_2m,dewpoint_2m,"
                           "windspeed_10m,surface_pressure,shortwave_radiation,"
                           "precipitation,cloudcover"),
                "timezone": "Africa/Harare"
            }, timeout=10)
        r.raise_for_status()
        h = r.json()["hourly"]
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
        return None, "Request timed out."
    except Exception as e:
        return None, f"Error: {e}"


# ════════════════════════════════════════════════════════════
# LOADERS
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading CNN-GRU model…")
def load_model():
    import onnxruntime as ort
    return ort.InferenceSession("cnn_gru_model.onnx",
                                providers=["CPUExecutionProvider"])

@st.cache_resource(show_spinner=False)
def load_scalers():
    with open("scaler_X.pkl","rb") as f: sx=pickle.load(f)
    with open("scaler_y.pkl","rb") as f: sy=pickle.load(f)
    return sx, sy

@st.cache_resource(show_spinner=False)
def load_config():
    with open("model_config.json") as f: return json.load(f)

@st.cache_data(show_spinner=False)
def load_history():
    df=pd.read_csv("Weather.csv",parse_dates=["Timestamp"])
    df.sort_values("Timestamp",inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df


# ════════════════════════════════════════════════════════════
# FEATURES & INFERENCE
# ════════════════════════════════════════════════════════════
FEATS = ["Temperature_C","Humidity_pct","Wind_Speed_ms","Pressure_hPa",
         "Solar_Radiation_Wm2","Dew_Point_C","Precipitation_mm","Cloud_Cover_pct",
         "hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos"]

def make_row(ts,temp,hum,wind,pres,solar,dew,rain,cloud):
    return {"Temperature_C":temp,"Humidity_pct":hum,"Wind_Speed_ms":wind,
            "Pressure_hPa":pres,"Solar_Radiation_Wm2":solar,"Dew_Point_C":dew,
            "Precipitation_mm":rain,"Cloud_Cover_pct":cloud,
            "hour_sin": np.sin(2*np.pi*ts.hour/24),
            "hour_cos": np.cos(2*np.pi*ts.hour/24),
            "month_sin":np.sin(2*np.pi*ts.month/12),
            "month_cos":np.cos(2*np.pi*ts.month/12),
            "dow_sin":  np.sin(2*np.pi*ts.weekday()/7),
            "dow_cos":  np.cos(2*np.pi*ts.weekday()/7)}

def build_seq(df_hist,row,sx,lookback=168):
    h=df_hist.tail(lookback-1).copy()
    for col,src,n in [("hour_sin","Hour",24),("hour_cos","Hour",24),
                      ("month_sin","Month",12),("month_cos","Month",12),
                      ("dow_sin","Day_of_Week",7),("dow_cos","Day_of_Week",7)]:
        fn=np.sin if "sin" in col else np.cos
        h[col]=fn(2*np.pi*h[src]/n)
    rows=h[FEATS].values.tolist()+[[row[f] for f in FEATS]]
    arr=np.array(rows,dtype=np.float32)
    return sx.transform(arr)[np.newaxis,...].astype(np.float32)

def run_predict(seq,sess,sy):
    inp=sess.get_inputs()[0].name
    out=sess.get_outputs()[0].name
    p=sess.run([out],{inp:seq})[0]
    r=sy.inverse_transform(p)
    return float(r[0,0]),float(r[0,1]),float(r[0,2])


# ════════════════════════════════════════════════════════════
# CHARTS  — fixed sizes, professional style
# ════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.facecolor": "white",
    "figure.dpi":       110,
})

def plot_history(df):
    recent = df.tail(7*24)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(recent["Timestamp"], recent["Temperature_C"],
            lw=1.2, color=C_NAVY, alpha=0.9)
    ax.fill_between(recent["Timestamp"], recent["Temperature_C"],
                    alpha=0.12, color=C_BLUE)
    ax.set_title("Recent 7-Day Temperature History",
                 fontsize=10, fontweight="bold", pad=5)
    ax.set_ylabel("°C", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(labelsize=8)
    plt.xticks(rotation=20)
    fig.tight_layout(pad=0.6)
    return fig

def plot_forecast_bar(t1h,t3h,t6h,t0):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    vals  = [t0, t1h, t3h, t6h]
    lbls  = ["Input", "+1 Hour", "+3 Hours", "+6 Hours"]
    colors= [C_BLUE, C_GREEN, C_ORANGE, C_NAVY]
    bars  = ax.bar(lbls, vals, color=colors, width=0.5,
                   edgecolor="white", zorder=3)
    ax.set_ylim(min(vals)-3, max(vals)+5)
    ax.set_title("Forecast Summary", fontsize=10, fontweight="bold", pad=5)
    ax.set_ylabel("°C", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                f"{v:.1f}°C", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    ax.axhline(t0, color="grey", ls="--", lw=1, alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout(pad=0.6)
    return fig

def plot_monthly(df):
    mnames=["Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"]
    data=[df[df["Month"]==m]["Temperature_C"].values for m in range(1,13)]
    fig, ax = plt.subplots(figsize=(7, 3))
    bp=ax.boxplot(data, patch_artist=True,
                  medianprops=dict(color="white",lw=2))
    for p in bp["boxes"]: p.set_facecolor(C_NAVY); p.set_alpha(0.8)
    ax.set_xticklabels(mnames, fontsize=8)
    ax.set_title("Monthly Temperature Distribution",
                 fontsize=10, fontweight="bold", pad=5)
    ax.set_ylabel("°C", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    return fig

def plot_diurnal(df):
    hg=df.groupby("Hour")["Temperature_C"].agg(["mean","std"])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.fill_between(hg.index, hg["mean"]-hg["std"],
                    hg["mean"]+hg["std"], alpha=0.15, color=C_BLUE)
    ax.plot(hg.index, hg["mean"], color=C_NAVY, lw=2,
            marker="o", ms=3)
    ax.set_title("Diurnal Pattern (Mean ± 1σ)",
                 fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel("Hour of Day", fontsize=9)
    ax.set_ylabel("°C", fontsize=9)
    ax.set_xticks(range(0,24,3))
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    return fig


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():

    # ── Banner ───────────────────────────────────────────────
    st.markdown("""
<div class='banner'>
  <h2> CNN-GRU Temperature Forecasting</h2>
  <p>University of Zimbabwe &nbsp;·&nbsp; Department of Analytics and Informatics
     &nbsp;·&nbsp; Horizons: +1h · +3h · +6h
     &nbsp;·&nbsp; Powered by Open-Meteo</p>
</div>""", unsafe_allow_html=True)

    # ── Load artefacts ───────────────────────────────────────
    ok = True
    try:
        sess       = load_model()
        sx, sy     = load_scalers()
        cfg        = load_config()
        df         = load_history()
        LB         = cfg.get("lookback", 168)
    except FileNotFoundError as e:
        st.warning(f" Demo Mode — `{e}`")
        ok = False; LB = 168
        try:    df = load_history()
        except: df = None

    # ── Session state ─────────────────────────────────────────
    for k,v in [("fetched",False),("fd",{}),("status",""),
                ("t1h",None),("t3h",None),("t6h",None),("t0",None)]:
        if k not in st.session_state: st.session_state[k] = v

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("##  Date & Time")
        date = st.date_input(
            "Observation Date",
            value=datetime.date.today()-datetime.timedelta(days=1),
            max_value=datetime.date.today()-datetime.timedelta(days=1))
        hour = st.slider("Observation Hour", 0, 23, 15)

        st.markdown("---")
        st.markdown("###  Auto-Fetch Weather")
        

        loc = st.selectbox("Location", [
            "Harare, Zimbabwe","Bulawayo, Zimbabwe","Custom coordinates"])
        coords = {"Harare, Zimbabwe":(-17.8252,31.0335),
                  "Bulawayo, Zimbabwe":(-20.1325,28.6264)}
        if loc == "Custom coordinates":
            lat = st.number_input("Latitude", -90.0,90.0,-17.8252,0.0001)
            lon = st.number_input("Longitude",-180.0,180.0,31.0335,0.0001)
        else:
            lat, lon = coords[loc]

        if st.button(" Fetch Weather Automatically"):
            with st.spinner(f"Fetching {date} {hour}:00…"):
                data, err = fetch_weather(date, hour, lat, lon)
            if data:
                st.session_state.fd      = data
                st.session_state.fetched = True
                st.session_state.status  = "ok"
            else:
                st.session_state.fetched = False
                st.session_state.status  = err

        if st.session_state.status == "ok":
            st.success(" Weather fetched — fields auto-filled")
        elif st.session_state.status:
            st.error(f" {st.session_state.status}")

        st.markdown("---")
        st.markdown("###  Weather Conditions")
        

        fd   = st.session_state.fd
        temp = st.number_input("Temperature (°C)",  -30.0, 60.0,   float(fd.get("Temperature_C",22.5)), 0.1)
        hum  = st.number_input("Humidity (%)",        0.0,100.0,   float(fd.get("Humidity_pct",65.0)), 0.5)
        dew  = st.number_input("Dew Point (°C)",    -40.0, 50.0,   float(fd.get("Dew_Point_C",15.0)), 0.1)
        wind = st.number_input("Wind Speed (m/s)",    0.0, 80.0,   float(fd.get("Wind_Speed_ms",3.5)), 0.1)
        pres = st.number_input("Pressure (hPa)",    800.0,1100.0,  float(fd.get("Pressure_hPa",1013.0)), 0.5)
        solar= st.number_input("Solar (W/m²)",        0.0,1500.0,  float(fd.get("Solar_Radiation_Wm2",350.0)), 5.0)
        cloud= st.slider("Cloud Cover (%)", 0, 100,  int(fd.get("Cloud_Cover_pct",40)))
        rain = st.number_input("Precipitation (mm)", 0.0, 500.0,   float(fd.get("Precipitation_mm",0.0)), 0.1)

        st.markdown("---")
        go = st.button(" Predict Temperature")

    # ── Run inference ─────────────────────────────────────────
    if go:
        ts  = pd.Timestamp(datetime.datetime.combine(date, datetime.time(hour)))
        row = make_row(ts,temp,hum,wind,pres,solar,dew,rain,cloud)
        with st.spinner("Running CNN-GRU inference…"):
            if ok and df is not None:
                seq = build_seq(df,row,sx,LB)
                t1h,t3h,t6h = run_predict(seq,sess,sy)
            else:
                t1h=temp+0.4; t3h=temp+1.2; t6h=temp+2.5
        st.session_state.t1h=t1h; st.session_state.t3h=t3h
        st.session_state.t6h=t6h; st.session_state.t0=temp

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        [" Forecast", " Data Explorer", " Model Performance"])

    # ════════════════════════════════════════════════════════
    # TAB 1 — FORECAST
    # ════════════════════════════════════════════════════════
    with tab1:
        left, right = st.columns([1, 1.8])

        with left:
            # Model status
            if ok:
                st.markdown(
                    f"<div class='ok'> CNN-GRU model loaded (ONNX) "
                    f"· Lookback: {LB}h (7 days)</div>",
                    unsafe_allow_html=True)
            else:
                st.error(" Demo Mode")

            # How it works
            st.markdown("<div class='sec'> How It Works</div>",
                        unsafe_allow_html=True)
            st.markdown(
                "1. Select a **date** in the sidebar\n"
                "2. Click ** Fetch Weather Automatically**\n"
                "3. Click ** Predict Temperature**\n"
                "4. Get forecasts at **+1h, +3h, +6h"
            )

            # Fetched weather
            if st.session_state.fetched and st.session_state.fd:
                st.markdown("<div class='sec'> Fetched Weather</div>",
                            unsafe_allow_html=True)
                LABELS = {
                    "Temperature_C":"Temperature","Humidity_pct":"Humidity",
                    "Dew_Point_C":"Dew Point","Wind_Speed_ms":"Wind Speed",
                    "Pressure_hPa":"Pressure","Solar_Radiation_Wm2":"Solar Rad.",
                    "Precipitation_mm":"Precipitation","Cloud_Cover_pct":"Cloud Cover"}
                UNITS  = {
                    "Temperature_C":"°C","Humidity_pct":"%","Dew_Point_C":"°C",
                    "Wind_Speed_ms":"m/s","Pressure_hPa":"hPa",
                    "Solar_Radiation_Wm2":"W/m²","Precipitation_mm":"mm",
                    "Cloud_Cover_pct":"%"}
                rows = "".join([
                    f"<div class='fw'>"
                    f"<span class='fw-k'>{LABELS[k]}</span>"
                    f"<span class='fw-v'>{v} {UNITS[k]}</span>"
                    f"</div>"
                    for k,v in st.session_state.fd.items()])
                st.markdown(rows, unsafe_allow_html=True)

        with right:
            # History chart — fixed size, not full width
            st.markdown("<div class='sec'> Recent 7-Day Temperature History</div>",
                        unsafe_allow_html=True)
            if df is not None:
                fig = plot_history(df)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
            else:
                st.info("Weather.csv not loaded.")

        # Forecast results
        if st.session_state.t1h is None:
            msg = (" Weather fetched — click ** Predict Temperature** in sidebar"
                   if st.session_state.fetched
                   else "← Select date in sidebar → Fetch Weather → Predict")
            st.info(msg)
        else:
            st.markdown("<div class='sec'> Forecast Results</div>",
                        unsafe_allow_html=True)
            t0  = st.session_state.t0
            t1h = st.session_state.t1h
            t3h = st.session_state.t3h
            t6h = st.session_state.t6h

            c1, c2, c3 = st.columns(3)
            for col, lbl, val in [
                (c1, "+1 Hour Ahead",  t1h),
                (c2, "+3 Hours Ahead", t3h),
                (c3, "+6 Hours Ahead", t6h)
            ]:
                d     = val - t0
                arrow = "▲" if d >= 0 else "▼"
                dc    = "#1e7e34" if d >= 0 else "#c0392b"
                with col:
                    st.markdown(f"""
<div class='fc'>
  <div class='fc-temp'>{val:.1f}°C</div>
  <div class='fc-label'>{lbl}</div>
  <div class='fc-delta' style='color:{dc};'>
    {arrow} {abs(d):.1f}°C vs input
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_chart, col_info = st.columns([1.5, 1])
            with col_chart:
                fig = plot_forecast_bar(t1h, t3h, t6h, t0)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
            with col_info:
                trend = (" RISING"  if t6h > t0 + 0.5 else
                         " FALLING" if t6h < t0 - 0.5 else
                         "️ STABLE")
                st.markdown(f"""
**Date:** {date} at {hour:02d}:00
**Input:** {t0:.1f}°C
**+1 Hour:** {t1h:.1f}°C
**+3 Hours:** {t3h:.1f}°C
**+6 Hours:** {t6h:.1f}°C
**Trend:** {trend}
""")

    # ════════════════════════════════════════════════════════
    # TAB 2 — DATA EXPLORER
    # ════════════════════════════════════════════════════════
    with tab2:
        if df is None:
            st.warning("Weather.csv not found.")
        else:
            st.markdown("<div class='sec'> Dataset Overview</div>",
                        unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total Records",   f"{len(df):,}")
            m2.metric("Mean Temperature",f"{df['Temperature_C'].mean():.1f}°C")
            m3.metric("Min Temperature", f"{df['Temperature_C'].min():.1f}°C")
            m4.metric("Max Temperature", f"{df['Temperature_C'].max():.1f}°C")

            st.markdown("<div class='sec'> Seasonal & Diurnal Patterns</div>",
                        unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                fig = plot_monthly(df)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with cb:
                fig = plot_diurnal(df)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown("<div class='sec'> Raw Data</div>",
                        unsafe_allow_html=True)
            yr = st.selectbox("Filter by Year",
                ["All"]+sorted(df["Year"].unique().tolist(),reverse=True))
            show = df if yr=="All" else df[df["Year"]==int(yr)]
            st.dataframe(show.tail(150), use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 3 — MODEL PERFORMANCE
    # ════════════════════════════════════════════════════════
    with tab3:
        st.markdown("<div class='sec'> Test-Set Evaluation Metrics</div>",
                    unsafe_allow_html=True)
        res = {
            "1h Ahead": {"RMSE":1.8472,"MAE":1.4380,"R²":0.9611},
            "3h Ahead": {"RMSE":1.9305,"MAE":1.5121,"R²":0.9575},
            "6h Ahead": {"RMSE":2.0366,"MAE":1.5970,"R²":0.9527},
        }
        c1,c2,c3 = st.columns(3)
        for col,(h,m) in zip([c1,c2,c3],res.items()):
            with col:
                st.markdown(f"""
<div class='mc'>
  <div class='mc-h'>{h}</div>
  <div class='mc-v'>R² = {m["R²"]}</div>
  <div class='mc-s'>
    RMSE: {m["RMSE"]}°C &nbsp;|&nbsp; MAE: {m["MAE"]}°C
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec'> Model Architecture</div>",
                    unsafe_allow_html=True)
        st.markdown("""
| Layer | Output Shape | Role |
|:---|:---|:---|
| Input | (None, 168, 14) | 168-hour window × 14 features |
| Conv1D(64, k=3) + BatchNorm | (None, 168, 64) | Local pattern extraction — tier 1 |
| Conv1D(128, k=3) + BatchNorm | (None, 168, 128) | Local pattern extraction — tier 2 |
| MaxPooling1D(2) | (None, 84, 128) | Compress 168 → 84 steps |
| Dropout(0.2) | (None, 84, 128) | Regularisation |
| GRU(128, return_seq=True) | (None, 84, 128) | Sequential memory — layer 1 |
| GRU(64, return_seq=False) | (None, 64) | Sequential context vector |
| Dense(64) → Dense(32) | (None, 32) | Non-linear projection |
| Output Dense(3) | (None, 3) | Forecast: +1h, +3h, +6h |
""")

        st.markdown("<div class='sec'> Training Configuration</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
| Parameter | Value |
|:---|:---|
| Optimiser | Adam |
| Learning Rate | 1 × 10⁻³ |
| Loss Function | MSE |
| Batch Size | 64 |
| Max Epochs | 100 |
""")
        with c2:
            st.markdown("""
| Parameter | Value |
|:---|:---|
| Lookback | 168 hours (7 days) |
| Input Features | 14 |
| Early Stopping | Patience = 15 |
| Train Period | 2021 – 2022 |
| Test Period | Jul – Dec 2023 |
""")

    # ── Footer ────────────────────────────────────────────────
    st.markdown(
        f"<hr style='margin:12px 0;border-color:#e0e0e0;'>"
        f"<div style='text-align:center;color:#888;font-size:0.72rem;'>"
        f"CNN-GRU Temperature Forecasting &nbsp;·&nbsp; "
        f"University of Zimbabwe &nbsp;·&nbsp; "
        f"Department of Analytics and Informatics &nbsp;·&nbsp; 2024 &nbsp;·&nbsp; "
        f"Weather data: "
        f"<a href='https://open-meteo.com' style='color:{C_BLUE};'>Open-Meteo</a>"
        f"</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

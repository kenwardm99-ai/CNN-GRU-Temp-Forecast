"""
CNN-GRU Temperature Forecasting — Ultra Compact Mobile App
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
    page_title="CNN-GRU Forecast",
    page_icon="🌡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

C_NAVY="#1F4E79"; C_BLUE="#2E75B6"; C_GREEN="#70AD47"; C_ORANGE="#ED7D31"

st.markdown(f"""<style>
/* Base */
* {{ box-sizing:border-box; margin:0; padding:0; }}
.block-container {{
    padding: 0.5rem 0.6rem !important;
    max-width: 100% !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{ background:{C_NAVY}; }}
[data-testid="stSidebar"] * {{ color:white!important; font-size:0.78rem!important; }}
[data-testid="stSidebar"] .block-container {{ padding:0.5rem!important; }}

/* Header */
.hdr {{ background:linear-gradient(135deg,{C_NAVY},{C_BLUE});
        padding:8px 12px; border-radius:8px; margin-bottom:8px; }}
.hdr h2 {{ color:white; font-size:0.95rem; margin:0; font-weight:700; }}
.hdr p  {{ color:#D6E4F0; font-size:0.68rem; margin:2px 0 0; }}

/* Section labels */
.sl {{ background:{C_NAVY}; color:white!important; padding:3px 8px;
       border-radius:5px; font-size:0.72rem; font-weight:700;
       margin:6px 0 4px; display:inline-block; }}

/* Forecast cards */
.fc {{ border:1.5px solid {C_BLUE}; border-radius:8px;
       padding:6px 4px; text-align:center; background:#f7faff; }}
.fc-t {{ font-size:1.1rem; font-weight:700; color:{C_NAVY}; line-height:1.2; }}
.fc-l {{ font-size:0.62rem; color:#555; }}
.fc-d {{ font-size:0.65rem; font-weight:700; margin-top:2px; }}

/* Metric cards */
.mc {{ border:1.5px solid {C_BLUE}; border-radius:8px;
       padding:6px 4px; text-align:center; background:#f7faff; }}
.mc-h {{ font-size:0.72rem; font-weight:700; color:{C_NAVY}; }}
.mc-r {{ font-size:1rem; font-weight:700; color:{C_NAVY}; }}
.mc-s {{ font-size:0.6rem; color:#555; }}

/* Status */
.ok-badge {{ background:#d4edda; border:1.5px solid {C_GREEN};
             border-radius:6px; padding:4px 8px; color:#155724;
             font-size:0.72rem; font-weight:600; margin:4px 0; }}

/* Fetch rows */
.fr {{ display:flex; justify-content:space-between;
       padding:2px 0; border-bottom:1px solid #eee;
       font-size:0.68rem; }}
.fk {{ color:#888; }}
.fv {{ font-weight:600; color:{C_NAVY}; }}

/* Predict button */
div[data-testid="stButton"] > button {{
    background:linear-gradient(135deg,{C_NAVY},{C_BLUE});
    color:white!important; border:none; border-radius:8px;
    padding:8px 16px; font-size:0.82rem;
    font-weight:700; width:100%; margin-top:4px;
}}

/* Tabs */
[data-testid="stTabs"] button {{
    font-size:0.72rem!important;
    padding:4px 8px!important;
}}

/* Streamlit default element size reductions */
[data-testid="stMetric"] {{ padding:4px!important; }}
[data-testid="stMetricLabel"] {{ font-size:0.68rem!important; }}
[data-testid="stMetricValue"] {{ font-size:0.9rem!important; }}
h3 {{ font-size:0.85rem!important; margin:6px 0 4px!important; }}
p, li, td, th {{ font-size:0.78rem!important; }}
table {{ font-size:0.72rem!important; }}
[data-testid="stAlert"] {{ font-size:0.72rem!important; padding:6px 10px!important; }}
[data-testid="stInfo"] {{ font-size:0.72rem!important; }}
</style>""", unsafe_allow_html=True)


# ── Fetcher ───────────────────────────────────────────────────
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


# ── Loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading…")
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


# ── Features ─────────────────────────────────────────────────
FEATS=["Temperature_C","Humidity_pct","Wind_Speed_ms","Pressure_hPa",
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


# ── Tiny plots ────────────────────────────────────────────────
def plot_history(df):
    r=df.tail(7*24)
    fig,ax=plt.subplots(figsize=(6,2))
    ax.plot(r["Timestamp"],r["Temperature_C"],lw=1,color=C_NAVY)
    ax.fill_between(r["Timestamp"],r["Temperature_C"],alpha=0.1,color=C_BLUE)
    ax.set_title("7-Day History",fontsize=8,fontweight="bold",pad=3)
    ax.set_ylabel("°C",fontsize=7); ax.tick_params(labelsize=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15); fig.tight_layout(pad=0.3); return fig

def plot_bar(t1h,t3h,t6h,t0):
    fig,ax=plt.subplots(figsize=(5,2.2))
    vals=[t0,t1h,t3h,t6h]; lbls=["Now","+1h","+3h","+6h"]
    bars=ax.bar(lbls,vals,color=[C_BLUE,C_GREEN,C_ORANGE,C_NAVY],
                width=0.5,edgecolor="white",zorder=3)
    ax.set_ylim(min(vals)-2,max(vals)+3)
    ax.set_title("Forecast",fontsize=8,fontweight="bold",pad=3)
    ax.set_ylabel("°C",fontsize=7); ax.tick_params(labelsize=7)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.1,
                f"{v:.1f}",ha="center",va="bottom",fontsize=7,fontweight="bold")
    ax.axhline(t0,color="grey",ls="--",lw=0.8,alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y",alpha=0.3,zorder=0)
    fig.tight_layout(pad=0.3); return fig

def plot_monthly(df):
    mnames=["J","F","M","A","M","J","J","A","S","O","N","D"]
    data=[df[df["Month"]==m]["Temperature_C"].values for m in range(1,13)]
    fig,ax=plt.subplots(figsize=(6,2.5))
    bp=ax.boxplot(data,patch_artist=True,medianprops=dict(color="white",lw=1.5))
    for p in bp["boxes"]: p.set_facecolor(C_NAVY); p.set_alpha(0.8)
    ax.set_xticklabels(mnames,fontsize=7)
    ax.set_title("Monthly Temperature",fontsize=8,fontweight="bold",pad=3)
    ax.set_ylabel("°C",fontsize=7); ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.3); return fig

def plot_diurnal(df):
    hg=df.groupby("Hour")["Temperature_C"].agg(["mean","std"])
    fig,ax=plt.subplots(figsize=(6,2.2))
    ax.fill_between(hg.index,hg["mean"]-hg["std"],hg["mean"]+hg["std"],
                    alpha=0.15,color=C_BLUE)
    ax.plot(hg.index,hg["mean"],color=C_NAVY,lw=1.5,marker="o",ms=2)
    ax.set_title("Diurnal Pattern",fontsize=8,fontweight="bold",pad=3)
    ax.set_xlabel("Hour",fontsize=7); ax.set_ylabel("°C",fontsize=7)
    ax.set_xticks(range(0,24,6)); ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.3); return fig


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():

    # Header
    st.markdown("""
<div class='hdr'>
  <h2>🌡️ CNN-GRU Temperature Forecast</h2>
  <p>University of Zimbabwe &nbsp;·&nbsp; +1h · +3h · +6h &nbsp;·&nbsp; Open-Meteo</p>
</div>""", unsafe_allow_html=True)

    # Load artefacts
    ok=True
    try:
        sess=load_model(); sx,sy=load_scalers()
        cfg=load_config(); df=load_history()
        LB=cfg.get("lookback",168)
    except FileNotFoundError as e:
        st.warning(f"Demo Mode — {e}"); ok=False; LB=168
        try:    df=load_history()
        except: df=None

    # Session state
    for k,v in [("fetched",False),("fd",{}),("status",""),
                ("t1h",None),("t3h",None),("t6h",None),("t0",None)]:
        if k not in st.session_state: st.session_state[k]=v

    # Sidebar
    with st.sidebar:
        st.markdown("**📅 Date & Hour**")
        date=st.date_input("Date",
            value=datetime.date.today()-datetime.timedelta(days=1),
            max_value=datetime.date.today()-datetime.timedelta(days=1),
            label_visibility="collapsed")
        hour=st.slider("Hour",0,23,15)
        loc=st.selectbox("Location",
            ["Harare, Zimbabwe","Bulawayo, Zimbabwe","Custom"],
            label_visibility="collapsed")
        coords={"Harare, Zimbabwe":(-17.8252,31.0335),
                "Bulawayo, Zimbabwe":(-20.1325,28.6264)}
        if loc=="Custom":
            lat=st.number_input("Lat",-90.0,90.0,-17.8252,0.0001)
            lon=st.number_input("Lon",-180.0,180.0,31.0335,0.0001)
        else: lat,lon=coords[loc]

        if st.button("🌐 Fetch Weather"):
            with st.spinner("Fetching…"):
                data,err=fetch_weather(date,hour,lat,lon)
            if data:
                st.session_state.fd=data
                st.session_state.fetched=True
                st.session_state.status="ok"
            else:
                st.session_state.fetched=False
                st.session_state.status=err

        if st.session_state.status=="ok":
            st.success("✅ Fetched!")
        elif st.session_state.status:
            st.error(f"❌ {st.session_state.status}")

        st.markdown("**🌡️ Conditions**")
        fd=st.session_state.fd
        temp =st.number_input("Temp °C",   -30.0, 60.0,  float(fd.get("Temperature_C",22.5)),0.1)
        hum  =st.number_input("Humidity %",  0.0,100.0,  float(fd.get("Humidity_pct",65.0)),0.5)
        dew  =st.number_input("Dew Pt °C", -40.0, 50.0,  float(fd.get("Dew_Point_C",15.0)),0.1)
        wind =st.number_input("Wind m/s",    0.0, 80.0,   float(fd.get("Wind_Speed_ms",3.5)),0.1)
        pres =st.number_input("Pres hPa",  800.0,1100.0,  float(fd.get("Pressure_hPa",1013.0)),0.5)
        solar=st.number_input("Solar W/m²", 0.0,1500.0,   float(fd.get("Solar_Radiation_Wm2",350.0)),5.0)
        cloud=st.slider("Cloud %",0,100,    int(fd.get("Cloud_Cover_pct",40)))
        rain =st.number_input("Rain mm",    0.0, 500.0,   float(fd.get("Precipitation_mm",0.0)),0.1)

        go=st.button("🔮 Predict")

    # Run prediction
    if go:
        ts=pd.Timestamp(datetime.datetime.combine(date,datetime.time(hour)))
        row=make_row(ts,temp,hum,wind,pres,solar,dew,rain,cloud)
        with st.spinner("Running…"):
            if ok and df is not None:
                seq=build_seq(df,row,sx,LB)
                t1h,t3h,t6h=run_predict(seq,sess,sy)
            else:
                t1h=temp+0.4; t3h=temp+1.2; t6h=temp+2.5
        st.session_state.t1h=t1h; st.session_state.t3h=t3h
        st.session_state.t6h=t6h; st.session_state.t0=temp

    # Tabs
    tab1,tab2,tab3=st.tabs(["🔮 Forecast","📊 Data","📈 Model"])

    # ── TAB 1 ────────────────────────────────────────────────
    with tab1:
        # Status row
        if ok:
            st.markdown(
                f"<div class='ok-badge'>✅ CNN-GRU (ONNX) · Lookback {LB}h</div>",
                unsafe_allow_html=True)
        else:
            st.error("⚠️ Demo Mode")

        # Fetched weather — compact 2-col grid
        if st.session_state.fetched and st.session_state.fd:
            st.markdown("<div class='sl'>🌐 Fetched Weather</div>",
                        unsafe_allow_html=True)
            fd2=st.session_state.fd
            LABELS={"Temperature_C":"Temp","Humidity_pct":"Humidity",
                    "Dew_Point_C":"Dew Pt","Wind_Speed_ms":"Wind",
                    "Pressure_hPa":"Pressure","Solar_Radiation_Wm2":"Solar",
                    "Precipitation_mm":"Rain","Cloud_Cover_pct":"Cloud"}
            UNITS ={"Temperature_C":"°C","Humidity_pct":"%","Dew_Point_C":"°C",
                    "Wind_Speed_ms":"m/s","Pressure_hPa":"hPa",
                    "Solar_Radiation_Wm2":"W/m²","Precipitation_mm":"mm",
                    "Cloud_Cover_pct":"%"}
            items=list(fd2.items())
            col1,col2=st.columns(2)
            for i,(k,v) in enumerate(items):
                with (col1 if i%2==0 else col2):
                    st.markdown(
                        f"<div class='fr'>"
                        f"<span class='fk'>{LABELS[k]}</span>"
                        f"<span class='fv'>{v}{UNITS[k]}</span>"
                        f"</div>",unsafe_allow_html=True)

        # Guide
        if st.session_state.t1h is None:
            msg=("✅ Fetched — click 🔮 Predict in sidebar"
                 if st.session_state.fetched
                 else "👈 Open sidebar → Fetch → Predict")
            st.info(msg)
        else:
            # Cards
            st.markdown("<div class='sl'>🔮 Results</div>",
                        unsafe_allow_html=True)
            t0=st.session_state.t0
            c1,c2,c3=st.columns(3)
            for col,lbl,val in [(c1,"+1h",st.session_state.t1h),
                                (c2,"+3h",st.session_state.t3h),
                                (c3,"+6h",st.session_state.t6h)]:
                d=val-t0; arrow="▲" if d>=0 else "▼"
                dc="#27ae60" if d>=0 else "#e74c3c"
                with col:
                    st.markdown(f"""
<div class='fc'>
  <div class='fc-t'>{val:.1f}°C</div>
  <div class='fc-l'>{lbl}</div>
  <div class='fc-d' style='color:{dc};'>{arrow}{abs(d):.1f}°</div>
</div>""",unsafe_allow_html=True)

            # Bar chart
            st.pyplot(plot_bar(
                st.session_state.t1h,st.session_state.t3h,
                st.session_state.t6h,t0),
                use_container_width=True)

            # Trend
            t6h=st.session_state.t6h
            trend=("📈 RISING" if t6h>t0+0.5 else
                   "📉 FALLING" if t6h<t0-0.5 else "➡️ STABLE")
            st.info(
                f"{t0:.1f}°→{st.session_state.t1h:.1f}°(+1h)"
                f"→{st.session_state.t3h:.1f}°(+3h)"
                f"→{t6h:.1f}°(+6h) {trend}")

        # History
        if df is not None:
            st.markdown("<div class='sl'>📉 7-Day History</div>",
                        unsafe_allow_html=True)
            st.pyplot(plot_history(df),use_container_width=True)

    # ── TAB 2 ────────────────────────────────────────────────
    with tab2:
        if df is None:
            st.warning("Weather.csv not found.")
        else:
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Rows",   f"{len(df):,}")
            m2.metric("Mean",   f"{df['Temperature_C'].mean():.1f}°")
            m3.metric("Min",    f"{df['Temperature_C'].min():.1f}°")
            m4.metric("Max",    f"{df['Temperature_C'].max():.1f}°")
            st.pyplot(plot_monthly(df), use_container_width=True)
            st.pyplot(plot_diurnal(df), use_container_width=True)
            yr=st.selectbox("Year",
               ["All"]+sorted(df["Year"].unique().tolist(),reverse=True))
            show=df if yr=="All" else df[df["Year"]==int(yr)]
            st.dataframe(show.tail(80),use_container_width=True)

    # ── TAB 3 ────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='sl'>📈 Test-Set Results</div>",
                    unsafe_allow_html=True)
        res={"1h":{"RMSE":1.847,"MAE":1.438,"R²":0.9611},
             "3h":{"RMSE":1.931,"MAE":1.512,"R²":0.9575},
             "6h":{"RMSE":2.037,"MAE":1.597,"R²":0.9527}}
        c1,c2,c3=st.columns(3)
        for col,(h,m) in zip([c1,c2,c3],res.items()):
            with col:
                st.markdown(f"""
<div class='mc'>
  <div class='mc-h'>+{h}</div>
  <div class='mc-r'>R²={m["R²"]}</div>
  <div class='mc-s'>RMSE {m["RMSE"]}°<br>MAE {m["MAE"]}°</div>
</div>""",unsafe_allow_html=True)

        st.markdown("<div class='sl'>🏗️ Architecture</div>",
                    unsafe_allow_html=True)
        st.markdown("""
| Layer | Shape | Role |
|---|---|---|
| Input | (168,14) | 168h × 14 features |
| Conv1D×2 + BN | (168,128) | Local patterns |
| MaxPool | (84,128) | 168→84 steps |
| GRU(128)+GRU(64) | (64,) | Sequential memory |
| Dense(64→32→3) | (3,) | +1h,+3h,+6h |
""")
        st.markdown("<div class='sl'>⚙️ Training</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "Adam · LR=1e-3 · Batch=64 · "
            "Lookback=168h · Features=14 · "
            "EarlyStopping(patience=15)")

    # Footer
    st.markdown(
        f"<div style='text-align:center;color:#aaa;font-size:0.62rem;"
        f"padding:6px 0;margin-top:4px;border-top:1px solid #eee;'>"
        f"CNN-GRU · University of Zimbabwe · 2024 · "
        f"<a href='https://open-meteo.com' style='color:{C_BLUE};'>"
        f"Open-Meteo</a></div>",
        unsafe_allow_html=True)

if __name__=="__main__":
    main()

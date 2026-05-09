ensures the model sees real recent temperatures
            # not old 2023 training data from Weather.csv
            real_seq_df, seq_err = fetch_sequence_from_meteo(
                date, hour, lat, lon, LB)

        with st.spinner("Running CNN-GRU inference…"):
            if ok and real_seq_df is not None:
                # Use real fetched sequence
                seq = build_seq(real_seq_df, row, sx, LB)
                t1h,t3h,t6h = run_predict(seq,sess,sy)
            elif ok and df is not None:
                # Fallback to Weather.csv if fetch failed
                if seq_err:
                    st.warning(f"Using historical data (real fetch failed: {seq_err})")
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

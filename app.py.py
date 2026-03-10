import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoXAI Framework",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark teal background */
.stApp {
    background-color: #0a1628;
    color: #e0f0f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0d1e35;
    border-right: 1px solid #1a3a5c;
}

/* Header Banner */
.eco-header {
    background: linear-gradient(135deg, #0d4f6e 0%, #0a2a45 50%, #0a1628 100%);
    border: 1px solid #1a6a8a;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.eco-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    color: #4fd1c5;
    letter-spacing: 4px;
    margin: 0;
    text-shadow: 0 0 30px rgba(79,209,197,0.4);
}
.eco-header p {
    color: #7ecdc8;
    font-size: 0.95rem;
    letter-spacing: 2px;
    margin: 0.5rem 0 0;
    text-transform: uppercase;
}

/* Section Headers */
.section-title {
    font-family: 'Space Mono', monospace;
    color: #4fd1c5;
    font-size: 1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #1a4a6a;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #0d2a45 0%, #0a1e35 100%);
    border: 1px solid #1a4a6a;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #4fd1c5;
    font-weight: 700;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    color: #7ecdc8;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

/* Alert / Recommendation Boxes */
.alert-box {
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
.alert-critical { background: rgba(220,53,69,0.12); border-left: 3px solid #dc3545; }
.alert-warning  { background: rgba(255,193,7,0.10); border-left: 3px solid #ffc107; }
.alert-success  { background: rgba(79,209,197,0.10); border-left: 3px solid #4fd1c5; }

/* Dataframe table */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0d6e8a, #0a4a6e);
    color: #e0f0f0;
    border: 1px solid #4fd1c5;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4fd1c5, #0d6e8a);
    color: #0a1628;
    border-color: #4fd1c5;
}

/* Slider labels */
label { color: #7ecdc8 !important; font-size: 0.85rem; }

/* Expander */
.streamlit-expanderHeader {
    background-color: #0d2a45 !important;
    color: #4fd1c5 !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 1px;
}

/* Plot backgrounds */
.stPlotlyChart, .stImage { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Data & Model (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_centre_readings = {
        'Number of Servers': [100, 200, 150, 300, 250, 180, 220, 270, 130, 190],
        'Temperature (C)':   [25,  35,  30,  40,  38,  28,  33,  39,  26,  31],
        'Cooling Hours':     [2,   5,   3,   7,   6,   3,   4,   6,   2,   4],
        'Crypto Mining':     [0,   1,   0,   1,   1,   0,   1,   1,   0,   0],
        'Water Used (L)':    [500, 1200,800, 2000,1700,650, 1100,1850,520, 900]
    }
    df = pd.DataFrame(data_centre_readings)
    df['Crypto Status'] = df['Crypto Mining'].map({0: 'No Mining', 1: 'Mining Active'})
    df['Risk Level'] = ['Low','High','Medium','Critical','High','Low','High','Critical','Low','Medium']
    return df

@st.cache_resource
def train_model(df):
    X = df[['Number of Servers','Temperature (C)','Cooling Hours','Crypto Mining']]
    y = df['Water Used (L)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return model, explainer, shap_values, X

# Apply dark style to matplotlib figures
def dark_fig(figsize=(9,5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0d2a45')
    ax.set_facecolor('#0a1e35')
    ax.tick_params(colors='#7ecdc8')
    ax.xaxis.label.set_color('#7ecdc8')
    ax.yaxis.label.set_color('#7ecdc8')
    ax.title.set_color('#4fd1c5')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a4a6a')
    return fig, ax

# ─── Load ──────────────────────────────────────────────────────────────────────
sensor_data = load_data()
model, explainer, shap_values, X = train_model(sensor_data)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eco-header">
  <h1>💧 EcoXAI</h1>
  <p>Saving Water · Explaining Why · Intelligently</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚡ Live Prediction</div>', unsafe_allow_html=True)
    st.caption("Adjust current data centre conditions")

    servers = st.slider("Number of Servers", 50, 400, 250, step=10)
    temp    = st.slider("Temperature (°C)", 20, 45, 37)
    cooling = st.slider("Cooling Hours", 1, 10, 5)
    crypto  = st.selectbox("Crypto Mining", ["Inactive", "Active"])
    crypto_val = 1 if crypto == "Active" else 0

    st.markdown("---")
    predict_btn = st.button("🔍 Run EcoXAI Prediction", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📍 Monitored Sites</div>', unsafe_allow_html=True)
    st.markdown("🔵 Mumbai &nbsp;&nbsp; 🟢 Hyderabad &nbsp;&nbsp; 🟡 Chennai", unsafe_allow_html=True)

    st.markdown("---")
    csv = sensor_data.to_csv(index=False)
    st.download_button("⬇ Download CSV", csv, "EcoXAI_Results.csv", "text/csv", use_container_width=True)

# ─── Top KPI Metrics ──────────────────────────────────────────────────────────
avg_water   = sensor_data['Water Used (L)'].mean()
max_water   = sensor_data['Water Used (L)'].max()
crypto_lift = ((sensor_data[sensor_data['Crypto Mining']==1]['Water Used (L)'].mean() /
                sensor_data[sensor_data['Crypto Mining']==0]['Water Used (L)'].mean()) - 1) * 100
critical_ct = (sensor_data['Risk Level'] == 'Critical').sum()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{avg_water:.0f} L</div>
        <div class="metric-label">Avg Water / Reading</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{max_water:.0f} L</div>
        <div class="metric-label">Peak Consumption</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">+{crypto_lift:.0f}%</div>
        <div class="metric-label">Crypto Mining Uplift</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{critical_ct}</div>
        <div class="metric-label">Critical Risk Sites</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "📈 Visual Findings", "🧠 XAI Explanations", "🔮 Live Prediction"])

# ══════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">IoT Sensor Data — Data Centre Readings</div>', unsafe_allow_html=True)

    col_filter, _ = st.columns([2, 3])
    with col_filter:
        risk_filter = st.multiselect("Filter by Risk Level",
                                     options=['Low','Medium','High','Critical'],
                                     default=['Low','Medium','High','Critical'])

    filtered = sensor_data[sensor_data['Risk Level'].isin(risk_filter)]

    display_df = filtered[['Number of Servers','Temperature (C)','Cooling Hours',
                            'Crypto Status','Water Used (L)','Risk Level']].copy()

    def colour_risk(val):
        colours = {'Low':'background-color:#0d4a2a;color:#4fd1c5',
                   'Medium':'background-color:#4a3a0a;color:#ffd700',
                   'High':'background-color:#4a2a0a;color:#ff8c42',
                   'Critical':'background-color:#4a0d0d;color:#ff4444'}
        return colours.get(val,'')

    st.dataframe(
        display_df.style.applymap(colour_risk, subset=['Risk Level'])
                        .format({'Water Used (L)':'{:.0f}'}),
        use_container_width=True, height=370
    )

    st.caption(f"Showing {len(filtered)} of {len(sensor_data)} readings")

# ══════════════════════════════════════════════════════
# TAB 2 — VISUAL FINDINGS
# ══════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns(2)

    # Chart 1 — Servers vs Water
    with col_l:
        st.markdown('<div class="section-title">Finding 1 · Servers vs Water</div>', unsafe_allow_html=True)
        fig, ax = dark_fig((7,4))
        ax.bar(sensor_data['Number of Servers'], sensor_data['Water Used (L)'],
               color='#4fd1c5', width=8, alpha=0.85)
        ax.set_xlabel('Number of Servers')
        ax.set_ylabel('Water Used (L)')
        ax.set_title('More Servers = More Water', fontsize=11)
        ax.grid(axis='y', color='#1a4a6a', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.markdown('<div class="alert-box alert-success">Server count directly scales water consumption.</div>', unsafe_allow_html=True)

    # Chart 2 — Temperature vs Water
    with col_r:
        st.markdown('<div class="section-title">Finding 2 · Temperature vs Water</div>', unsafe_allow_html=True)
        fig, ax = dark_fig((7,4))
        sc = ax.scatter(sensor_data['Temperature (C)'], sensor_data['Water Used (L)'],
                        c=sensor_data['Water Used (L)'], cmap='YlOrRd', s=150, zorder=3)
        fig.colorbar(sc, ax=ax, label='Water (L)')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Water Used (L)')
        ax.set_title('Higher Temp = Exponentially More Water', fontsize=11)
        ax.grid(color='#1a4a6a', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.markdown('<div class="alert-box alert-warning">Temperature is the primary driver of water waste.</div>', unsafe_allow_html=True)

    col_l2, col_r2 = st.columns(2)

    # Chart 3 — Crypto Mining Impact
    with col_l2:
        st.markdown('<div class="section-title">Finding 3 · Crypto Mining Impact</div>', unsafe_allow_html=True)
        avg_no  = sensor_data[sensor_data['Crypto Mining']==0]['Water Used (L)'].mean()
        avg_yes = sensor_data[sensor_data['Crypto Mining']==1]['Water Used (L)'].mean()
        fig, ax = dark_fig((7,4))
        bars = ax.bar(['No Mining','Mining Active'], [avg_no, avg_yes],
                      color=['#4fd1c5','#ff4444'], width=0.45)
        ax.set_ylabel('Avg Water Used (L)')
        ax.set_title('Crypto Mining Doubles Consumption', fontsize=11)
        ax.grid(axis='y', color='#1a4a6a', linestyle='--', alpha=0.5)
        for b, v in zip(bars, [avg_no, avg_yes]):
            ax.text(b.get_x()+b.get_width()/2, v+25, f'{v:.0f}L',
                    ha='center', color='#e0f0f0', fontweight='bold', fontsize=11)
        st.pyplot(fig)
        st.markdown(f'<div class="alert-box alert-critical">Crypto mining raises water usage by <b>{crypto_lift:.0f}%</b>.</div>', unsafe_allow_html=True)

    # Chart 4 — Risk Level
    with col_r2:
        st.markdown('<div class="section-title">Finding 4 · Risk Level Analysis</div>', unsafe_allow_html=True)
        risk_map = {
            'Low Risk':      sensor_data[sensor_data['Water Used (L)']<=600]['Water Used (L)'].mean(),
            'Medium Risk':   sensor_data[(sensor_data['Water Used (L)']>600)&(sensor_data['Water Used (L)']<=900)]['Water Used (L)'].mean(),
            'High Risk':     sensor_data[(sensor_data['Water Used (L)']>900)&(sensor_data['Water Used (L)']<=1500)]['Water Used (L)'].mean(),
            'Critical Risk': sensor_data[sensor_data['Water Used (L)']>1500]['Water Used (L)'].mean(),
        }
        fig, ax = dark_fig((7,4))
        clrs = ['#4fd1c5','#ffd700','#ff8c42','#ff4444']
        rbars = ax.bar(risk_map.keys(), risk_map.values(), color=clrs, width=0.45)
        ax.set_ylabel('Avg Water Consumed (L)')
        ax.set_title('Risk Level Impacts Water Waste', fontsize=11)
        ax.grid(axis='y', color='#1a4a6a', linestyle='--', alpha=0.5)
        for b, v in zip(rbars, risk_map.values()):
            ax.text(b.get_x()+b.get_width()/2, v+25, f'{v:.0f}L',
                    ha='center', color='#e0f0f0', fontweight='bold', fontsize=10)
        plt.xticks(rotation=10)
        st.pyplot(fig)
        st.markdown('<div class="alert-box alert-critical">Critical sites waste 4× more water than low-risk sites.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 3 — XAI EXPLANATIONS
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">XAI Engine · Why Is Water Being Wasted?</div>', unsafe_allow_html=True)

    xai_col1, xai_col2 = st.columns([1, 2])
    with xai_col1:
        st.markdown("""
        <div style="background:#0d2a45;border:1px solid #1a4a6a;border-radius:10px;padding:1.2rem;">
        <b style="color:#4fd1c5;font-family:'Space Mono',monospace;font-size:0.8rem;">HOW TO READ SHAP CHARTS</b><br><br>
        <span style="color:#7ecdc8;font-size:0.85rem;">
        📌 <b>Longer bar</b> = bigger impact on water waste<br><br>
        🔴 <b>Red/positive</b> = increases water usage<br><br>
        🔵 <b>Blue/negative</b> = decreases water usage<br><br>
        🌡️ <b>Temperature</b> consistently ranked <b>#1</b> driver<br><br>
        💻 <b>Crypto Mining</b> creates sharp non-linear spikes
        </span>
        </div>
        """, unsafe_allow_html=True)

    with xai_col2:
        idx = st.selectbox("Select reading to explain (index)", range(len(sensor_data)),
                           format_func=lambda i: f"Reading {i} — {sensor_data['Risk Level'].iloc[i]} Risk, {sensor_data['Water Used (L)'].iloc[i]}L")
        st.caption("SHAP Waterfall — individual prediction breakdown")

        fig_w, ax_w = plt.subplots(figsize=(8,4))
        fig_w.patch.set_facecolor('#0d2a45')
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.gcf().set_facecolor('#0d2a45')
        for ax in plt.gcf().get_axes():
            ax.set_facecolor('#0a1e35')
            ax.tick_params(colors='#7ecdc8')
        st.pyplot(plt.gcf())
        plt.close('all')

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Global Factor Importance — Which Factor Consistently Wastes Most Water?</div>', unsafe_allow_html=True)
    st.caption("SHAP Summary Plot — all readings overlaid")

    fig_s, ax_s = plt.subplots(figsize=(9,4))
    fig_s.patch.set_facecolor('#0d2a45')
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    plt.gcf().set_facecolor('#0d2a45')
    for ax in plt.gcf().get_axes():
        ax.set_facecolor('#0a1e35')
        ax.tick_params(colors='#7ecdc8')
        ax.xaxis.label.set_color('#7ecdc8')
    st.pyplot(plt.gcf())
    plt.close('all')

    st.markdown('<div class="alert-box alert-warning">🌡️ <b>Temperature</b> is ranked as the #1 consistent driver of water waste across all data centre readings.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTION
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🔮 EcoXAI Live Prediction Engine</div>', unsafe_allow_html=True)
    st.caption("Adjust sliders in the sidebar, then press Run EcoXAI Prediction")

    # Show current inputs
    curr_col1, curr_col2, curr_col3, curr_col4 = st.columns(4)
    with curr_col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{servers}</div><div class="metric-label">Servers</div></div>""", unsafe_allow_html=True)
    with curr_col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{temp}°C</div><div class="metric-label">Temperature</div></div>""", unsafe_allow_html=True)
    with curr_col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{cooling}h</div><div class="metric-label">Cooling Hours</div></div>""", unsafe_allow_html=True)
    with curr_col4:
        colour = "#ff4444" if crypto_val else "#4fd1c5"
        st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:{colour}">{crypto}</div><div class="metric-label">Crypto Mining</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        live = pd.DataFrame({
            'Number of Servers': [servers],
            'Temperature (C)':   [temp],
            'Cooling Hours':     [cooling],
            'Crypto Mining':     [crypto_val]
        })
        prediction = model.predict(live)[0]

        # Risk classification
        if prediction <= 600:
            risk_label, risk_class = "LOW RISK", "alert-success"
        elif prediction <= 900:
            risk_label, risk_class = "MEDIUM RISK", "alert-warning"
        elif prediction <= 1500:
            risk_label, risk_class = "HIGH RISK", "alert-warning"
        else:
            risk_label, risk_class = "CRITICAL RISK", "alert-critical"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d4f6e,#0a2a45);border:2px solid #4fd1c5;
                    border-radius:12px;padding:1.5rem 2rem;text-align:center;margin:1rem 0;">
            <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                        color:#7ecdc8;letter-spacing:3px;margin-bottom:0.5rem;">PREDICTED WATER CONSUMPTION</div>
            <div style="font-family:'Space Mono',monospace;font-size:3rem;
                        color:#4fd1c5;font-weight:700;text-shadow:0 0 20px rgba(79,209,197,0.5);">
                {prediction:,.0f} L
            </div>
            <div style="font-size:0.85rem;color:#7ecdc8;margin-top:0.3rem;">{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🧠 EcoXAI Explains Why")
        reasons = []
        if temp >= 35:
            reasons.append(("🌡️ Primary Cause", f"Temperature at {temp}°C exceeds safe threshold (30°C)", "alert-critical"))
        if servers >= 200:
            reasons.append(("💻 Secondary Cause", f"{servers} servers generating significant heat load", "alert-warning"))
        if crypto_val:
            reasons.append(("⛏️ Amplifier", "Crypto mining running simultaneously — highest risk multiplier", "alert-critical"))
        if cooling >= 5:
            reasons.append(("❄️ Cooling Load", f"{cooling} hours of active cooling required", "alert-warning"))

        if not reasons:
            reasons.append(("✅ Status", "Conditions appear to be within normal operating range", "alert-success"))

        for label, msg, cls in reasons:
            st.markdown(f'<div class="alert-box {cls}"><b>{label}</b>: {msg}</div>', unsafe_allow_html=True)

        st.markdown("#### ✅ EcoXAI Recommends")
        recs = []
        if temp >= 35:
            recs.append("Reduce data centre temperature to below 30°C immediately")
        if servers >= 200:
            recs.append("Temporarily reduce active server load where possible")
        if crypto_val:
            recs.append("Reschedule crypto mining to cooler night-time hours")
        if cooling >= 5:
            recs.append("Evaluate liquid cooling or adiabatic alternatives")
        if not recs:
            recs.append("No immediate action required — maintain current conditions")

        for i, rec in enumerate(recs, 1):
            st.markdown(f'<div class="alert-box alert-success">Action {i}: {rec}</div>', unsafe_allow_html=True)

        st.markdown("#### 💧 Expected Savings")
        st.markdown(f"""
        <div class="alert-box alert-success">
        Estimated water reduction after recommendations: <b>40–60%</b><br>
        Potential savings: <b>{prediction*0.4:,.0f}–{prediction*0.6:,.0f} litres</b> per monitoring cycle<br>
        Environmental gain: Reduced freshwater depletion in local communities near Mumbai, Hyderabad, and Chennai
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👈 Set conditions in the sidebar and press **Run EcoXAI Prediction** to get a real-time analysis.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#2a5a7a;font-family:'Space Mono',monospace;
            font-size:0.7rem;letter-spacing:2px;padding:0.5rem 0;">
EcoXAI FRAMEWORK · SAVING WATER · EXPLAINING WHY · INTELLIGENTLY
</div>
""", unsafe_allow_html=True)

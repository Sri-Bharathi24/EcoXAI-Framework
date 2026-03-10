import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="EcoXAI Framework",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Nunito', sans-serif !important; }
code, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* ── APP BACKGROUND ── */
.stApp { background-color: #0b1520; color: #f0f4f8; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: #111f30;
    border-right: 2px solid #00d4aa;
}
section[data-testid="stSidebar"] * { color: #f0f4f8 !important; font-weight: 600 !important; }

/* ── ALL LABELS & TEXT ── */
label, .stSlider label, p, span {
    color: #f0f4f8 !important;
    font-size: 0.97rem !important;
    font-weight: 700 !important;
}
h1,h2,h3,h4,h5 { color: #ffffff !important; font-weight: 800 !important; }
.stCaption, caption { color: #90b8c8 !important; font-size: 0.88rem !important; }

/* ── SELECTBOX — ACTIVE/INACTIVE FIX ── */
div[data-baseweb="select"] {
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div {
    background-color: #1a2f45 !important;
    border: 2px solid #00d4aa !important;
    border-radius: 10px !important;
    min-height: 48px !important;
}
div[data-baseweb="select"] * {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    background-color: transparent !important;
}
div[data-baseweb="select"] svg { fill: #00d4aa !important; }

/* Dropdown list */
div[data-baseweb="popover"] {
    background-color: #1a2f45 !important;
    border: 2px solid #00d4aa !important;
    border-radius: 10px !important;
}
div[data-baseweb="popover"] * {
    background-color: #1a2f45 !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #00d4aa !important;
    color: #0b1520 !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #111f30;
    border-radius: 12px;
    padding: 5px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #c8dde8 !important;
    font-size: 0.97rem !important;
    font-weight: 800 !important;
    padding: 10px 22px !important;
    border-radius: 9px !important;
    background-color: #1a2f45 !important;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background-color: #00d4aa !important;
    color: #0b1520 !important;
    box-shadow: 0 0 12px rgba(0,212,170,0.5) !important;
}

/* ── SLIDERS ── */
.stSlider > div > div > div > div { background: #00d4aa !important; }

/* ── MULTISELECT ── */
.stMultiSelect [data-baseweb="tag"] {
    background-color: #00d4aa !important;
    color: #0b1520 !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
}
.stMultiSelect * { color: #f0f4f8 !important; font-weight: 700 !important; }
.stMultiSelect > div > div {
    background-color: #1a2f45 !important;
    border: 2px solid #00d4aa !important;
    border-radius: 10px !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #009e7f) !important;
    color: #0b1520 !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    padding: 0.75rem 1.5rem !important;
    letter-spacing: 1px !important;
    box-shadow: 0 0 18px rgba(0,212,170,0.45) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 28px rgba(0,212,170,0.7) !important;
    transform: translateY(-1px) !important;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
    background: #1a2f45 !important;
    color: #00d4aa !important;
    border: 2px solid #00d4aa !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* ── INFO BOX ── */
.stAlert { background-color: #1a2f45 !important; border-color: #00d4aa !important; color: #f0f4f8 !important; border-radius: 10px !important; }

/* ── DATAFRAME ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── HEADER ── */
.eco-header {
    background: linear-gradient(135deg, #0d3558 0%, #111f30 100%);
    border: 2px solid #00d4aa;
    border-radius: 18px;
    padding: 2.8rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 0 40px rgba(0,212,170,0.15), inset 0 0 40px rgba(0,0,0,0.2);
}
.eco-header h1 {
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    color: #00d4aa !important;
    letter-spacing: 4px;
    margin: 0;
    text-shadow: 0 0 25px rgba(0,212,170,0.7), 0 0 50px rgba(0,212,170,0.3);
}
.eco-header p {
    color: #c8f0e8 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    margin: 0.7rem 0 0;
    letter-spacing: 2px;
    text-shadow: 0 0 12px rgba(200,240,232,0.4);
}

/* ── METRIC CARDS ── */
.metric-card {
    background: linear-gradient(135deg, #111f30, #1a2f45);
    border: 2px solid #1e3a55;
    border-radius: 14px;
    padding: 1.6rem 1rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,212,170,0.08);
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #00d4aa; box-shadow: 0 0 20px rgba(0,212,170,0.2); }
.metric-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #00d4aa !important;
    text-shadow: 0 0 18px rgba(0,212,170,0.55);
    display: block;
}
.metric-label {
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    color: #90b8c8 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.5rem;
    display: block;
}

/* ── SECTION TITLES ── */
.section-title {
    font-size: 0.95rem !important;
    font-weight: 900 !important;
    color: #00d4aa !important;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    border-bottom: 2px solid #1e3a55;
    padding-bottom: 0.6rem;
    margin-bottom: 1.2rem;
    text-shadow: 0 0 12px rgba(0,212,170,0.45);
}

/* ── ALERT BOXES ── */
.alert-box {
    border-radius: 10px;
    padding: 0.9rem 1.3rem;
    margin: 0.5rem 0;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    line-height: 1.7;
}
.alert-critical { background: rgba(255,60,60,0.12); border-left: 4px solid #ff4c4c; color: #ffc0c0 !important; }
.alert-warning  { background: rgba(255,200,0,0.10); border-left: 4px solid #ffc800; color: #fff0a0 !important; }
.alert-success  { background: rgba(0,212,170,0.10); border-left: 4px solid #00d4aa; color: #aaefdf !important; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model ────────────────────────────────────────
@st.cache_data
def load_data():
    d = {
        'Number of Servers': [100,200,150,300,250,180,220,270,130,190],
        'Temperature (C)':   [25,35,30,40,38,28,33,39,26,31],
        'Cooling Hours':     [2,5,3,7,6,3,4,6,2,4],
        'Crypto Mining':     [0,1,0,1,1,0,1,1,0,0],
        'Water Used (L)':    [500,1200,800,2000,1700,650,1100,1850,520,900]
    }
    df = pd.DataFrame(d)
    df['Crypto Status'] = df['Crypto Mining'].map({0:'No Mining',1:'Mining Active'})
    df['Risk Level'] = ['Low','High','Medium','Critical','High','Low','High','Critical','Low','Medium']
    return df

@st.cache_resource
def train_model(df):
    X = df[['Number of Servers','Temperature (C)','Cooling Hours','Crypto Mining']]
    y = df['Water Used (L)']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    m = RandomForestRegressor(n_estimators=100,random_state=42)
    m.fit(Xtr,ytr)
    exp = shap.Explainer(m,X)
    sv  = exp(X)
    return m,exp,sv,X

def dark_fig(figsize=(8,4)):
    fig,ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#111f30')
    ax.set_facecolor('#0b1520')
    ax.tick_params(colors='#c8dde8', labelsize=11)
    ax.xaxis.label.set_color('#c8dde8')
    ax.yaxis.label.set_color('#c8dde8')
    ax.title.set_color('#00d4aa')
    for s in ax.spines.values(): s.set_edgecolor('#1e3a55')
    ax.grid(axis='y', color='#1e3a55', linestyle='--', alpha=0.6)
    return fig,ax

sensor_data = load_data()
model,explainer,shap_values,X = train_model(sensor_data)

# ── Header ──────────────────────────────────────────────
st.markdown("""
<div class="eco-header">
  <h1>💧 EcoXAI</h1>
  <p>Saving Water · Explaining Why · Intelligently</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚡ Live Controls</div>', unsafe_allow_html=True)
    servers = st.slider("Number of Servers", 50, 400, 250, step=10)
    temp    = st.slider("Temperature (°C)", 20, 45, 37)
    cooling = st.slider("Cooling Hours", 1, 10, 5)
    crypto  = st.selectbox("Crypto Mining Status", ["Inactive", "Active"])
    crypto_val = 1 if crypto == "Active" else 0
    st.markdown("---")
    predict_btn = st.button("🔍 Run EcoXAI Prediction")
    st.markdown("---")
    st.markdown('<div class="section-title">📍 Monitored Sites</div>', unsafe_allow_html=True)
    st.markdown("🔵 **Mumbai**&nbsp;&nbsp;🟢 **Hyderabad**&nbsp;&nbsp;🟡 **Chennai**", unsafe_allow_html=True)
    st.markdown("---")
    csv = sensor_data.to_csv(index=False)
    st.download_button("⬇ Download CSV", csv, "EcoXAI_Results.csv", "text/csv", use_container_width=True)

# ── KPI Cards ───────────────────────────────────────────
avg_w  = sensor_data['Water Used (L)'].mean()
max_w  = sensor_data['Water Used (L)'].max()
c_lift = ((sensor_data[sensor_data['Crypto Mining']==1]['Water Used (L)'].mean() /
           sensor_data[sensor_data['Crypto Mining']==0]['Water Used (L)'].mean())-1)*100
crit   = (sensor_data['Risk Level']=='Critical').sum()

k1,k2,k3,k4 = st.columns(4)
for col,val,label in [
    (k1, f"{avg_w:.0f} L",  "Avg Water / Reading"),
    (k2, f"{max_w:.0f} L",  "Peak Consumption"),
    (k3, f"+{c_lift:.0f}%", "Crypto Mining Uplift"),
    (k4, str(crit),          "Critical Risk Sites"),
]:
    with col:
        st.markdown(f'<div class="metric-card"><span class="metric-value">{val}</span><span class="metric-label">{label}</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────
tab1,tab2,tab3,tab4 = st.tabs(["📊  Data Explorer","📈  Visual Findings","🧠  XAI Explanations","🔮  Live Prediction"])

# TAB 1
with tab1:
    st.markdown('<div class="section-title">IoT Sensor Data — Data Centre Readings</div>', unsafe_allow_html=True)
    cf,_ = st.columns([2,3])
    with cf:
        rf = st.multiselect("Filter by Risk Level",['Low','Medium','High','Critical'],default=['Low','Medium','High','Critical'])
    filt = sensor_data[sensor_data['Risk Level'].isin(rf)]
    disp = filt[['Number of Servers','Temperature (C)','Cooling Hours','Crypto Status','Water Used (L)','Risk Level']].copy()
    def colour_risk(val):
        c = {'Low':'background-color:#0d3a1e;color:#00f0cc','Medium':'background-color:#3a2e0d;color:#ffd700',
             'High':'background-color:#3a1a0d;color:#ff8c42','Critical':'background-color:#3a0d0d;color:#ff5555'}
        return c.get(val,'')
    st.dataframe(disp.style.applymap(colour_risk,subset=['Risk Level']).format({'Water Used (L)':'{:.0f}'}),use_container_width=True,height=380)
    st.caption(f"Showing {len(filt)} of {len(sensor_data)} readings")

# TAB 2
with tab2:
    cl,cr = st.columns(2)
    with cl:
        st.markdown('<div class="section-title">Finding 1 · Servers vs Water</div>', unsafe_allow_html=True)
        fig,ax = dark_fig()
        ax.bar(sensor_data['Number of Servers'],sensor_data['Water Used (L)'],color='#00d4aa',width=8,alpha=0.9)
        ax.set_xlabel('Number of Servers',fontsize=12); ax.set_ylabel('Water Used (L)',fontsize=12)
        ax.set_title('More Servers = More Water Waste',fontsize=14,fontweight='bold')
        st.pyplot(fig)
        st.markdown('<div class="alert-box alert-success">Server count directly scales water consumption.</div>',unsafe_allow_html=True)
    with cr:
        st.markdown('<div class="section-title">Finding 2 · Temperature vs Water</div>', unsafe_allow_html=True)
        fig,ax = dark_fig()
        sc = ax.scatter(sensor_data['Temperature (C)'],sensor_data['Water Used (L)'],c=sensor_data['Water Used (L)'],cmap='YlOrRd',s=180,zorder=3,edgecolors='white',linewidths=0.5)
        fig.colorbar(sc,ax=ax,label='Water (L)')
        ax.set_xlabel('Temperature (°C)',fontsize=12); ax.set_ylabel('Water Used (L)',fontsize=12)
        ax.set_title('Higher Temp = Exponentially More Water',fontsize=14,fontweight='bold')
        st.pyplot(fig)
        st.markdown('<div class="alert-box alert-warning">Temperature is the #1 driver of water waste.</div>',unsafe_allow_html=True)
    cl2,cr2 = st.columns(2)
    with cl2:
        st.markdown('<div class="section-title">Finding 3 · Crypto Mining Impact</div>', unsafe_allow_html=True)
        an = sensor_data[sensor_data['Crypto Mining']==0]['Water Used (L)'].mean()
        ay = sensor_data[sensor_data['Crypto Mining']==1]['Water Used (L)'].mean()
        fig,ax = dark_fig()
        bars = ax.bar(['No Mining','Mining Active'],[an,ay],color=['#00d4aa','#ff4c4c'],width=0.45)
        ax.set_ylabel('Avg Water Used (L)',fontsize=12)
        ax.set_title('Crypto Mining Doubles Consumption',fontsize=14,fontweight='bold')
        for b,v in zip(bars,[an,ay]):
            ax.text(b.get_x()+b.get_width()/2,v+30,f'{v:.0f}L',ha='center',color='white',fontweight='bold',fontsize=12)
        st.pyplot(fig)
        st.markdown(f'<div class="alert-box alert-critical">Crypto mining raises water usage by <b>{c_lift:.0f}%</b>.</div>',unsafe_allow_html=True)
    with cr2:
        st.markdown('<div class="section-title">Finding 4 · Risk Level Analysis</div>', unsafe_allow_html=True)
        rm = {'Low Risk':sensor_data[sensor_data['Water Used (L)']<=600]['Water Used (L)'].mean(),
              'Medium Risk':sensor_data[(sensor_data['Water Used (L)']>600)&(sensor_data['Water Used (L)']<=900)]['Water Used (L)'].mean(),
              'High Risk':sensor_data[(sensor_data['Water Used (L)']>900)&(sensor_data['Water Used (L)']<=1500)]['Water Used (L)'].mean(),
              'Critical Risk':sensor_data[sensor_data['Water Used (L)']>1500]['Water Used (L)'].mean()}
        fig,ax = dark_fig()
        rb = ax.bar(rm.keys(),rm.values(),color=['#00d4aa','#ffc800','#ff8c42','#ff4c4c'],width=0.45)
        ax.set_ylabel('Avg Water Consumed (L)',fontsize=12)
        ax.set_title('Risk Level vs Water Waste',fontsize=14,fontweight='bold')
        for b,v in zip(rb,rm.values()):
            ax.text(b.get_x()+b.get_width()/2,v+30,f'{v:.0f}L',ha='center',color='white',fontweight='bold',fontsize=11)
        plt.xticks(rotation=10); st.pyplot(fig)
        st.markdown('<div class="alert-box alert-critical">Critical sites waste 4x more water than low-risk sites.</div>',unsafe_allow_html=True)

# TAB 3
with tab3:
    st.markdown('<div class="section-title">XAI Engine · Why Is Water Being Wasted?</div>', unsafe_allow_html=True)
    x1,x2 = st.columns([1,2])
    with x1:
        st.markdown("""
        <div style="background:#111f30;border:2px solid #00d4aa;border-radius:14px;padding:1.5rem;box-shadow:0 0 18px rgba(0,212,170,0.15);">
        <div style="color:#00d4aa;font-size:1rem;font-weight:900;letter-spacing:1.5px;margin-bottom:1rem;text-shadow:0 0 10px rgba(0,212,170,0.5);">HOW TO READ SHAP</div>
        <div style="color:#f0f4f8;font-size:0.95rem;font-weight:600;line-height:2.5;">
        📌 <b>Longer bar</b> = bigger impact<br>
        🔴 <b>Red</b> = increases water usage<br>
        🔵 <b>Blue</b> = decreases water usage<br>
        🌡️ <b>Temperature</b> = #1 driver<br>
        ⛏️ <b>Crypto Mining</b> = sharp spikes
        </div></div>""", unsafe_allow_html=True)
    with x2:
        idx = st.selectbox("Select reading to explain", range(len(sensor_data)),
                           format_func=lambda i: f"Reading {i}  —  {sensor_data['Risk Level'].iloc[i]} Risk  —  {sensor_data['Water Used (L)'].iloc[i]}L")
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.gcf().set_facecolor('#111f30')
        for ax in plt.gcf().get_axes():
            ax.set_facecolor('#0b1520'); ax.tick_params(colors='#c8dde8',labelsize=11)
        st.pyplot(plt.gcf()); plt.close('all')
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Global Factor Importance — All Readings</div>', unsafe_allow_html=True)
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    plt.gcf().set_facecolor('#111f30')
    for ax in plt.gcf().get_axes():
        ax.set_facecolor('#0b1520'); ax.tick_params(colors='#c8dde8',labelsize=11); ax.xaxis.label.set_color('#c8dde8')
    st.pyplot(plt.gcf()); plt.close('all')
    st.markdown('<div class="alert-box alert-warning">🌡️ <b>Temperature</b> is consistently the #1 driver of water waste across all readings.</div>', unsafe_allow_html=True)

# TAB 4
with tab4:
    st.markdown('<div class="section-title">🔮 EcoXAI Live Prediction Engine</div>', unsafe_allow_html=True)
    st.caption("Adjust sliders in the sidebar → Press Run EcoXAI Prediction")
    p1,p2,p3,p4 = st.columns(4)
    for col,val,label,colour in [
        (p1, str(servers), "Servers", "#00d4aa"),
        (p2, f"{temp}°C",  "Temperature", "#ff8c42" if temp>=35 else "#00d4aa"),
        (p3, f"{cooling}h","Cooling Hours","#00d4aa"),
        (p4, crypto,       "Crypto Mining","#ff4c4c" if crypto_val else "#00d4aa"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{colour};text-shadow:0 0 16px {colour}99;">{val}</span><span class="metric-label">{label}</span></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if predict_btn:
        live = pd.DataFrame({'Number of Servers':[servers],'Temperature (C)':[temp],'Cooling Hours':[cooling],'Crypto Mining':[crypto_val]})
        pred = model.predict(live)[0]
        if pred<=600:   rl,rc = "LOW RISK",    "#00d4aa"
        elif pred<=900: rl,rc = "MEDIUM RISK",  "#ffc800"
        elif pred<=1500:rl,rc = "HIGH RISK",    "#ff8c42"
        else:           rl,rc = "CRITICAL RISK","#ff4c4c"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#111f30,#1a2f45);border:3px solid {rc};
                    border-radius:16px;padding:2.2rem;text-align:center;margin:1.5rem 0;
                    box-shadow:0 0 30px {rc}33;">
            <div style="font-size:0.85rem;color:#90b8c8;letter-spacing:3px;font-weight:800;margin-bottom:0.6rem;">PREDICTED WATER CONSUMPTION</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:3.8rem;color:{rc};font-weight:700;text-shadow:0 0 25px {rc}99;line-height:1.1;">{pred:,.0f} L</div>
            <div style="font-size:1rem;color:{rc};font-weight:900;letter-spacing:3px;margin-top:0.6rem;text-shadow:0 0 12px {rc}88;">{rl}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("### 🧠 EcoXAI Explains Why")
        reasons = []
        if temp>=35:      reasons.append(("🌡️ Primary Cause",  f"Temperature at {temp}°C exceeds safe threshold of 30°C","alert-critical"))
        if servers>=200:  reasons.append(("💻 Secondary Cause", f"{servers} servers generating excessive heat load","alert-warning"))
        if crypto_val:    reasons.append(("⛏️ Amplifier",       "Crypto mining active — highest risk multiplier","alert-critical"))
        if cooling>=5:    reasons.append(("❄️ Cooling Load",    f"{cooling} hours of active cooling required","alert-warning"))
        if not reasons:   reasons.append(("✅ Status",          "All conditions within normal operating range","alert-success"))
        for lbl,msg,cls in reasons:
            st.markdown(f'<div class="alert-box {cls}"><b>{lbl}</b>: {msg}</div>', unsafe_allow_html=True)
        st.markdown("### ✅ EcoXAI Recommends")
        recs = []
        if temp>=35:     recs.append("Reduce data centre temperature to below 30°C immediately")
        if servers>=200: recs.append("Temporarily reduce active server load where possible")
        if crypto_val:   recs.append("Reschedule crypto mining to cooler night-time hours")
        if cooling>=5:   recs.append("Evaluate liquid cooling or adiabatic cooling alternatives")
        if not recs:     recs.append("No immediate action required — maintain current conditions")
        for i,r in enumerate(recs,1):
            st.markdown(f'<div class="alert-box alert-success"><b>Action {i}:</b> {r}</div>', unsafe_allow_html=True)
        st.markdown("### 💧 Expected Water Savings")
        st.markdown(f"""<div class="alert-box alert-success">
        Estimated reduction: <b>40 – 60%</b><br>
        Potential savings: <b>{pred*0.4:,.0f} – {pred*0.6:,.0f} litres</b> per monitoring cycle<br>
        Environmental impact: Reduced freshwater depletion near Mumbai, Hyderabad and Chennai
        </div>""", unsafe_allow_html=True)
    else:
        st.info("👈  Adjust the sliders in the sidebar, then press  Run EcoXAI Prediction  to see results.")

# ── Footer ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#00d4aa;font-size:0.85rem;font-weight:700;
            letter-spacing:2.5px;padding:0.6rem 0;text-shadow:0 0 12px rgba(0,212,170,0.4);">
💧 EcoXAI FRAMEWORK &nbsp;·&nbsp; SAVING WATER &nbsp;·&nbsp; EXPLAINING WHY &nbsp;·&nbsp; INTELLIGENTLY
</div>
""", unsafe_allow_html=True)

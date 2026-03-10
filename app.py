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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #0f1923;
    color: #e8eaed;
}

section[data-testid="stSidebar"] {
    background-color: #162032;
    border-right: 2px solid #00c9a7;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #162032;
    border-radius: 10px;
    padding: 6px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    background-color: #1e2d40 !important;
}
.stTabs [aria-selected="true"] {
    background-color: #00c9a7 !important;
    color: #0f1923 !important;
}

.eco-header {
    background: linear-gradient(135deg, #0f3460 0%, #162032 100%);
    border: 2px solid #00c9a7;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
}
.eco-header h1 {
    font-family: 'Inter', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #00c9a7;
    letter-spacing: 3px;
    margin: 0;
}
.eco-header p {
    color: #a8d8d0;
    font-size: 1.1rem;
    font-weight: 500;
    margin: 0.6rem 0 0;
    letter-spacing: 1px;
}

.metric-card {
    background: #162032;
    border: 2px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Fira Code', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00c9a7;
}
.metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #8fa8c0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #00c9a7;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 2px solid #1e3a5f;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

.alert-box {
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin: 0.6rem 0;
    font-size: 0.95rem;
    font-weight: 500;
    line-height: 1.6;
}
.alert-critical { background: rgba(220,53,69,0.15); border-left: 4px solid #dc3545; color: #ffb3b3; }
.alert-warning  { background: rgba(255,193,7,0.12); border-left: 4px solid #ffc107; color: #ffe680; }
.alert-success  { background: rgba(0,201,167,0.12); border-left: 4px solid #00c9a7; color: #a8f0e0; }

label { color: #a8d8d0 !important; font-size: 0.95rem !important; font-weight: 500 !important; }

.stButton > button {
    background: linear-gradient(135deg, #00c9a7, #0f7a6b);
    color: #0f1923;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 700;
    padding: 0.7rem 1.5rem;
    letter-spacing: 1px;
    width: 100%;
}

p, span, div { color: #e8eaed; }
h1, h2, h3, h4 { color: #ffffff; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

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
    fig.patch.set_facecolor('#162032')
    ax.set_facecolor('#0f1923')
    ax.tick_params(colors='#a8d8d0', labelsize=10)
    ax.xaxis.label.set_color('#a8d8d0')
    ax.yaxis.label.set_color('#a8d8d0')
    ax.title.set_color('#00c9a7')
    for s in ax.spines.values(): s.set_edgecolor('#1e3a5f')
    ax.grid(axis='y', color='#1e3a5f', linestyle='--', alpha=0.5)
    return fig,ax

sensor_data = load_data()
model,explainer,shap_values,X = train_model(sensor_data)

st.markdown("""
<div class="eco-header">
  <h1>💧 EcoXAI</h1>
  <p>Saving Water · Explaining Why · Intelligently</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-title">⚡ Live Controls</div>', unsafe_allow_html=True)
    servers = st.slider("Number of Servers", 50, 400, 250, step=10)
    temp    = st.slider("Temperature (°C)", 20, 45, 37)
    cooling = st.slider("Cooling Hours", 1, 10, 5)
    crypto  = st.selectbox("Crypto Mining", ["Inactive","Active"])
    crypto_val = 1 if crypto == "Active" else 0
    st.markdown("---")
    predict_btn = st.button("🔍 Run EcoXAI Prediction")
    st.markdown("---")
    st.markdown('<div class="section-title">📍 Monitored Sites</div>', unsafe_allow_html=True)
    st.markdown("🔵 **Mumbai** &nbsp; 🟢 **Hyderabad** &nbsp; 🟡 **Chennai**", unsafe_allow_html=True)
    st.markdown("---")
    csv = sensor_data.to_csv(index=False)
    st.download_button("⬇ Download CSV", csv, "EcoXAI_Results.csv", "text/csv", use_container_width=True)

avg_w  = sensor_data['Water Used (L)'].mean()
max_w  = sensor_data['Water Used (L)'].max()
c_lift = ((sensor_data[sensor_data['Crypto Mining']==1]['Water Used (L)'].mean() /
           sensor_data[sensor_data['Crypto Mining']==0]['Water Used (L)'].mean())-1)*100
crit   = (sensor_data['Risk Level']=='Critical').sum()

k1,k2,k3,k4 = st.columns(4)
for col, val, label in [
    (k1, f"{avg_w:.0f} L", "Avg Water / Reading"),
    (k2, f"{max_w:.0f} L", "Peak Consumption"),
    (k3, f"+{c_lift:.0f}%", "Crypto Mining Uplift"),
    (k4, str(crit), "Critical Risk Sites")
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs(["📊 Data Explorer","📈 Visual Findings","🧠 XAI Explanations","🔮 Live Prediction"])

with tab1:
    st.markdown('<div class="section-title">IoT Sensor Data — Data Centre Readings</div>', unsafe_allow_html=True)
    cf,_ = st.columns([2,3])
    with cf:
        rf = st.multiselect("Filter by Risk Level",['Low','Medium','High','Critical'],
                            default=['Low','Medium','High','Critical'])
    filt = sensor_data[sensor_data['Risk Level'].isin(rf)]
    disp = filt[['Number of Servers','Temperature (C)','Cooling Hours','Crypto Status','Water Used (L)','Risk Level']].copy()
    def colour_risk(val):
        c = {'Low':'background-color:#0d3a1e;color:#00f0cc',
             'Medium':'background-color:#3a2e0d;color:#ffd700',
             'High':'background-color:#3a1a0d;color:#ff8c42',
             'Critical':'background-color:#3a0d0d;color:#ff5555'}
        return c.get(val,'')
    st.dataframe(
        disp.style.applymap(colour_risk, subset=['Risk Level']).format({'Water Used (L)':'{:.0f}'}),
        use_container_width=True, height=380
    )
    st.caption(f"Showing {len(filt)} of {len(sensor_data)} readings")

with tab2:
    cl,cr = st.columns(2)
    with cl:
        st.markdown('<div class="section-title">Finding 1 · Servers vs Water</div>', unsafe_allow_html=True)
        fig,ax = dark_fig()
        ax.bar(sensor_data['Number of Servers'], sensor_data['Water Used (L)'], color='#00c9a7', width=8, alpha=0.9)
        ax.set_xlabel('Number of Servers', fontsize=11)
        ax.set_ylabel('Water Used (L)', fontsize=11)
        ax.set_title('More Servers = More Water Waste', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        st.markdown('<div class="alert-box alert-success">Server count directly scales water consumption.</div>', unsafe_allow_html=True)
    with cr:
        st.markdown('<div class="section-title">Finding 2 · Temperature vs Water</div>', unsafe_allow_html=True)
        fig,ax = dark_fig()
        sc = ax.scatter(sensor_data['Temperature (C)'], sensor_data['Water Used (L)'],
                        c=sensor_data['Water Used (L)'], cmap='YlOrRd', s=180, zorder=3, edgecolors='white', linewidths=0.5)
        fig.colorbar(sc, ax=ax, label='Water (L)')
        ax.set_xlabel('Temperature (°C)', fontsize=11)
        ax.set_ylabel('Water Used (L)', fontsize=11)
        ax.set_title('Higher Temp = Exponentially More Water', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        st.markdown('<div class="alert-box alert-warning">Temperature is the #1 driver of water waste.</div>', unsafe_allow_html=True)
    cl2,cr2 = st.columns(2)
    with cl2:
        st.markdown('<div class="section-title">Finding 3 · Crypto Mining Impact</div>', unsafe_allow_html=True)
        an = sensor_data[sensor_data['Crypto Mining']==0]['Water Used (L)'].mean()
        ay = sensor_data[sensor_data['Crypto Mining']==1]['Water Used (L)'].mean()
        fig,ax = dark_fig()
        bars = ax.bar(['No Mining','Mining Active'],[an,ay], color=['#00c9a7','#ff4444'], width=0.45)
        ax.set_ylabel('Avg Water Used (L)', fontsize=11)
        ax.set_title('Crypto Mining Doubles Consumption', fontsize=13, fontweight='bold')
        for b,v in zip(bars,[an,ay]):
            ax.text(b.get_x()+b.get_width()/2, v+30, f'{v:.0f}L', ha='center', color='white', fontweight='bold', fontsize=12)
        st.pyplot(fig)
        st.markdown(f'<div class="alert-box alert-critical">Crypto mining raises water usage by <b>{c_lift:.0f}%</b>.</div>', unsafe_allow_html=True)
    with cr2:
        st.markdown('<div class="section-title">Finding 4 · Risk Level Analysis</div>', unsafe_allow_html=True)
        rm = {
            'Low Risk':      sensor_data[sensor_data['Water Used (L)']<=600]['Water Used (L)'].mean(),
            'Medium Risk':   sensor_data[(sensor_data['Water Used (L)']>600)&(sensor_data['Water Used (L)']<=900)]['Water Used (L)'].mean(),
            'High Risk':     sensor_data[(sensor_data['Water Used (L)']>900)&(sensor_data['Water Used (L)']<=1500)]['Water Used (L)'].mean(),
            'Critical Risk': sensor_data[sensor_data['Water Used (L)']>1500]['Water Used (L)'].mean(),
        }
        fig,ax = dark_fig()
        rb = ax.bar(rm.keys(), rm.values(), color=['#00c9a7','#ffd700','#ff8c42','#ff4444'], width=0.45)
        ax.set_ylabel('Avg Water Consumed (L)', fontsize=11)
        ax.set_title('Risk Level vs Water Waste', fontsize=13, fontweight='bold')
        for b,v in zip(rb,rm.values()):
            ax.text(b.get_x()+b.get_width()/2, v+30, f'{v:.0f}L', ha='center', color='white', fontweight='bold', fontsize=11)
        plt.xticks(rotation=10)
        st.pyplot(fig)
        st.markdown('<div class="alert-box alert-critical">Critical sites waste 4x more water than low-risk sites.</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-title">XAI Engine · Why Is Water Being Wasted?</div>', unsafe_allow_html=True)
    x1,x2 = st.columns([1,2])
    with x1:
        st.markdown("""
        <div style="background:#162032;border:2px solid #1e3a5f;border-radius:12px;padding:1.4rem;">
        <b style="color:#00c9a7;font-size:0.95rem;letter-spacing:1px;">HOW TO READ SHAP</b><br><br>
        <span style="color:#a8d8d0;font-size:0.92rem;line-height:2.2;">
        📌 <b>Longer bar</b> = bigger impact<br>
        🔴 <b>Red</b> = increases water usage<br>
        🔵 <b>Blue</b> = decreases water usage<br>
        🌡️ <b>Temperature</b> = #1 driver<br>
        ⛏️ <b>Crypto Mining</b> = sharp spikes
        </span>
        </div>
        """, unsafe_allow_html=True)
    with x2:
        idx = st.selectbox("Select reading to explain", range(len(sensor_data)),
                           format_func=lambda i: f"Reading {i} — {sensor_data['Risk Level'].iloc[i]} Risk — {sensor_data['Water Used (L)'].iloc[i]}L")
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.gcf().set_facecolor('#162032')
        for ax in plt.gcf().get_axes():
            ax.set_facecolor('#0f1923')
            ax.tick_params(colors='#a8d8d0', labelsize=10)
        st.pyplot(plt.gcf())
        plt.close('all')
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Global Factor Importance</div>', unsafe_allow_html=True)
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    plt.gcf().set_facecolor('#162032')
    for ax in plt.gcf().get_axes():
        ax.set_facecolor('#0f1923')
        ax.tick_params(colors='#a8d8d0', labelsize=10)
        ax.xaxis.label.set_color('#a8d8d0')
    st.pyplot(plt.gcf())
    plt.close('all')
    st.markdown('<div class="alert-box alert-warning">🌡️ <b>Temperature</b> is the #1 consistent driver of water waste.</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-title">🔮 EcoXAI Live Prediction Engine</div>', unsafe_allow_html=True)
    st.caption("Adjust sliders in the sidebar → Press Run EcoXAI Prediction")
    p1,p2,p3,p4 = st.columns(4)
    for col,val,label,colour in [
        (p1, str(servers), "Servers", "#00c9a7"),
        (p2, f"{temp}°C", "Temperature", "#ff8c42" if temp>=35 else "#00c9a7"),
        (p3, f"{cooling}h", "Cooling Hours", "#00c9a7"),
        (p4, crypto, "Crypto Mining", "#ff4444" if crypto_val else "#00c9a7"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{colour}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if predict_btn:
        live = pd.DataFrame({'Number of Servers':[servers],'Temperature (C)':[temp],'Cooling Hours':[cooling],'Crypto Mining':[crypto_val]})
        pred = model.predict(live)[0]
        if pred<=600: rl,rc = "LOW RISK","#00c9a7"
        elif pred<=900: rl,rc = "MEDIUM RISK","#ffd700"
        elif pred<=1500: rl,rc = "HIGH RISK","#ff8c42"
        else: rl,rc = "CRITICAL RISK","#ff4444"
        st.markdown(f"""
        <div style="background:#162032;border:3px solid {rc};border-radius:16px;
                    padding:2rem;text-align:center;margin:1.5rem 0;">
            <div style="font-size:0.85rem;color:#8fa8c0;letter-spacing:3px;font-weight:600;margin-bottom:0.5rem;">PREDICTED WATER CONSUMPTION</div>
            <div style="font-family:'Fira Code',monospace;font-size:3.5rem;color:{rc};font-weight:700;">{pred:,.0f} L</div>
            <div style="font-size:1rem;color:{rc};font-weight:700;letter-spacing:2px;margin-top:0.3rem;">{rl}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("### 🧠 EcoXAI Explains Why")
        reasons = []
        if temp>=35: reasons.append(("🌡️ Primary Cause", f"Temperature at {temp}°C exceeds safe threshold (30°C)", "alert-critical"))
        if servers>=200: reasons.append(("💻 Secondary Cause", f"{servers} servers generating significant heat load", "alert-warning"))
        if crypto_val: reasons.append(("⛏️ Amplifier", "Crypto mining running — highest risk multiplier", "alert-critical"))
        if cooling>=5: reasons.append(("❄️ Cooling Load", f"{cooling} hours of active cooling required", "alert-warning"))
        if not reasons: reasons.append(("✅ Status", "Conditions within normal range", "alert-success"))
        for label,msg,cls in reasons:
            st.markdown(f'<div class="alert-box {cls}"><b>{label}</b>: {msg}</div>', unsafe_allow_html=True)
        st.markdown("### ✅ EcoXAI Recommends")
        recs = []
        if temp>=35: recs.append("Reduce temperature to below 30°C immediately")
        if servers>=200: recs.append("Temporarily reduce active server load")
        if crypto_val: recs.append("Reschedule crypto mining to cooler night hours")
        if cooling>=5: recs.append("Evaluate liquid cooling or adiabatic alternatives")
        if not recs: recs.append("No immediate action required — maintain current conditions")
        for i,r in enumerate(recs,1):
            st.markdown(f'<div class="alert-box alert-success"><b>Action {i}:</b> {r}</div>', unsafe_allow_html=True)
        st.markdown("### 💧 Expected Water Savings")
        st.markdown(f"""
        <div class="alert-box alert-success">
        Estimated reduction: <b>40-60%</b><br>
        Potential savings: <b>{pred*0.4:,.0f} - {pred*0.6:,.0f} litres</b> per cycle<br>
        Impact: Reduced freshwater depletion near Mumbai, Hyderabad and Chennai
        </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Set conditions in the sidebar and press Run EcoXAI Prediction")

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#3a6a5a;font-size:0.8rem;font-weight:500;letter-spacing:2px;padding:0.5rem 0;">
💧 EcoXAI FRAMEWORK · SAVING WATER · EXPLAINING WHY · INTELLIGENTLY
</div>
""", unsafe_allow_html=True)

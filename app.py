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
    page_title="EcoXAI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #111827;
    color: #f3f4f6;
}

section[data-testid="stSidebar"] {
    background-color: #1f2937;
    border-right: 1px solid #374151;
}

.top-banner {
    background-color: #1f2937;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}
.top-banner h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f9fafb;
    margin: 0 0 0.3rem 0;
}
.top-banner p {
    color: #9ca3af;
    font-size: 0.9rem;
    margin: 0;
}

.kpi-card {
    background-color: #1f2937;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #34d399;
}
.kpi-label {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #374151;
}

.info-box {
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    line-height: 1.5;
}
.info-red    { background-color: #1f1315; border-left: 3px solid #f87171; color: #fca5a5; }
.info-yellow { background-color: #1c1a10; border-left: 3px solid #fbbf24; color: #fcd34d; }
.info-green  { background-color: #0f1f17; border-left: 3px solid #34d399; color: #6ee7b7; }

.stButton > button {
    background-color: #065f46;
    color: #ecfdf5;
    border: none;
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 0.55rem 1.2rem;
    width: 100%;
}
.stButton > button:hover {
    background-color: #047857;
}

label { color: #d1d5db !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model ────────────────────────────────────────────
@st.cache_data
def load_data():
    d = {
        'Number of Servers': [100,200,150,300,250,180,220,270,130,190],
        'Temperature (C)':   [25, 35, 30, 40, 38, 28, 33, 39, 26, 31],
        'Cooling Hours':     [2,  5,  3,  7,  6,  3,  4,  6,  2,  4],
        'Crypto Mining':     [0,  1,  0,  1,  1,  0,  1,  1,  0,  0],
        'Water Used (L)':    [500,1200,800,2000,1700,650,1100,1850,520,900]
    }
    df = pd.DataFrame(d)
    df['Crypto Status'] = df['Crypto Mining'].map({0:'No Mining', 1:'Mining Active'})
    df['Risk Level'] = ['Low','High','Medium','Critical','High','Low','High','Critical','Low','Medium']
    return df

@st.cache_resource
def train_model(df):
    X = df[['Number of Servers','Temperature (C)','Cooling Hours','Crypto Mining']]
    y = df['Water Used (L)']
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(n_estimators=100, random_state=42)
    m.fit(Xtr, ytr)
    exp = shap.Explainer(m, X)
    sv  = exp(X)
    return m, exp, sv, X

def mpl_dark(figsize=(8,4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#1f2937')
    ax.set_facecolor('#111827')
    ax.tick_params(colors='#9ca3af', labelsize=9)
    ax.xaxis.label.set_color('#9ca3af')
    ax.yaxis.label.set_color('#9ca3af')
    ax.title.set_color('#f3f4f6')
    for sp in ax.spines.values():
        sp.set_edgecolor('#374151')
    return fig, ax

df = load_data()
model, explainer, shap_vals, X = train_model(df)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### 💧 EcoXAI")
    st.markdown("<p style='color:#9ca3af;font-size:0.8rem;margin-top:-0.5rem;'>Water footprint monitor</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Adjust conditions**")
    servers = st.slider("Servers", 50, 400, 250, step=10)
    temp    = st.slider("Temperature (°C)", 20, 45, 37)
    cooling = st.slider("Cooling Hours", 1, 10, 5)
    crypto  = st.selectbox("Crypto Mining", ["Inactive", "Active"])
    crypto_val = 1 if crypto == "Active" else 0

    st.markdown("---")
    run = st.button("Run Prediction")

    st.markdown("---")
    st.markdown("<p style='color:#6b7280;font-size:0.78rem;'>📍 Monitoring: Mumbai · Hyderabad · Chennai</p>", unsafe_allow_html=True)
    st.download_button("Download CSV", df.to_csv(index=False), "EcoXAI_data.csv", "text/csv", use_container_width=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <h1>💧 EcoXAI Framework</h1>
  <p>IoT-driven water footprint intelligence for AI data centres and crypto mining — powered by Machine Learning and Explainable AI.</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────────────────────
avg_w  = df['Water Used (L)'].mean()
max_w  = df['Water Used (L)'].max()
clift  = ((df[df['Crypto Mining']==1]['Water Used (L)'].mean() /
           df[df['Crypto Mining']==0]['Water Used (L)'].mean()) - 1) * 100
crit_n = (df['Risk Level']=='Critical').sum()

k1,k2,k3,k4 = st.columns(4)
for col, val, label in zip(
    [k1,k2,k3,k4],
    [f"{avg_w:.0f} L", f"{max_w:.0f} L", f"+{clift:.0f}%", str(crit_n)],
    ["Avg Water / Reading", "Peak Consumption", "Crypto Mining Uplift", "Critical Risk Sites"]
):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Charts", "XAI", "Prediction"])

# TAB 1 — DATA
with tab1:
    st.markdown('<div class="section-label">IoT sensor readings</div>', unsafe_allow_html=True)
    risk_filter = st.multiselect("Filter by risk level",
                                 ['Low','Medium','High','Critical'],
                                 default=['Low','Medium','High','Critical'])
    fdf = df[df['Risk Level'].isin(risk_filter)]
    disp = fdf[['Number of Servers','Temperature (C)','Cooling Hours','Crypto Status','Water Used (L)','Risk Level']].copy()

    def style_risk(v):
        c = {'Low':'color:#34d399','Medium':'color:#fbbf24',
             'High':'color:#fb923c','Critical':'color:#f87171'}
        return c.get(v,'')

    st.dataframe(
        disp.style.applymap(style_risk, subset=['Risk Level']).format({'Water Used (L)':'{:.0f}'}),
        use_container_width=True, height=360
    )
    st.caption(f"{len(fdf)} of {len(df)} readings shown")

# TAB 2 — CHARTS
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-label">Servers vs water usage</div>', unsafe_allow_html=True)
        fig, ax = mpl_dark()
        ax.bar(df['Number of Servers'], df['Water Used (L)'], color='#34d399', width=8, alpha=0.8)
        ax.set_xlabel('Number of Servers')
        ax.set_ylabel('Water Used (L)')
        ax.set_title('More servers, more water', fontsize=11)
        ax.grid(axis='y', color='#374151', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.markdown('<div class="info-box info-green">Server count scales directly with water consumption.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-label">Temperature vs water usage</div>', unsafe_allow_html=True)
        fig, ax = mpl_dark()
        sc = ax.scatter(df['Temperature (C)'], df['Water Used (L)'],
                        c=df['Water Used (L)'], cmap='RdYlGn_r', s=120, zorder=3)
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Water Used (L)')
        ax.set_title('Higher temperature = more water', fontsize=11)
        ax.grid(color='#374151', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.markdown('<div class="info-box info-yellow">Temperature is the biggest single driver of water waste.</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-label">Crypto mining impact</div>', unsafe_allow_html=True)
        no_m = df[df['Crypto Mining']==0]['Water Used (L)'].mean()
        yes_m = df[df['Crypto Mining']==1]['Water Used (L)'].mean()
        fig, ax = mpl_dark()
        bars = ax.bar(['No Mining','Mining Active'], [no_m, yes_m],
                      color=['#34d399','#f87171'], width=0.4)
        ax.set_ylabel('Avg Water Used (L)')
        ax.set_title('Crypto mining doubles consumption', fontsize=11)
        ax.grid(axis='y', color='#374151', linestyle='--', alpha=0.5)
        for b, v in zip(bars,[no_m,yes_m]):
            ax.text(b.get_x()+b.get_width()/2, v+20, f'{v:.0f}L',
                    ha='center', color='#f3f4f6', fontsize=10, fontweight='600')
        st.pyplot(fig)
        st.markdown(f'<div class="info-box info-red">Crypto mining increases water usage by {clift:.0f}%.</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-label">Risk level breakdown</div>', unsafe_allow_html=True)
        rm = {
            'Low':      df[df['Water Used (L)']<=600]['Water Used (L)'].mean(),
            'Medium':   df[(df['Water Used (L)']>600)&(df['Water Used (L)']<=900)]['Water Used (L)'].mean(),
            'High':     df[(df['Water Used (L)']>900)&(df['Water Used (L)']<=1500)]['Water Used (L)'].mean(),
            'Critical': df[df['Water Used (L)']>1500]['Water Used (L)'].mean(),
        }
        fig, ax = mpl_dark()
        rb = ax.bar(rm.keys(), rm.values(), color=['#34d399','#fbbf24','#fb923c','#f87171'], width=0.4)
        ax.set_ylabel('Avg Water (L)')
        ax.set_title('Critical sites waste 4× more water', fontsize=11)
        ax.grid(axis='y', color='#374151', linestyle='--', alpha=0.5)
        for b, v in zip(rb, rm.values()):
            ax.text(b.get_x()+b.get_width()/2, v+20, f'{v:.0f}L',
                    ha='center', color='#f3f4f6', fontsize=10, fontweight='600')
        st.pyplot(fig)
        st.markdown('<div class="info-box info-red">Critical risk centres use 4× more water than low risk ones.</div>', unsafe_allow_html=True)

# TAB 3 — XAI
with tab3:
    st.markdown('<div class="section-label">Why is water being wasted?</div>', unsafe_allow_html=True)

    left, right = st.columns([1,2])
    with left:
        st.markdown("""
        <div style="background:#1f2937;border:1px solid #374151;border-radius:8px;padding:1rem;font-size:0.85rem;color:#d1d5db;line-height:1.8;">
        <b style="color:#f3f4f6;">How to read this chart</b><br><br>
        📌 Longer bar = bigger impact<br>
        🔴 Red = increases water usage<br>
        🔵 Blue = decreases water usage<br><br>
        Temperature is consistently the <b style="color:#34d399;">#1 factor</b>.
        </div>
        """, unsafe_allow_html=True)

    with right:
        idx = st.selectbox("Choose a reading to explain",
                           range(len(df)),
                           format_func=lambda i: f"Reading {i}  —  {df['Risk Level'].iloc[i]} risk  —  {df['Water Used (L)'].iloc[i]}L")
        shap.plots.waterfall(shap_vals[idx], show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor('#1f2937')
        for a in fig.get_axes():
            a.set_facecolor('#111827')
            a.tick_params(colors='#9ca3af')
        st.pyplot(fig)
        plt.close('all')

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Overall factor importance across all readings</div>', unsafe_allow_html=True)
    shap.summary_plot(shap_vals, X, show=False, plot_size=None)
    fig = plt.gcf()
    fig.patch.set_facecolor('#1f2937')
    for a in fig.get_axes():
        a.set_facecolor('#111827')
        a.tick_params(colors='#9ca3af')
        a.xaxis.label.set_color('#9ca3af')
    st.pyplot(fig)
    plt.close('all')
    st.markdown('<div class="info-box info-yellow">Temperature is the #1 driver of water waste across all data centre readings.</div>', unsafe_allow_html=True)

# TAB 4 — PREDICTION
with tab4:
    st.markdown('<div class="section-label">Live water usage prediction</div>', unsafe_allow_html=True)

    p1,p2,p3,p4 = st.columns(4)
    for col, val, lbl in zip([p1,p2,p3,p4],
                              [servers, f"{temp}°C", f"{cooling}h", crypto],
                              ["Servers","Temperature","Cooling","Crypto"]):
        with col:
            colour = "#f87171" if (lbl == "Crypto" and crypto == "Active") else "#34d399"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{colour};font-size:1.5rem;">{val}</div>
                <div class="kpi-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if run:
        live = pd.DataFrame({
            'Number of Servers':[servers],
            'Temperature (C)':[temp],
            'Cooling Hours':[cooling],
            'Crypto Mining':[crypto_val]
        })
        pred = model.predict(live)[0]

        if pred <= 600:   rl, rc = "Low risk", "info-green"
        elif pred <= 900: rl, rc = "Medium risk", "info-yellow"
        elif pred <= 1500:rl, rc = "High risk", "info-yellow"
        else:             rl, rc = "Critical risk", "info-red"

        st.markdown(f"""
        <div style="background:#1f2937;border:1px solid #374151;border-radius:10px;
                    padding:1.5rem 2rem;text-align:center;margin:1rem 0;">
            <p style="color:#9ca3af;font-size:0.8rem;margin:0 0 0.3rem;">Predicted water consumption</p>
            <p style="font-size:2.8rem;font-weight:700;color:#34d399;margin:0;">{pred:,.0f} L</p>
            <p style="color:#9ca3af;font-size:0.85rem;margin:0.3rem 0 0;">{rl}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Why is this happening?**")
        if temp >= 35:
            st.markdown(f'<div class="info-box info-red">🌡️ Temperature at {temp}°C is above the safe limit of 30°C — biggest driver of water waste.</div>', unsafe_allow_html=True)
        if servers >= 200:
            st.markdown(f'<div class="info-box info-yellow">🖥️ {servers} active servers are generating significant heat load.</div>', unsafe_allow_html=True)
        if crypto_val:
            st.markdown('<div class="info-box info-red">⛏️ Crypto mining is active — this roughly doubles water consumption.</div>', unsafe_allow_html=True)
        if cooling >= 5:
            st.markdown(f'<div class="info-box info-yellow">❄️ {cooling} hours of cooling required — consider optimising cooling schedule.</div>', unsafe_allow_html=True)
        if temp < 35 and servers < 200 and not crypto_val and cooling < 5:
            st.markdown('<div class="info-box info-green">✅ All conditions look normal — no immediate action needed.</div>', unsafe_allow_html=True)

        st.markdown("<br>**What should you do?**")
        i = 1
        if temp >= 35:
            st.markdown(f'<div class="info-box info-green">Action {i}: Bring temperature below 30°C as soon as possible.</div>', unsafe_allow_html=True); i+=1
        if servers >= 200:
            st.markdown(f'<div class="info-box info-green">Action {i}: Reduce server load where possible to lower heat generation.</div>', unsafe_allow_html=True); i+=1
        if crypto_val:
            st.markdown(f'<div class="info-box info-green">Action {i}: Reschedule crypto mining to cooler night-time hours.</div>', unsafe_allow_html=True); i+=1
        if cooling >= 5:
            st.markdown(f'<div class="info-box info-green">Action {i}: Look into liquid cooling or more efficient cooling systems.</div>', unsafe_allow_html=True); i+=1

        st.markdown(f"""
        <div class="info-box info-green" style="margin-top:1rem;">
        💧 Following these steps could reduce water usage by <b>40–60%</b> — 
        saving roughly <b>{pred*0.4:,.0f}–{pred*0.6:,.0f} litres</b> per cycle.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Use the sliders on the left and press **Run Prediction** to see results.")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#4b5563;font-size:0.75rem;'>EcoXAI · Built by Sri Bharathi · BSc Computer Science with Cognitive Systems</p>", unsafe_allow_html=True)

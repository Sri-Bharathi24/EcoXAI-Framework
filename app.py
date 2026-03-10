import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(
    page_title="EcoXAI",
    page_icon="💧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .hero-box {
        background: linear-gradient(135deg, #0d1b2a, #1a3a4a);
        border: 2px solid #00d4aa;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-box {
        background: #0d1b2a;
        border: 1px solid #00d4aa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .footer-box {
        background: #0d1b2a;
        border: 1px solid #00d4aa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Data
data = {
    'Number of Servers':  [100, 200, 150, 300, 250, 180, 220, 270, 130, 190],
    'Temperature (C)':    [25,  35,  30,  40,  38,  28,  33,  39,  26,  31],
    'Cooling Hours':      [2,   5,   3,   7,   6,   3,   4,   6,   2,   4],
    'Crypto Mining':      [0,   1,   0,   1,   1,   0,   1,   1,   0,   0],
    'Water Used (L)':     [500, 1200,800, 2000,1700,650, 1100,1850,520, 900],
    'Location':           ['Chennai', 'Mumbai', 'Hyderabad', 'Mumbai',
                          'Chennai', 'Hyderabad', 'Mumbai', 'Chennai',
                          'Hyderabad', 'Mumbai']
}
df = pd.DataFrame(data)
df['Risk Level'] = ['Low','High','Medium','Critical','High',
                    'Low','High','Critical','Low','Medium']
df['Crypto Status'] = df['Crypto Mining'].map({0:'No Mining', 1:'Mining Active'})

# Train Model
X = df[['Number of Servers', 'Temperature (C)', 'Cooling Hours', 'Crypto Mining']]
y = df['Water Used (L)']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Hero Section
st.markdown("""
<div class='hero-box'>
    <h1 style='color: white; font-size: 3rem;'>💧 EcoXAI</h1>
    <p style='color: #00d4aa; font-size: 1.2rem;'>
        Saving Water · Explaining Why · Intelligently
    </p>
    <p style='color: #aaa; font-size: 0.95rem; margin-top: 10px;'>
        An IoT-Driven Explainable AI Framework that monitors water footprint
        in AI Data Centres and Crypto Mining — and explains exactly why
        water is being wasted.
    </p>
</div>
""", unsafe_allow_html=True)

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

avg_water = df['Water Used (L)'].mean()
peak_water = df['Water Used (L)'].max()
crypto_uplift = ((df[df['Crypto Mining']==1]['Water Used (L)'].mean() /
                  df[df['Crypto Mining']==0]['Water Used (L)'].mean()) - 1) * 100
critical_sites = len(df[df['Risk Level'] == 'Critical'])

with col1:
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-value'>{avg_water:.0f} L</div>
        <div class='metric-label'>Avg Water / Reading</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-value'>{peak_water:.0f} L</div>
        <div class='metric-label'>Peak Consumption</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-value'>+{crypto_uplift:.0f}%</div>
        <div class='metric-label'>Crypto Mining Uplift</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-value'>{critical_sites}</div>
        <div class='metric-label'>Critical Risk Sites</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### ⚡ LIVE CONTROLS")
    st.markdown("---")
    servers = st.slider("Number of Servers", 100, 300, 250)
    temp = st.slider("Temperature (°C)", 25, 40, 37)
    cooling = st.slider("Cooling Hours", 2, 7, 5)
    crypto = st.selectbox("Crypto Mining", [0, 1],
                          format_func=lambda x: "Active" if x==1 else "Inactive")
    st.markdown("---")
    if st.button("🔍 Run EcoXAI Prediction", use_container_width=True):
        st.session_state.run_prediction = True

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer",
    "📈 Visual Findings",
    "🔍 XAI Explanations",
    "⚡ Live Prediction"
])

# TAB 1 - Data Explorer
with tab1:
    st.subheader("IoT Sensor Data from Indian Data Centres")
    st.markdown("*Real time readings from Chennai, Hyderabad and Mumbai data centres*")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("📍 India Location Water Usage")

    location_data = df.groupby('Location')['Water Used (L)'].mean().reset_index()
    colors = ['#00d4aa', '#ff6b6b', '#ffd93d']

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    bars = ax.bar(location_data['Location'],
                  location_data['Water Used (L)'],
                  color=colors, width=0.4)
    for bar, val in zip(bars, location_data['Water Used (L)']):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 10, f'{val:.0f}L',
                ha='center', color='white', fontweight='bold')
    ax.set_ylabel('Average Water (Litres)', color='white')
    ax.set_title('EcoXAI: India Data Centre Water Usage by City',
                 color='white', pad=15)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    st.pyplot(fig)

# TAB 2 - Visual Findings
with tab2:
    st.subheader("Water Usage Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.patch.set_facecolor('#0d1b2a')
        ax1.set_facecolor('#0d1b2a')
        ax1.bar(df['Number of Servers'], df['Water Used (L)'],
                color='steelblue', width=8)
        ax1.set_xlabel('Number of Servers', color='white')
        ax1.set_ylabel('Water Used (L)', color='white')
        ax1.set_title('Servers vs Water Usage', color='white')
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig1)
        st.caption("More servers = More water wasted")

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#0d1b2a')
        ax2.set_facecolor('#0d1b2a')
        ax2.scatter(df['Temperature (C)'], df['Water Used (L)'],
                    color='#ff6b6b', s=150)
        ax2.set_xlabel('Temperature (C)', color='white')
        ax2.set_ylabel('Water Used (L)', color='white')
        ax2.set_title('Temperature vs Water Usage', color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig2)
        st.caption("Higher temperature = Exponentially more water")

    col_c, col_d = st.columns(2)

    with col_c:
        crypto_off = df[df['Crypto Mining']==0]['Water Used (L)'].mean()
        crypto_on = df[df['Crypto Mining']==1]['Water Used (L)'].mean()
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor('#0d1b2a')
        ax3.set_facecolor('#0d1b2a')
        bars3 = ax3.bar(['No Mining', 'Mining Active'],
                        [crypto_off, crypto_on],
                        color=['#00d4aa', '#ff6b6b'], width=0.4)
        for bar, val in zip(bars3, [crypto_off, crypto_on]):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     val + 10, f'{val:.0f}L',
                     ha='center', color='white', fontweight='bold')
        ax3.set_ylabel('Average Water (L)', color='white')
        ax3.set_title('Crypto Mining Impact', color='white')
        ax3.tick_params(colors='white')
        for spine in ax3.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig3)
        st.caption("Crypto mining doubles water consumption")

    with col_d:
        risk_groups = df.groupby('Risk Level')['Water Used (L)'].mean()
        colors_risk = ['#ff4444', '#00d4aa', '#ff9900', '#ffd93d']
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        fig4.patch.set_facecolor('#0d1b2a')
        ax4.set_facecolor('#0d1b2a')
        ax4.pie(risk_groups, labels=risk_groups.index,
                colors=colors_risk, autopct='%1.1f%%',
                textprops={'color': 'white'})
        ax4.set_title('Water by Risk Level', color='white')
        st.pyplot(fig4)
        st.caption("Critical risk centres waste most water")

# TAB 3 - XAI Explanations
with tab3:
    st.subheader("Why Is Water Being Wasted?")
    st.markdown("*EcoXAI uses SHAP to explain exactly which factor contributes most*")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    st.markdown("#### Factor Impact on Single Reading:")
    fig5, ax5 = plt.subplots(figsize=(9, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig5)
    st.caption("Temperature is the biggest factor in water waste")

    st.markdown("#### Overall Factor Importance:")
    fig6, ax6 = plt.subplots(figsize=(9, 4))
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig6)
    st.caption("High temperature and more servers consistently waste most water")

# TAB 4 - Live Prediction
with tab4:
    st.subheader("Real Time Water Usage Prediction")
    st.markdown("*Adjust the sliders on the left and run prediction*")

    new_data = pd.DataFrame({
        'Number of Servers': [servers],
        'Temperature (C)': [temp],
        'Cooling Hours': [cooling],
        'Crypto Mining': [crypto]
    })

    prediction = model.predict(new_data)[0]

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Current Conditions:**")
        st.write(f"🖥️ Servers: **{servers}**")
        st.write(f"🌡️ Temperature: **{temp}°C**")
        st.write(f"⏰ Cooling Hours: **{cooling} hrs**")
        st.write(f"🪙 Crypto Mining: **{'Active' if crypto==1 else 'Inactive'}**")

    with col_p2:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-value'>{prediction:.0f} L</div>
            <div class='metric-label'>Predicted Water Usage</div>
        </div>""", unsafe_allow_html=True)

        if prediction > 1500:
            st.error("🚨 CRITICAL — Immediate action needed!")
            st.write("**Reduce temperature first!**")
        elif prediction > 1000:
            st.warning("⚠️ HIGH RISK — Monitor closely")
            st.write("**Consider reducing server load**")
        elif prediction > 700:
            st.info("ℹ️ MEDIUM RISK — Acceptable range")
        else:
            st.success("✅ LOW RISK — Efficient operation")

    st.markdown("---")
    st.markdown("**EcoXAI Recommendation:**")
    st.write("1️⃣ Reduce temperature to below 30°C — saves most water")
    st.write("2️⃣ Reduce server count during low demand hours")
    st.write("3️⃣ Schedule crypto mining during cooler night hours")

# Footer
st.markdown("""
<div class='footer-box'>
    <p style='color: #00d4aa; font-size: 1rem; font-weight: bold;'>
        EcoXAI Framework
    </p>
    <p style='color: #aaa; font-size: 0.85rem;'>
        Developed by Sri Bharathi | 2nd Year BSc Computer Science with Cognitive Systems
    </p>
    <p style='color: #666; font-size: 0.8rem;'>
        IoT · Explainable AI · Water Intelligence · Sustainable Computing
    </p>
</div>
""", unsafe_allo

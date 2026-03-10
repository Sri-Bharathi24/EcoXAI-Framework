import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="EcoXAI Dashboard", layout="wide")

# ----------------------------
# DARK STYLE
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("EcoXAI Framework")
st.subheader("IoT-Driven Water Footprint Intelligence for AI Data Centres")

# ----------------------------
# DATA
# ----------------------------

data = {
    'Servers':[100,200,150,300,250,180,220,270,130,190],
    'Temperature':[25,35,30,40,38,28,33,39,26,31],
    'Cooling Hours':[2,5,3,7,6,3,4,6,2,4],
    'Crypto Mining':[0,1,0,1,1,0,1,1,0,0],
    'Water Used':[500,1200,800,2000,1700,650,1100,1850,520,900]
}

df = pd.DataFrame(data)

# ----------------------------
# TRAIN MODEL
# ----------------------------

X = df[['Servers','Temperature','Cooling Hours','Crypto Mining']]
y = df['Water Used']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train)

# ----------------------------
# TABS
# ----------------------------

tab1,tab2,tab3,tab4 = st.tabs([
    "IoT Data",
    "Analytics",
    "AI Prediction",
    "EcoXAI Insights"
])

# ----------------------------
# TAB 1
# ----------------------------

with tab1:

    st.header("IoT Sensor Data")
    st.dataframe(df)

# ----------------------------
# TAB 2
# ----------------------------

with tab2:

    st.header("Water Usage Analytics")

    fig,ax = plt.subplots()
    ax.scatter(df['Temperature'],df['Water Used'])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Water Used")
    st.pyplot(fig)

# ----------------------------
# TAB 3
# ----------------------------

with tab3:

    st.header("Live Prediction")

    servers = st.slider("Servers",50,500,200)
    temp = st.slider("Temperature",20,45,30)
    cool = st.slider("Cooling Hours",1,10,4)
    crypto = st.selectbox("Crypto Mining",[0,1])

    input_data = pd.DataFrame({
        'Servers':[servers],
        'Temperature':[temp],
        'Cooling Hours':[cool],
        'Crypto Mining':[crypto]
    })

    prediction = model.predict(input_data)

    st.metric("Predicted Water Usage (Litres)",round(prediction[0],2))

# ----------------------------
# TAB 4
# ----------------------------

with tab4:

    st.header("EcoXAI Explanation")

    st.write("""
EcoXAI analyses water consumption patterns in AI data centres.

Key drivers of water waste:
• High temperatures  
• Increased server load  
• Extended cooling hours  
• Crypto mining activity
""")

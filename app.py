import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EcoXAI", layout="wide")

st.title("EcoXAI Framework")
st.subheader("IoT-Driven Water Footprint Intelligence for AI Data Centres")
st.write("Saving Water. Explaining Why. Intelligently.")

# -------------------------------
# DATA (Simulated IoT Sensor Data)
# -------------------------------

data = {
    'Number of Servers':[100,200,150,300,250,180,220,270,130,190],
    'Temperature':[25,35,30,40,38,28,33,39,26,31],
    'Cooling Hours':[2,5,3,7,6,3,4,6,2,4],
    'Crypto Mining':[0,1,0,1,1,0,1,1,0,0],
    'Water Used':[500,1200,800,2000,1700,650,1100,1850,520,900]
}

df = pd.DataFrame(data)

st.header("IoT Sensor Data")
st.dataframe(df)

# -------------------------------
# TRAIN MODEL
# -------------------------------

X = df[['Number of Servers','Temperature','Cooling Hours','Crypto Mining']]
y = df['Water Used']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

st.success("AI Model trained successfully")

# -------------------------------
# GRAPH 1
# -------------------------------

st.subheader("Servers vs Water Usage")

fig1,ax1 = plt.subplots()
ax1.bar(df['Number of Servers'],df['Water Used'])
ax1.set_xlabel("Servers")
ax1.set_ylabel("Water Used (Litres)")
st.pyplot(fig1)

# -------------------------------
# GRAPH 2
# -------------------------------

st.subheader("Temperature vs Water Usage")

fig2,ax2 = plt.subplots()
ax2.scatter(df['Temperature'],df['Water Used'])
ax2.set_xlabel("Temperature")
ax2.set_ylabel("Water Used (Litres)")
st.pyplot(fig2)

# -------------------------------
# LIVE PREDICTION
# -------------------------------

st.header("EcoXAI Live Prediction")

servers = st.slider("Number of Servers",50,500,200)
temp = st.slider("Temperature",20,45,30)
cool = st.slider("Cooling Hours",1,10,4)
crypto = st.selectbox("Crypto Mining",[0,1])

input_data = pd.DataFrame({
    'Number of Servers':[servers],
    'Temperature':[temp],
    'Cooling Hours':[cool],
    'Crypto Mining':[crypto]
})

prediction = model.predict(input_data)

st.metric("Predicted Water Usage (Litres)", round(prediction[0],2))
# EcoXAI: IoT-Driven Water Footprint Intelligence for AI Data Centres and Crypto Mining

Developed by Sri Bharathi | 2nd Year BSc Computer Science with Cognitive Systems

Live Application: [Open EcoXAI Live App](https://ecoxai-framework-djnpurvfde3jx8zaqmtiaq.streamlit.app/)

https://github.com/Sri-Bharathi24/EcoXAI-Framework/blob/main/%5BEcoXAI%20App%5D(streamlit_screenshot.png).png

---

## About EcoXAI

EcoXAI is an original IoT-Driven Explainable AI Framework developed to monitor and intelligently explain water footprint in AI Data Centres and Cryptocurrency Mining Infrastructure.

Unlike traditional monitoring systems that only show numbers, EcoXAI explains exactly which factor is responsible for water waste and recommends what to fix first.

---

## Problem Statement

Data centres and cryptocurrency mining facilities consume millions of litres of fresh water daily for server cooling. The challenge is not just measuring water usage but understanding which operational factor contributes most to water waste.

Existing systems monitor water consumption but fail to explain the root cause. This gap leads to inefficient interventions and continued water depletion near communities and agricultural zones.

---

## Solution

EcoXAI addresses this gap by combining IoT sensor data, Machine Learning and Explainable AI to deliver intelligent water footprint analysis.

The framework monitors four key factors in real time including number of active servers, data centre temperature, cooling duration and cryptocurrency mining activity. It then uses SHAP based XAI to explain exactly which factor is driving water consumption and provides actionable recommendations.

---

## Key Findings

Temperature is the primary driver of water waste in data centres. When temperature exceeds 35 degrees Celsius, water consumption increases exponentially.

Cryptocurrency mining when active increases average water consumption by 133 percent compared to normal operations.

Critical risk data centres consume four times more water than low risk facilities.

Among Indian cities monitored, Hyderabad data centres recorded the highest average water consumption followed by Mumbai and Chennai.

---

## Technology Stack

Python was used for data processing and machine learning. Google Colab served as the development environment. Random Forest algorithm was used for water usage prediction. SHAP library was used for explainability and XAI analysis. Streamlit was used for live web application deployment. Power BI was used for interactive dashboard visualization. IoT simulation was used for sensor data generation.

---

## Project Structure

The repository contains the following files. The main application file app.py contains the Streamlit web application. The file ecoxai.py contains the core EcoXAI framework and analysis code. The file requirements.txt lists all required Python libraries. The file EcoXAI Results CSV contains the processed IoT sensor dataset. The graphs folder contains all visual output screenshots. The dashboard screenshot shows the Power BI intelligence dashboard.

---

## How to Run Locally

Install the required libraries using the requirements.txt file. Then run the application using the Streamlit run command on app.py. The application will open in your browser automatically.

---

## Developer

Sri Bharathi
2nd Year BSc Computer Science with Cognitive Systems
Presented at Tech Titans 2K26, Vellalar College for Women

---

## Research Context

This project is part of the EcoXAI research initiative exploring explainable AI frameworks for environmental sustainability. It extends the EdupathXAI framework concept to environmental domain applications.

Future scope includes expanding EcoXAI to cover quantum computing infrastructure, scientific simulation clusters and large scale AI model training facilities which represent even greater water footprint challenges.

---

Saving Water. Explaining Why. Intelligently.

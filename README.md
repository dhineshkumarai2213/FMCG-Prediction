# COGNIFORE AI FOR FMCG DEMAND  FORECASTING &amp; SUPPLY CHAIN  OPTIMIZATION SYSTEM
Here’s your **README.md** prepared in the same clean format as your previous project:

---

# 📦 FMCG AI — Demand Forecasting & Supply Chain Optimization System

A comprehensive AI-powered supply chain system that predicts demand, optimizes inventory, and improves distribution efficiency using machine learning and analytics.

---

## 🎯 Project Overview

An end-to-end intelligent system that:

* Predicts future product demand using a **Random Forest ML model**
* Optimizes inventory with **EOQ (Economic Order Quantity)** and **Reorder Point**
* Plans warehouse-to-retailer dispatch using a **priority-based greedy algorithm**
* Detects anomalies using **Z-Score and IQR methods**
* Provides a **real-time analytics dashboard** for monitoring and decision-making 

---

## 🚀 Features

* 🔐 User Authentication System (Admin & Analyst roles)
* 📈 Demand Forecasting with Machine Learning
* 📦 Inventory Optimization (EOQ, Safety Stock, ROP)
* 🚚 Supply Chain Dispatch Optimization
* ⚠️ Anomaly Detection & Risk Analysis
* 📊 Interactive Dashboard with Charts & KPIs
* 📉 Model Performance Metrics (R², MAE, RMSE)

---

## 🛠 Tech Stack

**Frontend:**

* HTML5, CSS3, JavaScript
* Chart.js

**Backend:**

* Python (Flask)
* Flask-CORS

**Machine Learning:**

* Scikit-learn
* Pandas, NumPy
* Joblib

**Database:**

* SQLite

---

## 📂 Project Structure

```
fmcg_ai_project/
├── backend/
│   ├── app.py
│   ├── train_model.py
│   ├── forecast_model.py
│   ├── inventory_logic.py
│   ├── supply_chain.py
│   ├── anomaly_detection.py
│   ├── database.py
│   ├── model/
│   │   └── demand_model.pkl
│   ├── data/
│   │   └── sales_data.csv
│   └── requirements.txt
│
├── frontend/
│   ├── templates/
│   │   ├── login.html
│   │   └── dashboard.html
│   └── static/
│       ├── css/style.css
│       └── js/dashboard.js
│
├── database/
│   └── fmcg.db
│
└── README.md
```

---

## ⚙️ Getting Started

### 📌 Prerequisites

* Python 3.10+
* pip package manager
* Basic understanding of Flask

---

### 🔧 Installation

Clone the repository:

```bash
git clone <your-repo-link>
cd fmcg_ai_project
```

Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

---

### 🧠 Train the Model (Run Once)

```bash
python train_model.py
```

This will:

* Generate dataset
* Train Random Forest model
* Save model file (`.pkl`)

### ▶️ Run the Application

```bash
python app.py
```

Open browser:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 🔐 Login Credentials

| Role    | Username | Password   |
| ------- | -------- | ---------- |
| Admin   | admin    | admin123   |
| Analyst | analyst  | analyst123 |

## 📊 System Modules

### 1. Demand Forecasting

* Predicts future demand based on historical data
* Uses lag features and rolling averages

### 2. Inventory Optimization

* EOQ calculation
* Reorder Point
* Safety Stock estimation

### 3. Supply Chain Dispatch

* Priority-based allocation
* Considers urgency, distance, and stock risk

### 4. Anomaly Detection

* Z-Score method
* IQR method
* Highlights unusual demand spikes

### 5. Dashboard

* KPI metrics
* Charts for demand and anomalies
* Risk indicators

## 🔌 API Endpoints

| Method | Endpoint          | Description               |
| ------ | ----------------- | ------------------------- |
| POST   | /api/login        | User authentication       |
| POST   | /api/logout       | Logout                    |
| POST   | /api/forecast     | Demand prediction         |
| POST   | /api/inventory    | Inventory optimization    |
| GET    | /api/supply-chain | Dispatch planning         |
| GET    | /api/anomalies    | Detect anomalies          |
| GET    | /api/history      | Demand history            |
| GET    | /api/metrics      | Model performance metrics |

## 🧠 Machine Learning Details

* Model: **Random Forest Regressor**
* Handles non-linear patterns and seasonality
* Accuracy: ~92%–96% (R² score)
* Uses lag features and rolling trends

## ⚠️ Limitations

* Uses synthetic dataset
* Single-product forecasting
* Not deployed on cloud

---

## 🔮 Future Enhancements

* Real-world dataset integration
* Multi-product forecasting
* LSTM / Prophet models
* Cloud deployment (AWS/GCP)
* Alert system (Email/SMS)

---

## 🎓 Project Type

Final Year Engineering Project
Domain: **AI & Data Science in Supply Chain Optimization**




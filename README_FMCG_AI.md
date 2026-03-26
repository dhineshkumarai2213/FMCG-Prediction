# 📦 FMCG AI — Demand Forecasting & Supply Chain Optimization System  

A comprehensive AI-powered supply chain system that predicts demand, optimizes inventory, and improves distribution efficiency using machine learning and analytics.

---

## 🎯 Project Overview  

An end-to-end intelligent system that:  
- Predicts future product demand using a Random Forest ML model  
- Optimizes inventory with EOQ (Economic Order Quantity) and Reorder Point  
- Plans warehouse-to-retailer dispatch using a priority-based greedy algorithm  
- Detects anomalies using Z-Score and IQR methods  
- Provides a real-time analytics dashboard for monitoring and decision-making  

---

## 🚀 Features  

- User Authentication System (Admin & Analyst roles)  
- Demand Forecasting with Machine Learning  
- Inventory Optimization (EOQ, Safety Stock, ROP)  
- Supply Chain Dispatch Optimization  
- Anomaly Detection & Risk Analysis  
- Interactive Dashboard with Charts & KPIs  
- Model Performance Metrics (R², MAE, RMSE)  

---

## 🛠 Tech Stack  

Frontend: HTML5, CSS3, JavaScript, Chart.js  
Backend: Python (Flask), Flask-CORS  
Machine Learning: Scikit-learn, Pandas, NumPy, Joblib  
Database: SQLite  

---

## 📂 Project Structure  

```
fmcg_ai_project/
├── backend/
├── frontend/
├── database/
└── README.md
```

---

## ⚙️ Getting Started  

### Prerequisites  
- Python 3.10+  
- pip  

### Installation  
```bash
git clone <your-repo-link>
cd fmcg_ai_project
cd backend
pip install -r requirements.txt
```

### Train Model  
```bash
python train_model.py
```

### Run Application  
```bash
python app.py
```

Open: http://127.0.0.1:5000  

---

## 🔐 Login Credentials  

Admin → admin / admin123  
Analyst → analyst / analyst123  

---

## 📊 Modules  

- Demand Forecasting  
- Inventory Optimization  
- Supply Chain Dispatch  
- Anomaly Detection  
- Dashboard  

---

## 🔌 API Endpoints  

POST /api/login  
POST /api/forecast  
POST /api/inventory  
GET /api/supply-chain  
GET /api/anomalies  

---

## ⚠️ Limitations  

- Synthetic dataset  
- Single-product forecasting  

---

## 🔮 Future Enhancements  

- Real-world dataset  
- Multi-product forecasting  
- Cloud deployment  

---

## 🎓 Project Type  

Final Year Engineering Project  
AI & Data Science Application

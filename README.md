# Retail Sales Forecasting Dashboard 📊

## 🔍 Project Overview

This project predicts **weekly retail sales** for stores or products using historical data and external factors (holidays, temperature, fuel price, CPI, unemployment). It features an **interactive Streamlit dashboard** where users can upload their own sales datasets, explore KPIs, visualize trends, and download forecasted sales.

---

## ✨ Key Features

* 📂 **CSV Upload** – Works with any sales dataset (not limited to Walmart).
* ⏳ **Future Forecasting** – Predicts multiple weeks ahead using lag features & external factors.
* 📊 **Interactive Dashboard** – Historical + forecast charts, collapsible tables, and KPIs.
* 💡 **Feature Importance** – Explains which factors most influence sales.
* 📥 **Downloadable Reports** – Export predicted sales as CSV for sharing or reporting.

---

## 🛠 Tech Stack

* **Python** – Core programming language
* **Pandas** – Data cleaning & preprocessing
* **XGBoost** – Predictive modeling (regressor)
* **Plotly** – Interactive visualizations
* **Streamlit** – Dashboard & deployment
* **Joblib** – Model saving/loading

---

## 📂 Project Structure

```
retail-forecasting/
│
├── streamlit_app.py              # Main Streamlit app
├── requirements.txt              # Project dependencies
├── Retail_Sales_Forecasting_Project.pdf   # Documentation
└── README.md                     # Project description
```

---

## 🚀 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/retail-forecasting.git
cd retail-forecasting
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## 📖 Example Dataset Format

Your CSV should include:

* **Date** (weekly)
* **Sales** (numeric sales values)
* **Store/Product** identifier
* *(Optional)* Holiday flag, Temperature, CPI, Unemployment, Fuel price

---

## 🎯 Learning Outcomes

* Hands-on **data analytics workflow**: data cleaning → feature engineering → modeling → dashboarding.
* Applied **time series forecasting** with machine learning (XGBoost).
* Built a **professional BI-style dashboard** with KPIs and charts.
* Gained skills in **data visualization, business insight, and reporting**.

---

## 📜 Documentation

A detailed PDF explaining the methodology, tools, workflow, and how to showcase this project in CVs and scholarships is included:
📄 `Retail_Sales_Forecasting_Project.pdf`

---

## 🌐 Deployment (Optional)

You can deploy the project online using [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub.
2. Connect Streamlit Cloud to your GitHub.
3. Deploy `streamlit_app.py`.

Your app will be live at:

```
https://<your-username>-retail-forecasting.stre
```

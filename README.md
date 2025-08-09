# 📊 Cryptocurrency Volatility Prediction – Project Documentation

This repository contains the complete documentation for the **Cryptocurrency Volatility Prediction** project.  
The project focuses on predicting **Bitcoin's daily volatility** using machine learning techniques based on historical OHLCV data.

---

## 📂 Contents
- **HLD_Document.pdf** – High-Level Design with architecture overview, project scope, and components.
- **LLD_Document.pdf** – Low-Level Design with detailed module breakdown and data flow.
- **Pipeline_Architecture.pdf** – System flow diagram with stage-by-stage explanations.
- **Final_Report.pdf** – Complete academic-style report with EDA, model details, and conclusions.

---

## 🛠 Tech Stack
- **Programming Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, Streamlit
- **Model Used:** Random Forest Regressor

---

## 📜 Dataset Overview
- Columns: `open`, `high`, `low`, `close`, `volume`, `marketCap`, `timestamp`, `crypto_name`, `date`
- Data Source: Historical cryptocurrency market data
- Focus: **Bitcoin** for analysis and prediction

---

## 📈 Project Pipeline
1. **Data Ingestion** – Load historical Bitcoin OHLCV data from CSV.
2. **Preprocessing** – Clean missing values, handle invalid data, filter for Bitcoin.
3. **Feature Engineering** – Create `volatility`, `liquidity_ratio`, and `rolling_volatility`.
4. **EDA** – Generate correlation heatmaps, volatility trends, and distributions.
5. **Model Training** – Train Random Forest Regressor.
6. **Evaluation** – Measure RMSE, MAE, and R² score.
7. **Deployment** – Serve predictions via Streamlit web app.

---

## 📄 How to Use
1. **Download** the PDFs in this repository.
2. Review `HLD_Document.pdf` and `LLD_Document.pdf` for design details.
3. Open `Final_Report.pdf` for the complete project explanation.
4. Use `Pipeline_Architecture.pdf` to understand the flow.


---

> This repository contains documentation only. The full code, dataset, and deployment files are available in the development repository.

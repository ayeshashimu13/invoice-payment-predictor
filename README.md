# 📄 Invoice Payment Predictor

Predicting **when vendor invoices will get paid** — and **whether they'll be late** — to help finance teams chase the risky ones early instead of guessing.

---

## The problem

Finance teams receive hundreds of invoices and don't know which ones will be paid late. So they chase everyone equally, wasting time. This project uses past invoices to predict payment behavior automatically.

## What it does

Given an invoice's details (vendor, category, region, amount, payment terms), the system predicts:

- **How many days** it will take to get paid (regression)
- **Whether it will be late** — yes / no (classification)

A simple **Streamlit web app** lets anyone enter an invoice and get both predictions instantly.

## How it was built

1. **Data cleaning** — handled real-world mess: duplicate rows, amounts stored as text (`"$10,161.58"`), mixed date formats, inconsistent vendor names, and missing values.
2. **Feature engineering** — built the prediction target (`days_to_pay`) from the due and payment dates; turned text columns into numbers using one-hot encoding.
3. **Modeling** — trained and compared several models:

| Model | Avg error (days off) |
|---|---|
| Baseline (guess the average) | 11.73 |
| **Linear Regression** | **10.69** ✅ |
| Random Forest | 11.83 |
| XGBoost | 10.91 |

   The classification model catches **~75% of late invoices** (recall).
4. **Storytelling** — charts and plain-language narrative so non-technical readers can follow.
5. **Deployment** — saved the trained models and served them through a Streamlit app.

## Results

- Predicts payment timing within **~11 days** on average.
- Flags late invoices with **75% recall**, helping finance act early.
- The simplest model won — a useful reminder that more complex isn't always better.

## ⚠️ Limitations (honest)

The biggest reasons invoices are paid late — **disputes, customer cash-flow problems, phone calls** — are *not recorded in the data*. So no model can predict perfectly; there's a natural accuracy ceiling. This tool narrows the guess; it doesn't replace human judgment.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · Streamlit

## How to run

​```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
​```

# Hull Tactical - Market Prediction 📈

This repository contains my solutions and modeling pipeline for the [Kaggle Hull Tactical - Market Prediction Competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction).

## 📖 Overview
Wisdom from most personal finance experts and the Efficient Market Hypothesis (EMH) suggests that timing the market is impossible, as everything knowable is already priced in. This project challenges that orthodoxy using machine learning. 

The goal of this competition is to predict the daily excess returns of the S&P 500 and dynamically adjust market exposure. By leveraging a rich dataset of technical, macroeconomic, and proprietary indicators, the objective is to build a risk-aware model that outputs an optimal daily allocation (between `0` and `2`) to the S&P 500, outperforming the market while strictly managing volatility constraints.

## 🎯 The Challenge
Financial time-series forecasting is notoriously difficult due to extreme noise and non-stationary market regimes. Key challenges addressed in this repository include:
* **Low Signal-to-Noise Ratio:** Financial data is heavily dominated by noise. Relying purely on point-forecasts often leads to severe overfitting.
* **Fat Tails & Volatility Clustering:** Market returns do not follow a normal distribution. Extreme events occur more frequently than expected, requiring robust risk-management (e.g., CVaR penalties).
* **Time-Series Leakage:** Preventing look-ahead bias through strict walk-forward (Purged/Embargoed) cross-validation.
* **Rolling API Evaluation:** The test set is evaluated sequentially using a time-series API, requiring models to efficiently update and generate predictions row-by-row (Buy Today, Sell Tomorrow).

## 📊 Dataset & Features
The dataset contains decades of historical market data with 98 features categorized into 7 distinct families:
* **D (D1-D9):** Technical/Dummy indicators
* **E (E1-E20):** Macroeconomic indicators
* **V (V1-V15):** Volatility and variance signals
* **S (S1-S15):** Sentiment signals
* **M (M1-M20):** Market dynamics & momentum
* **T (T1-T10):** Trend and timing signals
* **P (P1-P9):** Proprietary Hull Tactical signals

**Target Variable:** `market_forward_excess_returns` — The S&P 500 forward return minus the rolling 5-year mean forward return, winsorized to handle extreme outliers.

## ⚖️ Evaluation Metric
Submissions are evaluated using a custom **Volatility-Constrained Sharpe Ratio**. 
This metric aggressively penalizes strategies that take on significantly more volatility than the underlying market, or fail to outperform the market's baseline return. It explicitly rewards consistent, risk-adjusted excess returns rather than volatile, lucky bets.

## 📂 Repository Structure (Notebooks)

*(Note: The exact execution order is represented by the numbering of the notebooks below)*

* **`01_EDA_and_Feature_Engineering.ipynb`**
  * Explores the distributions, missingness (especially in early periods), and stationarity of the 7 feature families.
  * Handles volatility clustering and tail-risk analysis.
  * Implements dynamic rolling-window features and lag variables.
  
* **`02_Baseline_Models.ipynb`**
  * Establishes a foundation using Ridge Regression and simple LightGBM models.
  * Sets up the Walk-Forward Validation pipeline to ensure realistic, non-leaky out-of-fold (OOF) scoring.

* **`03_Probabilistic_Forecasting_Ensemble.ipynb`**
  * The core predictive engine. Instead of a single point prediction, this trains a family of ML models (XGBoost, CatBoost, LightGBM) to characterize the *conditional distribution* of market excess returns.
  * Uses risk-aware optimization to translate predicted mean, variance, and left-tail quantiles into the final `[0, 2]` allocation sizing.

* **`04_API_Submission.ipynb`**
  * The final inference script tailored for Kaggle's `kaggle_evaluation` API.
  * Manages the stateful `is_scored` logic, updates lagged targets daily, and efficiently processes the sequential stream without timing out the 9-hour Kaggle kernel limit.

## 🛠️ Key Technologies
* **Data Processing:** `pandas`, `numpy`, `scipy`
* **Modeling:** `lightgbm`, `xgboost`, `catboost`, `scikit-learn`
* **Validation:** Custom Walk-Forward / Purged Time-Series Split
* **Environment:** Kaggle Notebooks (API integrated)

## 🚀 How to Run
1. Clone this repository: 
   ```bash
   git clone [https://github.com/sauravdas101/Hull_Market_prediction.git](https://github.com/sauravdas101/Hull_Market_prediction.git)

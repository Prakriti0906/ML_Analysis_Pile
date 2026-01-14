# 🏗️ ML Analysis of Pile Capacity: Uncertainty Quantification & Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange?style=for-the-badge)
![Uncertainty Quantification](https://img.shields.io/badge/Uncertainty-Conformal%20Prediction%20%7C%20Quantile%20Regression-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

A comprehensive machine learning framework designed to predict **Pile Bearing Capacity**. This project addresses the inherent variability in geotechnical data by moving beyond standard point predictions to implement rigorous **Uncertainty Quantification (UQ)** workflows, ensuring that foundation design decisions are backed by statistical confidence.

---

## 📑 Table of Contents (Navigation)

1. [📌 Project Overview](#-project-overview)
2. [📂 Repository Structure](#-repository-structure)
3. [📊 Dataset Details](#-dataset-details)
4. [🛠️ Workflow & Methodology](#-workflow--methodology)
    - [Phase 1: Hyperparameter Tuning](#phase-1-hyperparameter-tuning)
    - [Phase 2: Quantile Regression](#phase-2-quantile-regression)
    - [Phase 3: Probabilistic Distribution](#phase-3-probabilistic-distribution)
    - [Phase 4: Conformal Predictions](#phase-4-conformal-predictions)

---

## 📌 Project Overview

Accurate prediction of pile bearing capacity is critical for structural safety and cost optimization. This repository provides a robust pipeline for analyzing geotechnical pile data using advanced Machine Learning regressors. Key features include:

* **Advanced Optimization**: Automated hyperparameter tuning using **Optuna**.
* **Interval Estimation**: **Quantile Regression** to estimate the range of probable pile capacities.
* **Full Distribution Modeling**: Using **NGBoost** and **PGBM** to predict the entire probability distribution of the target capacity.
* **Guaranteed Coverage**: **Conformal Prediction** (using MAPIE & PUNCC) to generate prediction intervals with mathematically guaranteed validity (e.g., 90% confidence).

---

## 📂 Repository Structure

The project is organized into modular directories representing the analysis pipeline.

### 1. 🗂️ [Data](./Data)
Contains the geotechnical engineering dataset.
* **[`train.csv`](./Data/train.csv)**: Labeled data used for model training.
* **[`test.csv`](./Data/test.csv)**: Held-out data for final model evaluation.

### 2. 🎛️ [Hyperparameter Tuning](./Hyperparameter%20Tuning)
Before training UQ models, base regressors are optimized to minimize error.
* **[`Optuna_autosampler_pile.ipynb`](./Hyperparameter%20Tuning/Optuna_autosampler_pile.ipynb)**: Implements Bayesian Optimization via Optuna to find optimal parameters for XGBoost, CatBoost, LightGBM, etc.
* **`test_results.xlsx`**: Stores the optimized parameter sets and comparative scores across various pruners and samplers.

### 3. 📉 [Quantile Regression](./Quantile%20Regression)
Focuses on predicting conditional quantiles (e.g., $Q_{0.05}$ and $Q_{0.95}$) rather than just the mean.
* **[`Quantile_Regression.ipynb`](./Quantile%20Regression/Quantile_Regression.ipynb)**: Trains models to minimize Pinball Loss.
* **`Results/`**: Contains prediction files for **XGBoost, LightGBM, CatBoost, HGBM, GPBoost,** and **PGBM**.

### 4. 📊 [Probabilistic Distribution](./Probabilistic%20Distribution)
Treats the target variable as a distribution (e.g., Normal or Laplace) to capture aleatoric uncertainty.
* **[`Probabilistic__Distribution_pile capacity.ipynb`](./Probabilistic%20Distribution/Probabilistic__Distribution_pile%20capacity.ipynb)**: Implements **NGBoost** and **PGBM** to output $\mu$ (mean) and $\sigma$ (standard deviation) for pile capacity.
* **`Results/`**: Includes Matrix Evaluation metrics, Calibration curves, and PIT (Probability Integral Transform) histograms.

### 5. 🛡️ [Conformal Prediction](./Conformal%20Prediction)
Applies rigorous statistical calibration to ensure prediction intervals are valid.
* **[`Conformal Predictions(MAPIE,PUNCC)_pile.ipynb`](./Conformal%20Prediction/Conformal%20Predictions(MAPIE,PUNCC)_pile.ipynb)**: Uses Split Conformal Prediction (SCP) and Cross-Validation (CV+) techniques.
* **`Results/`**: CSVs containing lower and upper bounds tailored to a specific $\alpha$ (error rate).

---

## 📊 Dataset Details

The analysis is based on geotechnical data containing geometric and soil parameters governing pile behavior:

| Feature | Description |
| :--- | :--- |
| **F1** | Geotechnical Soil/Load Factor 1 |
| **F2** | Geotechnical Soil/Load Factor 2 |
| **Zv1** | Depth/Vertical Parameter 1 |
| **Zv2** | Depth/Vertical Parameter 2 |
| **phi** | Angle of Internal Friction ($\phi$) |
| **Th** | Pile Thickness/Diameter parameter |
| **L** | Pile Length |
| **Total** | **Target Variable**: Ultimate Pile Bearing Capacity |

---

## 🛠️ Workflow & Methodology

### Phase 1: Hyperparameter Tuning
We utilize **Optuna** with the Tree-structured Parzen Estimator (TPE) sampler.
1.  **Search Space**: Defined for Learning Rate, Max Depth, Subsample, and Regularization (L1/L2).
2.  **Objective**: Minimize RMSE (Root Mean Squared Error) on cross-validation folds.
3.  **Outcome**: Best parameters are saved to excel files and passed to the UQ modules.

### Phase 2: Quantile Regression
Standard regression predicts the conditional mean $E[Y|X]$. Quantile regression predicts $Q_\tau(Y|X)$.
* **Models**: XGBoost, CatBoost, LightGBM, GradientBoosting, HGBM, GPBoost.
* **Application**: We predict the 5th and 95th percentiles to create a 90% prediction interval.
* **Metric**: Pinball Loss (Quantile Loss).

### Phase 3: Probabilistic Distribution
This approach assumes $Y|X \sim \mathcal{D}(\theta)$.
* **NGBoost**: Uses Natural Gradients to boost the parameters of a distribution.
* **PGBM**: Probabilistic Gradient Boosting Machines that optimize Continuous Ranked Probability Score (CRPS).
* **Benefit**: Allows calculating the probability of capacity falling below a required design load.

### Phase 4: Conformal Predictions
A wrapper method that calibrates any base model to provide valid intervals.
* **Libraries Used**: `MAPIE` and `PUNCC`.
* **Guarantee**: If we set confidence to 90%, the true pile capacity is mathematically guaranteed to fall within the predicted range 90% of the time (under exchangeability assumptions).
* **Metric**: Mean Prediction Interval Width (MPIW) vs. Prediction Interval Coverage Probability (PICP).

---

*Analysis by [Danesh Selwal and Prakriti Bisht].*

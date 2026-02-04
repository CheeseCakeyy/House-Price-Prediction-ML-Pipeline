# Housing Prices – Kaggle Competition Learning

This repository contains two complete end-to-end approaches for the **Housing Prices Competition for Kaggle Learn Users**:

1. A classical machine learning pipeline using scikit-learn  
2. A deep learning pipeline using TensorFlow/Keras  

The project demonstrates how different modeling approaches affect performance on structured tabular data.

---

## Project Overview

The goal of this project is to predict the **SalePrice** of houses based on a wide variety of numerical and categorical features such as:

- location  
- size  
- quality  
- year built  
- construction details  

The competition is hosted on Kaggle and evaluates submissions using RMSE on unseen test data.

---

## Problem Statement

Given structured housing-related attributes, build a regression model capable of accurately predicting the final selling price of a house.

---

## Technologies Used

- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib  
- TensorFlow / Keras  
- Kaggle Notebooks  

---

# Iteration 1 – Classical Machine Learning Approach (Baseline)

### Overview

The first iteration focused on building a robust traditional machine learning pipeline using scikit-learn.  
All preprocessing and modeling steps were implemented using **Pipelines** to ensure reproducibility and prevent data leakage.

---

### Pipeline Steps

- Handling missing values using `SimpleImputer`
  - Numerical → median imputation  
  - Categorical → most-frequent / constant “missing” strategy  
- Feature scaling using `StandardScaler`
- Categorical encoding using `OneHotEncoder`
- Feature selection using Lasso (`SelectFromModel`)
- Final model: `RandomForestRegressor`

---

### Model Performance (Iteration 1)

- Validation **R² score**: ~0.84  
- Validation **MAE**: ~17,200  
- Validation **RMSE**: ~26,308  
- **Kaggle Leaderboard Score(Test RMSE Score)**: ~16963.88  
- Rank: **1963 / 4647 (~Top 42%)**

This served as a strong baseline for further experimentation.

---

# Iteration 2 – Deep Learning Approach

### Motivation

After establishing a solid classical ML baseline, a second iteration was implemented using a **neural network** to explore whether deep learning could capture more complex relationships in the data.

---

### Approach

- Reused the **same feature engineering and preprocessing pipeline** from Iteration 1  
- All categorical and numerical transformations handled using `ColumnTransformer`
- Output from sklearn preprocessing fed directly into a Keras neural network

---

### Neural Network Architecture

A simple fully connected network was designed for tabular regression:

Input (79 features)
→ Dense(160, relu)
→ Dense(80, relu)
→ Dense(40, relu)
→ Output(1)




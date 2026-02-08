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

 ```text
- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib  
- TensorFlow / Keras  
- Kaggle Notebooks  
 ```
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
 
 ```text
- Validation R² score: ~0.84  
- Validation MAE: ~17,200  
- Validation RMSE: ~26,308  
- Kaggle Leaderboard Score(Test RMSE Score): ~16963.88  
- Rank: 1840 / 4647 (~Top 39%)
 ```
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

```text
Input (79 features)
→ Dense(160, relu)
→ Dense(80, relu)
→ Dense(40, relu)
→ Output(1)
```

- Activation: ReLU for hidden layers  
- Optimizer: Adam  
- Loss: Mean Squared Error  
- Metric: Root Mean Squared Error  

---

### Training Setup

- Data preprocessing handled with scikit-learn  
- Training performed on Kaggle Notebooks using GPU accelerator  
- Batch size tuning and early experimentation with learning parameters

---

### Results – Iteration 2

The deep learning model significantly outperformed the classical ML baseline:

```text
- Kaggle Leaderboard Score: ~15403.56 RMSE
- Improvement from baseline: ~16963.88  → 15403.56 RMSE
- Rank: 1840 → 400
```
This demonstrated that, for this dataset, a properly designed neural network combined with strong preprocessing can greatly outperform tree-based models.

---

## Key Learnings

Through both iterations, the following important lessons were learned:

- Proper preprocessing is critical for tabular deep learning  
- Neural networks can be integrated smoothly with sklearn pipelines  
- Validation metrics can sometimes appear misleading compared to leaderboard results  
- Feature engineering remains valuable even for deep learning models  
- Pipelines prevent data leakage and ensure reproducible results  
- GPU acceleration is not always necessary for small tabular networks  

---

# Iteration 3 – Improved Deep Learning Approach

### Objective

After achieving strong results with the first deep learning model in Iteration 2, this iteration focused on further improving performance through better architecture design, stronger regularization, and more stable training techniques.

The goal was to extract more predictive power from the existing features without changing the core dataset or preprocessing pipeline.

---

### Key Improvements Over Iteration 2

The following enhancements were introduced compared to the previous deep learning model:

- More refined and deeper neural network architecture  
- Introduction of Dropout layers to reduce overfitting  
- Better hyperparameter tuning:
  - number of layers  
  - neurons per layer  
  - learning rate and optimizer settings  
- Improved training strategy and validation monitoring  

All preprocessing steps from earlier iterations were reused to ensure a fair and consistent comparison.

---

### Updated Neural Network Design

The model architecture was redesigned to improve generalization and convergence:

```text
Input Features
→ Dense(160, relu)
→ Dropout layers
→ Dense(80, relu)
→ Dropout layers
→ Dense(40, relu)
→ Output layer (1 neuron)
```

This structure allowed the model to capture more complex relationships in the data while preventing overfitting.

---

### Results – Iteration 3

This iteration produced the best results of the entire project:

```text
- Kaggle Leaderboard Score: ~14605 RMSE  
- Previous DL Score: ~15403 RMSE  
- Improvement: ~800 RMSE reduction  
- Rank: 400 → 242 out of 4356 participants  
```

The improvements clearly demonstrate the impact of better architecture design and regularization on tabular deep learning performance.

---

### Key Takeaways from Iteration 3

- Careful neural network design significantly improves tabular regression performance  
- Regularization techniques like dropout and batch normalization are highly effective  
- Deep learning models require structured experimentation and tuning  
- Large performance gains are possible even without adding new features  

This iteration confirmed that a well-designed deep learning model can substantially outperform both classical machine learning methods and earlier neural network versions.

---

## Overall Progress Across Iterations


| Approach | Model Type | RMSE Score | Rank |
|--------|------------|-----------|------|
| Iteration 1 | Random Forest (Classical ML) | ~16963 | 1840 |
| Iteration 2 | Basic Neural Network | ~15403 | 400 |
| Iteration 3 | Improved Neural Network | ~14605 | 242 |


## Repository Structure

```text
house-price-prediction-ml-pipeline/
├── house_price_prediction_classical_ml_approach.py
├── housing-prices-competition-deep-learning-approach.ipynb
├── housing-prices-competition-deep-learning-approach-version(2).ipynb
├── submission.csv
├── submission_DL(1).csv
├── submission_DL(2).csv
└── README.md
 ```
---

## Kaggle Competition

Housing Prices Competition for Kaggle Learn Users  
https://www.kaggle.com/competitions/home-data-for-ml-course
---

## Final Notes

This project was built as a practical learning exercise to:

- understand end-to-end ML workflows  
- compare classical ML vs deep learning  
- gain hands-on experience with Kaggle competitions  
- strengthen debugging and model evaluation skills  

The transition from a Random Forest baseline to a deep learning solution provided valuable insights into how neural networks behave on structured tabular data.

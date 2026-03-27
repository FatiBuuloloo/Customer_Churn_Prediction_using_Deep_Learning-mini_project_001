# Customer Churn Prediction using Deep Learning

## Project Overview
This repository contains a Deep Learning pipeline designed to predict bank customer churn. The project focuses on robust data preprocessing, handling overlapping/ambiguous data points, feature engineering, and addressing class imbalance using an Artificial Neural Network (ANN).

## Data Pipeline & Preprocessing
To ensure data quality and model reliability, the following steps were implemented:
1. **Feature Selection:** Removed irrelevant columns (e.g., Surname, geography, specific tf-idf features) to reduce noise.
2. **Feature Engineering:** Created new derived features to capture customer behavior:
   - `Prod_per_Tenure`
   - `Balance_per_product`
   - `Age_non_active`
   - `Credit_per_tenure`
3. **Data Ambiguity Filtering:** Identified and removed profiles with identical key features but conflicting churn labels, ensuring the model learns from consistent data boundaries.
4. **Scaling:** Applied `StandardScaler` to continuous numerical features.
5. **Imbalance Handling:** Utilized **Tomek Links** for undersampling the majority class near the decision boundary, combined with **Class Weights** (1:5 ratio) during model training.

## Model Architecture & Training
The classification is performed using a Sequential Keras Neural Network:
* **Architecture:** - Input layer matching the engineered features.
  - Three hidden Dense layers (128, 100, 64 neurons) using `tanh` activation.
  - Dropout layers (0.3, 0.2, 0.1) and L2 Regularization (0.02) to prevent overfitting.
  - Output layer with 1 neuron (`sigmoid` activation) for binary classification.
* **Optimization Strategy:** - Optimizer: `RMSprop`.
  - Callbacks: `ReduceLROnPlateau` (factor 0.2, patience 5) and `EarlyStopping` (patience 10).

## Results & Evaluation
The model was evaluated using a default classification threshold of 0.7. The evaluation metrics focus on both overall accuracy and the model's ability to rank probabilities correctly (PR-AUC), which is crucial for imbalanced datasets.

### Classification Metrics (Threshold = 0.7)
* **Accuracy:** 86.82%
* **AUC Score (Hard Prediction):** 0.7840

```text
              precision    recall  f1-score   support

           0       0.91      0.93      0.92      8142
           1       0.69      0.64      0.67      2106

    accuracy                           0.87     10248
   macro avg       0.80      0.78      0.79     10248
weighted avg       0.86      0.87      0.87     10248

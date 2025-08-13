# Social Network Ads Purchase Prediction
This project predicts whether a user will purchase a product based on features like **Gender**, **Age**, and **Estimated Salary**, using **Logistic Regression** with **hyperparameter tuning**.

## 📌 Project Overview
The dataset contains information on social network users and whether they purchased a product after seeing an ad.  
The goal is to:
1. Train a Logistic Regression model.
2. Use Polynomial Features to capture non-linear patterns.
3. Perform **GridSearchCV** hyperparameter tuning.
4. Save the best model using `pickle`.
5. Use the saved model for future predictions.

---
## 📂 Dataset
**File:** `Social_Network_Ads.csv`  
**Features:**
- `Gender` → Male/Female
- `Age` → Integer value
- `EstimatedSalary` → Annual salary in USD
- `Purchased` → Target variable (0 = No, 1 = Yes)

---
## ⚙️ Steps Performed
1. **Data Preprocessing**
   - Encoded `Gender` into numerical format.
   - Train-test split (75% training, 25% testing).
   - Standardization using `StandardScaler`.

2. **Model Training & Hyperparameter Tuning**
   - Pipeline: `PolynomialFeatures` → `LogisticRegression`.
   - Used `GridSearchCV` with 5-fold cross-validation.
   - Parameters tuned:
     ```python
     {
         'poly__degree': [1, 2, 3],
         'clf__C': [0.01, 0.1, 1, 10],
         'clf__penalty': ['l1', 'l2'],
         'clf__solver': ['liblinear', 'saga']
     }
     ```

3. **Best Model Parameters**
  - clf__C: 0.1
  - clf__penalty: l1
  - clf__solver: liblinear
  - poly__degree: 2

4. **Performance**
- **Best CV Score:** 0.8969
- **Test Accuracy:** 0.9125
- **Classification Report:**
  ```
              precision    recall  f1-score   support

         0     0.94         0.92    0.93        51
         1     0.87         0.90    0.88        29

  accuracy                          0.91        80
  macro avg    0.90        0.91    0.91        80
  weighted avg 0.91        0.91    0.91        80
  ```
  - **Test ROC-AUC:** 0.95605

---

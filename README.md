# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecommunications customer is likely to churn. The project combines exploratory data analysis, preprocessing, classification modeling, model comparison, and business interpretation.

## Project Overview

Customer churn is a major business problem across subscription-based industries. The objective of this project is to analyze customer behavior, identify factors associated with churn, and build a machine learning model that can help businesses identify customers at risk of leaving.

## Objectives

- Understand the characteristics of customers who churn.
- Perform data cleaning and exploratory data analysis (EDA).
- Prepare numerical and categorical variables for machine learning.
- Train and compare classification models.
- Evaluate models using metrics appropriate for an imbalanced churn problem.
- Interpret model coefficients to identify churn-risk signals.
- Translate model findings into practical retention recommendations.

## Dataset

This project uses the IBM Telco Customer Churn dataset containing **7,043 customer records and 21 columns**. `TotalCharges` is converted to numeric, rows with missing `TotalCharges` are removed, duplicate rows are removed, and `customerID` is excluded from modeling.

The raw dataset is used locally/through the notebook environment rather than committed to the public repository.

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

## Project Structure

```text
customer-churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_churn_eda.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── download_data.py
│   ├── data_preprocessing.py
│   └── train_models.py
├── models/
├── visualizations/
├── README.md
├── requirements.txt
└── .gitignore
```

## Machine Learning Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Preprocessing
      ↓
Train/Test Split
      ↓
Logistic Regression + Random Forest
      ↓
Model Evaluation
      ↓
Model Interpretation
      ↓
Business Recommendations
```

## Experimental Setup

- Target: `Churn` (`Yes = 1`, `No = 0`)
- Test size: **20%**
- Stratified train/test split
- Random state: **42**
- Numeric variables: median imputation + standardization
- Categorical variables: most-frequent imputation + one-hot encoding
- Class balancing: `class_weight="balanced"`

## Model Results

The models were evaluated on the held-out test set of **1,407 customers**.

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.73 | 0.49 | **0.80** | 0.61 | **0.8351** |
| Random Forest | **0.77** | **0.55** | 0.68 | 0.61 | 0.8282 |

### Selected Model: Logistic Regression

Random Forest achieved higher overall accuracy (77%), but Logistic Regression achieved substantially higher recall for the churn class (80% vs. 68%) and a slightly higher ROC-AUC (0.8351 vs. 0.8282).

Because the business objective is to identify customers who may leave so that retention teams can intervene, **Logistic Regression is selected as the primary model**. This prioritizes detection of actual churners over maximizing overall accuracy.

### Logistic Regression Confusion Matrix

```text
                 Predicted
              No Churn   Churn
Actual
No Churn          723      310
Churn              76      298
```

The model correctly identified **298 of 374 actual churners**, corresponding to approximately **80% churn recall**. It missed 76 churners and flagged 310 non-churners as potential churners.

## Model Interpretation

The Logistic Regression coefficients provide directional associations with predicted churn risk. Positive coefficients indicate higher predicted churn relative to the relevant reference category; negative coefficients indicate lower predicted churn. These are **associations, not causal effects**.

### Stronger positive churn-risk signals

| Feature | Coefficient |
|---|---:|
| Contract: Month-to-month | **+0.6961** |
| Internet Service: Fiber optic | **+0.6609** |
| Total Charges | **+0.6074** |
| Streaming TV: Yes | +0.2521 |
| Streaming Movies: Yes | +0.2447 |
| Payment Method: Electronic check | +0.2291 |
| Online Security: No | +0.1994 |
| Tech Support: No | +0.1769 |

### Stronger lower-churn associations

| Feature | Coefficient |
|---|---:|
| Tenure | **-1.2504** |
| Contract: Two year | **-0.7777** |
| Monthly Charges | **-0.6211** |
| Internet Service: DSL | **-0.5888** |
| Dependents: Yes | -0.2058 |
| Payment Method: Bank transfer (automatic) | -0.1812 |

Some encoded `No internet service` variables overlap with the same underlying internet-service condition and therefore should not be interpreted as independent business drivers.

The negative MonthlyCharges coefficient is also not interpreted as a causal claim; correlated predictors and the multivariable model structure can produce counterintuitive coefficient directions.

## Business Recommendations

1. **Prioritize month-to-month customers for retention campaigns.** Contract type is one of the strongest positive churn-risk signals, while two-year contracts have a strong negative coefficient.
2. **Investigate the fiber-optic customer segment.** Fiber optic has a strong positive coefficient and should be examined alongside pricing, service quality, complaints, and contract structure.
3. **Review customers without online security and technical support.** These variables are positively associated with churn and could support targeted service-bundle experiments.
4. **Investigate electronic-check customers.** Electronic check is positively associated with churn; payment experience and alternative automatic-payment options can be tested as retention interventions.
5. **Use tenure for customer lifecycle segmentation.** Newer customers should receive stronger onboarding and early-retention attention because tenure is the strongest negative coefficient in the fitted model.

These recommendations are hypotheses for business action and should be validated with controlled experiments or additional causal analysis before being treated as causal conclusions.

## Limitations

- The analysis uses a single public telecommunications dataset and may not generalize to every industry or customer population.
- Logistic Regression coefficients describe associations within this model, not causation.
- The classification threshold was not optimized for a specific financial cost of false positives versus false negatives.
- Model performance can change with new customer data and different business conditions.

## Project Status

**Portfolio-ready ML baseline** — data preparation, EDA, model development, evaluation, interpretation, and business recommendations are documented. Future improvements can include threshold optimization, cross-validation, hyperparameter tuning, calibration, and deployment as an API or dashboard.

## Author

**Hadi Shaikh**  
B.Sc. Data Science | Python | SQL | Machine Learning | Data Analytics

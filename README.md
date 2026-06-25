# Credit Card User Trends

This project analyzes credit card customer churn for Thera Bank and builds classification models to help identify customers at risk of closing their accounts. The notebook walks through data understanding, exploratory analysis, missing value treatment, class imbalance handling, model training, hyperparameter tuning, and business recommendations.

## Repository Contents

- `Credit Card Users Churn Prediction.ipynb`: Main analysis notebook.
- `BankChurners.csv`: Credit card customer dataset used by the notebook.
- `.gitignore`: Ignores local notebook checkpoints and environment artifacts.

## Dataset

The dataset contains 10,127 customer records and 21 columns. The target variable is `Attrition_Flag`, which indicates whether a customer is an existing customer or an attrited customer.

Key feature groups include:

- Customer demographics: age, gender, education, marital status, income category, and dependent count.
- Account relationship details: months on book, total relationship count, inactive months, and contact count.
- Credit card behavior: credit limit, revolving balance, open-to-buy balance, transaction amount, transaction count, and utilization ratio.

## Analysis Workflow

1. Load and inspect the data.
2. Drop unnecessary identifiers such as `CLIENTNUM`.
3. Explore univariate and multivariate patterns.
4. Treat missing or placeholder values in education, marital status, and income category.
5. Encode categorical variables and split the dataset into training and test sets.
6. Address class imbalance with SMOTE oversampling and random undersampling.
7. Train and compare logistic regression, decision tree, random forest, bagging, AdaBoost, gradient boosting, and XGBoost models.
8. Tune boosting models with grid search and randomized search.
9. Compare models using recall, precision, accuracy, and F1 score.

## Model Results

The modeling focuses on recall because the business goal is to find as many likely attriting customers as possible.

Top test results from the notebook:

| Model | Test Accuracy | Test Recall | Test Precision | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| Gradient with Grid Search | 0.97170 | 0.89549 | 0.92585 | 0.91042 |
| Gradient Boosting with Random Search | 0.97170 | 0.89549 | 0.92585 | 0.91042 |
| AdaBoost with Grid Search | 0.96611 | 0.88730 | 0.90021 | 0.89370 |
| AdaBoost with Random Search | 0.96479 | 0.88115 | 0.89770 | 0.88935 |

The notebook identifies `Total_Trans_Ct`, `Total_Revolving_Bal`, and `Total_Trans_Amt` as the most important churn indicators.

## Key Insights

- About 16% of customers in the dataset attrited.
- Customers with lower transaction counts, lower revolving balances, and lower transaction amounts are more likely to attrite.
- Customers inactive for recent months show higher attrition risk.
- Middle-aged customers, especially ages 36 to 55, are a notable attrition segment.
- Customers with postgraduate or doctorate education levels showed higher attrition in the analysis.
- Customers with fewer bank products are more likely to attrite than customers with broader product relationships.
- Frequent customer contacts may signal unresolved service issues or dissatisfaction.

## Business Recommendations

- Offer targeted rewards, cashback, or usage incentives to customers with declining transaction counts.
- Proactively re-engage customers who have been inactive recently.
- Review service quality for customers contacted multiple times in the past year.
- Create retention campaigns for middle-aged, highly educated customers who may be comparing competing bank offers.
- Encourage broader product adoption to strengthen customer relationships.

## How to Run

1. Clone this repository.
2. Create and activate a Python environment.
3. Install the required analysis libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost statsmodels jupyter
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `Credit Card Users Churn Prediction.ipynb` and run the cells from top to bottom.

## Notes

The notebook includes saved outputs and visualizations. Re-running the notebook may produce slightly different results if library versions differ from the original environment.

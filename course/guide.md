# Guide - Index

The content was split into smaller files to make study and maintenance easier.

## Guides

1. [Linear Regression](unidade_1/linear_regression.md)
2. [Logistic Regression](unidade_1/logistic_regression.md)
3. [K-Nearest Neighbors (KNN)](unidade_1/knn.md)
4. [Workflow](workflow.md)

## Recommended Reading Order

1. Start with the model workflow guide;
2. Read the model workflow details;
3. Study linear regression;
4. Continue with logistic regression;
5. Study KNN;
6. Use the implementation checklist below as a final validation.

## Implementation Guide

### Checklist

#### Simple Linear Regression

- [ ] Explore data distributions and summary statistics;
- [ ] Validate assumptions (linearity, independence, homoscedasticity, normality);
- [ ] Select feature(s) using Pearson correlation;
- [ ] Split data into train/test;
- [ ] Fit `LinearRegression()`;
- [ ] Evaluate with MSE, RMSE, and $R^2$;
- [ ] Analyze residuals and visualize fitted line.

#### Multiple Linear Regression

- [ ] Inspect feature correlations and detect multicollinearity (VIF);
- [ ] Scale features (standardization or normalization);
- [ ] Split train/test;
- [ ] Fit model and inspect coefficients;
- [ ] Evaluate on test data;
- [ ] Review residual diagnostics.

#### Regularization (Ridge/Lasso)

- [ ] Decide between Ridge (keep all features) and Lasso (feature selection);
- [ ] Tune $\alpha$ using `RidgeCV()` or `LassoCV()`;
- [ ] Compare against baseline linear regression;
- [ ] Document model choice and trade-offs.

#### Logistic Regression

- [ ] Confirm binary target and class balance;
- [ ] Use stratified train/test split;
- [ ] Scale features;
- [ ] Train baseline `LogisticRegression()`;
- [ ] Compute confusion matrix, Accuracy, Precision, Recall, F1, ROC-AUC;
- [ ] Tune regularization (`penalty`, `C`) and validate with CV.

#### KNN

- [ ] Scale features before distance-based modeling;
- [ ] Select distance metric;
- [ ] Evaluate multiple $k$ values with cross-validation;
- [ ] Train final model with best $k$;
- [ ] Report confusion matrix and classification metrics.

### General Best Practices

#### Do

1. Validate assumptions before trusting model outputs;
2. Keep train/test (and optionally validation) strictly separated;
3. Use cross-validation for hyperparameter tuning;
4. Compare models using consistent metrics;
5. Document all preprocessing and model decisions.

#### Avoid

1. Evaluating only on training data;
2. Ignoring multicollinearity in multiple regression;
3. Skipping scaling for KNN or regularized models;
4. Choosing hyperparameters without validation;
5. Interpreting a high $R^2$ as universal proof of quality.

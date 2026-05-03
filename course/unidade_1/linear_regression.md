# Guide: Linear Regression

Linear regression models continuous targets. This guide follows the same 10-step structure used in `workflow.md`.

## Linear Regression Overview

### Simple Linear Regression

Simple linear regression describes the relationship between one predictor and one continuous target.

$$y = \theta_0 + \theta_1x + \epsilon$$

- $x$ is the single input feature;
- $y$ is the continuous target;
- $\theta_0$ is the intercept;
- $\theta_1$ is the slope;
- $\epsilon$ is the error term.

### Multiple Linear Regression

Multiple linear regression extends the same idea to several predictors.

$$y = \theta_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_px_p + \epsilon$$

- $x_1, x_2, \ldots, x_p$ are the input features;
- each coefficient measures the effect of one feature while the others are held constant;
- this version is useful when the target depends on more than one factor.

### Main Difference

- Simple linear regression uses one feature;
- Multiple linear regression uses several features;
- both follow the same fitting logic, but multiple regression requires more attention to scaling and multicollinearity.

### Multiple Regression Example

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

## Step 1: Data Exploration and Quality Checks
Inspect dataset quality before modeling.
```python
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
```

## Step 2: Validate Modeling Assumptions

Validate whether linear regression is appropriate.
- Check linearity between predictors and target;
- Inspect potential multicollinearity among predictors;
- Plan residual checks for homoscedasticity and normality.

## Step 3: Feature Selection
Select stable and meaningful predictors.
```python
correlations = df.corr()["target"].drop("target").sort_values(ascending=False)
print(correlations)

selected_features = correlations.index[:3].tolist()  # example
X = df[selected_features]
y = df["target"]
```

## Step 4: Normalization or Scaling
Apply scaling when required by model design.
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

## Step 5: Data Split
Split data into train and test subsets.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Step 6: Model Fitting (Adjustment)
Train baseline and, when needed, regularized variants.
Core equation for multiple linear regression:
$$y = \theta_0 + \theta_1x_1 + \cdots + \theta_px_p + \epsilon$$
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

model = LinearRegression()
model.fit(X_train_scaled, y_train)

ridge_model = Ridge(alpha=1.5).fit(X_train_scaled, y_train)
lasso_model = Lasso(alpha=0.2).fit(X_train_scaled, y_train)
```

## Step 7: Assessment
Evaluate predictive quality with complementary regression metrics.
```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test_scaled)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
```

## Step 8: Coefficient or Feature-Importance Analysis
Interpret variable influence from coefficients.
```python
import pandas as pd

coef = pd.Series(model.coef_, index=X.columns)
print(coef.sort_values(ascending=False))
```

## Step 9: Residual or Error Analysis
Validate error behavior.
```python
import matplotlib.pyplot as plt

residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.title("Residuals vs Predicted")
plt.show()
```

## Step 10: Report
Document results and decision.
- Summarize assumptions, selected features, metrics, and diagnostics;
- Justify final model choice;
- Register limitations and next steps.
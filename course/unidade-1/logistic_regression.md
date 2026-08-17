# Guide: Logistic Regression

Logistic regression is used for binary classification and follows the same 10-step workflow.

## Step 1: Data Exploration and Quality Checks
Inspect dataset and target behavior before modeling.

```python
print(df["target"].value_counts(normalize=True))
print(df.isnull().sum())
print(df.duplicated().sum())
```

## Step 2: Validate Modeling Assumptions
Validate whether logistic modeling is appropriate.
- Confirm binary target;
- Check separability trends;
- Anticipate regularization needs.

## Step 3: Feature Selection
Select informative and leakage-safe predictors.
```python
X = df.drop(columns=["target"])
y = df["target"]
print(X.columns)
```

## Step 4: Normalization or Scaling
Scaling is recommended for stable optimization.
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

## Step 5: Data Split
Use stratified split to preserve class proportions.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## Step 6: Model Fitting (Adjustment)
Logistic regression is used when the target has two classes (0/1).
$$
\hat{p} = P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \theta_0 + \theta_1x_1 + \cdots + \theta_px_p
$$

Class prediction uses a threshold (typically 0.5):
- If $\hat{p} \ge 0.5$, predict class 1;
- Otherwise, predict class 0.

### Loss Function: Binary Cross-Entropy (Log Loss)

$$
J(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\left[y^{(i)}\log(\hat{p}^{(i)}) + (1-y^{(i)})\log(1-\hat{p}^{(i)})\right]
$$

- Build scaler + model pipeline;
- Fit the model on training data.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

pipe.fit(X_train, y_train)
```

## Step 7: Assessment
Evaluate class predictions and probability quality.
```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

## Step 8: Coefficient or Feature-Importance Analysis
Inspect coefficient direction and relative strength.
```python
import pandas as pd

coef = pd.Series(pipe.named_steps["model"].coef_[0], index=X.columns)
print(coef.sort_values(ascending=False))
```

## Step 9: Residual or Error Analysis
Analyze classification errors and threshold behavior.
```python
fp = ((y_test == 0) & (y_pred == 1)).sum()
fn = ((y_test == 1) & (y_pred == 0)).sum()
print(f"FP: {fp}, FN: {fn}")
```

## Step 10: Report
Summarize final decisions and limitations.
- Summarize metrics and class behavior;
- Document regularization choice;
- Record limitations and next iterations.

## Regularization in Logistic Regression

In `sklearn`, regularization strength is controlled by `C`:
- Smaller `C`: stronger regularization;
- Larger `C`: weaker regularization.

```python
logreg_l2 = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
logreg_l1 = LogisticRegression(penalty="l1", C=1.0, solver="liblinear", max_iter=2000)
```

## Cross-Validation

Use stratified k-fold to estimate model stability before final testing.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")

print(f"CV F1: {scores.mean():.4f} +/- {scores.std():.4f}")
```

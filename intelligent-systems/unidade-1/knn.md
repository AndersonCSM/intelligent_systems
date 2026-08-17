# Guide: K-Nearest Neighbors (KNN)

KNN is a supervised, instance-based algorithm that classifies a sample using its nearest neighbors.

## Step 1: Data Exploration and Quality Checks
Inspect dataset quality and class structure.
```python
print(df.info())
print(df.isnull().sum())
print(df["target"].value_counts(normalize=True))
```

## Step 2: Validate Modeling Assumptions
Validate whether KNN is appropriate.

## Step 3: Feature Selection
Select robust predictors for distance calculations.
```python
X = df.drop(columns=["target"])
y = df["target"]
```

## Step 4: Normalization or Scaling
KNN is highly sensitive to scale, so scaling is mandatory.
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

## Step 5: Data Split
Split with stratification to preserve class proportions.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## Step 6: Model Fitting (Adjustment)

Fit a baseline pipeline with scaler + KNN.
```python
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2))
])

knn_pipe.fit(X_train, y_train)
```

## Step 7: Assessment

Evaluate classification performance.
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = knn_pipe.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")
```

## Step 8: Coefficient or Feature-Importance Analysis
KNN does not provide direct coefficients. Use permutation importance or model-agnostic tools if interpretation is required.

## Step 9: Residual or Error Analysis
Analyze classification errors.
```python
misclassified = X_test[y_test != y_pred]
print(misclassified.head())
```

## Step 10: Report
Document decisions and trade-offs.
- Record best $k$, metric choice, and distance metric;
- Summarize strengths and limitations;
- Suggest next iterations.

## Hyperparameter Search for $k$

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

best_k = None
best_score = -1

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in [3, 5, 7, 11, 13]:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    if scores.mean() > best_score:
        best_score = scores.mean()
        best_k = k

print(f"Best k: {best_k}, Mean F1: {best_score:.4f}")
```

# Workflow Guide

This guide defines a reusable workflow for projects.

## Main Steps

1. Data exploration and quality checks;
2. Validate modeling assumptions;
3. Feature selection;
4. Normalization or scaling;
5. Data split;
6. Model fitting (adjustment);
7. Assessment;
8. Coefficient or feature-importance analysis;
9. Residual/error analysis;
10. Report.

## Step 1: Data Exploration and Quality Checks

Collect evidence about the dataset before making modeling decisions.

### What to Do

- Inspect the structure with `df.info()`;
- Summarize numeric variables with `df.describe()`;
- Check missing values with `df.isnull().sum()`;
- Check duplicates with `df.duplicated().sum()`;
- Plot histograms to inspect distributions;
- Plot boxplots to detect outliers;
- Build a correlation matrix;
- Use scatter plots to compare each predictor against the target.

### What This Step Produces

- A dataset profile;
- A data quality summary;
- Distribution and outlier evidence;
- Correlation and scatter-plot evidence.

## Step 2: Validate Modeling Assumptions

Interpret the evidence from Step 1 and decide whether the regression model is appropriate.

### What to Do

- Check whether the predictor-target relationship is approximately linear;
- Check whether predictor-predictor correlation suggests multicollinearity risk;
- Decide whether a threshold such as 0.85 indicates concern;
- Use the evidence from Step 1 plus domain knowledge to confirm model suitability;
- Decide whether transformations or regularization are needed.

### What This Step Produces

- A clear pass/fail or pass-with-warning assessment;
- A note about multicollinearity;
- A modeling decision for the next steps.

## Step 3: Feature Selection

Choose the input variables that will enter the model.

### What to Do

- Start with the correlation matrix from Step 1 and the assumption check from Step 2;
- Rank features by absolute correlation with the target;
- Keep variables that contribute meaningful signal;
- If there are many correlated features, verify whether regularization already stabilizes them;
- Compare the selected set with the final model coefficients later.

### What This Step Produces

- The final feature set;
- A short justification for each selected variable.

## Step 4: Normalization or Scaling

Place features on a common scale after the feature set has been decided.

### What to Do

- Compare scaling strategies such as `StandardScaler`, `MinMaxScaler`, and `RobustScaler`;
- Fit the scaler only on the training set;
- Transform the test set using the same fitted scaler;
- If the dataset contains outliers, prefer a robust approach;
- Keep the choice aligned with the model type and the data distribution.

### What This Step Produces

- The selected scaler;
- The scaled training and test sets;
- A justification for the scaler choice.

## Step 5: Data Split

Separate data used for learning from data used only for evaluation.

### What to Do

- Split the data into training and testing sets;
- Use a fixed random seed for reproducibility;
- Keep the split consistent with the project objective;
- If needed, test alternative split ratios to verify stability.

### What This Step Produces

- Train/test proportions;
- A reproducible random seed;
- A final split strategy.

## Step 6: Model Fitting (Adjustment)

Train the selected model with the prepared dataset.

### What to Do

- Fit the baseline regression model;
- If regularization is needed, train `RidgeCV` and/or `LassoCV`;
- Use cross-validation to tune hyperparameters;
- Compare models under the same preprocessing pipeline;
- Keep the training pipeline reproducible.

### What This Step Produces

- The trained model;
- The best hyperparameters;
- The complete training configuration.

## Step 7: Assessment

Measure how well the model predicts unseen data.

### What to Do

- Predict on the test set;
- Compute regression metrics such as `R2`, `RMSE`, and `MAE`;
- Compare multiple models if necessary;
- Decide which model is best based on both performance and interpretability.

### What This Step Produces

- A table of metrics;
- A model comparison summary;
- The final model choice.

## Step 8: Coefficient or Feature-Importance Analysis

Explain how each variable affects the prediction.

### What to Do

- Extract model coefficients;
- Compare coefficient signs and magnitudes;
- Visualize coefficients with a bar chart;
- Check whether the signs are consistent with domain knowledge;
- Use this analysis to support the final interpretation.

### What This Step Produces

- A coefficient table;
- A coefficient plot;
- A short interpretation of variable influence.

## Step 9: Residual or Error Analysis

Check whether the prediction errors are well-behaved.

### What to Do

- Compute residuals as `y_true - y_pred`;
- Plot residuals versus predicted values;
- Inspect the residual histogram;
- Build a QQ plot to assess approximate normality;
- Look for heteroscedasticity, curvature, or systematic bias.

### What This Step Produces

- Residual diagnostics plots;
- A conclusion about error behavior;
- Evidence supporting model validity.

## Step 10: Report

Summarize the full project in a concise, decision-oriented way.

### What to Do

- State the problem and the dataset used;
- Describe the preprocessing pipeline;
- Present the selected model and hyperparameters;
- Report the final metrics;
- Interpret the coefficients;
- Summarize the residual analysis;
- Mention limitations and future improvements;
- Close with the practical conclusion.

### What This Step Produces

- A concise final narrative;
- A summary table with metrics;
- A clear decision about model suitability.

## Recommended Output Per Step

1. **Data exploration:** data profile, quality summary, outlier notes;
2. **Assumptions:** checklist with pass/fail and evidence;
3. **Feature selection:** selected features and rationale;
4. **Normalization:** scaler choice and fitted parameters;
5. **Split:** train/validation/test proportions and seed;
6. **Model fitting:** model type, hyperparameters, training details;
7. **Assessment:** primary and secondary metrics;
8. **Coefficients:** interpretation and stability analysis;
9. **Residuals:** diagnostic plots and pattern checks;
10. **Report:** final conclusions, limitations, and next actions.

## Clarification About Redundancy

- **Data exploration** collects evidence only;
- **Model assumption validation** interprets that evidence and decides whether the model is appropriate;
- **Normalization/scaling** is a separate modeling-preparation step and should remain outside Step 1.

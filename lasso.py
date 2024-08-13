#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, RocCurveDisplay
import sys
import time

# Check for input score file argument
if len(sys.argv) != 2:
    sys.exit(sys.argv[0] + ": Expecting score file.")

# Load your dataset
dat = pd.read_csv(sys.argv[1])

# Prepare data for glmnet equivalent
X = dat.drop(columns=["Sex", "Phenotype", "sample"])
y = dat["Phenotype"]

# Initialize lists to store results
accuracies = []
roc_aucs = []

start = time.time()

print(start)

# Run 100 iterations
for i in range(1):
    j = i + 1
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit a logistic regression model with L1 regularization (Lasso)
    logreg_cv = LogisticRegressionCV(
        Cs=10,  # Number of regularization strengths to try
        cv=10,  # Number of folds in cross-validation
        penalty="l1",
        solver="liblinear",
        max_iter = 10000
    ).fit(X_train_scaled, y_train)

    stop = time.time() - start

    print(f"Iteration #{j} done @ {stop}")

    # Get best lambda value
    best_lambda = logreg_cv.C_[0]

    # View coefficients for the best lambda
    coefficients = logreg_cv.coef_.flatten()

    # Identify non-zero coefficients (important variables)
    important_vars = X_train.columns[coefficients != 0]

    # Make predictions on the test set
    predictions_prob = logreg_cv.predict_proba(X_test_scaled)[:, 1]
    predictions = logreg_cv.predict(X_test_scaled)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions_prob)

    # Store results
    accuracies.append(accuracy)
    roc_aucs.append(roc_auc)

    important_indices = np.where(coefficients != 0)[0]
    important_vars = X_train.columns[important_indices]
    important_coefficients = coefficients[important_indices]

    important_coeff_df = pd.DataFrame({
        "Variable": important_vars,
        "Coefficient": important_coefficients
    })

    important_coeff_df.to_csv(f"important_var{j}.csv", index=False)

stop = time.time() - start
print(f"Time elapsed: {stop} seconds.")

# Save results to a CSV file
results_df = pd.DataFrame({
    'Iteration': range(1),
    'Accuracy': accuracies,
    'ROC AUC': roc_aucs
})
results_df.to_csv("test_evaluation_results.csv", index=False)

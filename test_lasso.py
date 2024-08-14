#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, RocCurveDisplay
import sys
import time
import argparse

def lasso(file, iteration_num, random_state, prefix):
    # Load your dataset
    dat = pd.read_csv(file)

    # Prepare data for logistic regression
    X = dat.drop(columns=["Sex", "Phenotype", "sample"])
    y = dat["Phenotype"]

    # Initialize lists to store results
    accuracies = []
    roc_aucs = []
    models = []
    X_tests = []
    y_tests = []

    start = time.time()

    # Run specified number of iterations
    for i in range(iteration_num):
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

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
            max_iter=10000,
            random_state=random_state
        ).fit(X_train_scaled, y_train)

        # View coefficients for the best lambda
        coefficients = logreg_cv.coef_.flatten()

        # Make predictions on the test set
        predictions_prob = logreg_cv.predict_proba(X_test_scaled)[:, 1]
        predictions = logreg_cv.predict(X_test_scaled)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions_prob)

        # Store results
        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)

        # Store model and test data for ROC plotting
        models.append(logreg_cv)
        X_tests.append(X_test_scaled)
        y_tests.append(y_test)

    stop = time.time() - start
    print(f"Time elapsed: {stop} seconds.")

    # Save results to a CSV file
    results_df = pd.DataFrame({
        'Iteration': range(1, iteration_num + 1),
        'Accuracy': accuracies,
        'ROC AUC': roc_aucs
    })
    results_df.to_csv(f"test_evaluation_results_{prefix}.csv", index=False)

    return models, X_tests, y_tests, accuracies

def plot_roc_curves(models1, X_tests1, y_tests1, accuracies1, models2, X_tests2, y_tests2, accuracies2, label1, label2):
    plt.figure(figsize=(10, 8))

    for model, X_test, y_test, accuracy in zip(models1, X_tests1, y_tests1, accuracies1):
        RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=plt.gca(),
            name=f'{label1} (ACC = {accuracy:.2f})',
            color=(0.149, 0.616, 0.808)
            )

    for model, X_test, y_test, accuracy in zip(models2, X_tests2, y_tests2, accuracies2):
        RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=plt.gca(),
            name=f'{label2} (ACC = {accuracy:.2f})',
            color=(0.659, 0.000, 0.329)
            )

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("roc_curves_comparison.png", bbox_inches='tight')
    plt.show()

def main():
    bi_file = sys.argv[1]
    nobi_file = sys.argv[2]
    random_state = 843
    iterations = 1
    labels = ["With Bipolar", "Without Bipolar"]

    # Run Lasso for both datasets
    models1, X_tests1, y_tests1, accuracies1 = lasso(bi_file, iterations, random_state, "bi")
    models2, X_tests2, y_tests2, accuracies2 = lasso(nobi_file, iterations, random_state, "nobi")

    # Plot ROC curves
    plot_roc_curves(models1, X_tests1, y_tests1, accuracies1, models2, X_tests2, y_tests2, accuracies2, labels[0], labels[1])

if __name__=="__main__":
    main()

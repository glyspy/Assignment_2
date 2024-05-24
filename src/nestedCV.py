from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score, f1_score,
                             fbeta_score, recall_score, precision_score, average_precision_score,
                             confusion_matrix)
import numpy as np
import optuna
from optuna.integration import OptunaSearchCV
from tqdm import tqdm
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="OptunaSearchCV is experimental")
logging.getLogger('optuna').setLevel(logging.WARNING)
optuna.logging.disable_default_handler()


class NestedCV:
    def __init__(self, classifiers, seeds, param_grids, rounds=10, inner_cv_folds=3, outer_cv_folds=5, n_trials=50, scoring='roc_auc'):
        self.classifiers = classifiers
        self.param_grids = param_grids
        self.rounds = rounds
        self.inner_cv_folds = inner_cv_folds
        self.outer_cv_folds = outer_cv_folds
        self.n_trials = n_trials
        self.scoring = scoring
        self.seeds = seeds
        
    def fit(self, X, y):
        # Dictionaries to store the outer loop scores and best parameters
        all_outer_scores = {}
        all_outer_params = {}
        
        # Iterate through each classifier and its parameter grid
        for clf_name, clf in self.classifiers.items():
            print(f"\nUsing classifier: {clf_name}\n")
            param_grid = self.param_grids[clf_name]

            # List to store outer loop scores for the classifier and fold parameters
            outer_scores = []
            fold_parameters = []

            # Perform 10 rounds of nested CV
            for round_num in tqdm(range(self.rounds)):
                print(f"\nStarting round {round_num + 1}...")
                outer_seed = self.seeds[round_num * 3]
                inner_seed = self.seeds[round_num * 3 + 1]
                opt_seed = self.seeds[round_num * 3 + 2]

                # Define outer and inner cross-validation folds
                outer_cv = StratifiedKFold(n_splits=self.outer_cv_folds, shuffle=True, random_state=outer_seed)
                inner_cv = StratifiedKFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=inner_seed)

                # Iterate through each fold in the outer loop
                for train_index, test_index in tqdm(outer_cv.split(X, y)):
                    # Split data into training and test sets
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # Perform hyperparameter optimization using OptunaSearchCV
                    optuna_search = OptunaSearchCV(estimator=clf, param_distributions=param_grid,
                                                   cv=inner_cv, n_trials=self.n_trials,
                                                   random_state=opt_seed, scoring=self.scoring, n_jobs=-1)
                    
                    # Fit OptunaSearchCV on the training set
                    optuna_search.fit(X_train, y_train)

                    # Get the best estimator and parameters
                    best_estimator = optuna_search.best_estimator_

                    # Append the parameters of the current outer loop
                    fold_parameters.append(optuna_search.best_params_)

                    # Make predictions using the best estimator
                    y_pred = best_estimator.predict(X_test)

                    # Calculate performance metrics
                    # metrics = self._evaluate(y_test, y_pred)
                    # Calculate performance metrics
                    metrics = self._evaluate(y_test, y_pred, best_estimator, X_test)

                    outer_scores.append(metrics)

                # Save the parameters for the classifier for each outer loop
                all_outer_params[clf_name] = fold_parameters

            # Save the outer loop scores for the classifier
            all_outer_scores[clf_name] = outer_scores
            all_outer_params[clf_name] = fold_parameters
        
        # Return dictionaries with outer loop scores and best parameters for each classifier
        return all_outer_scores, all_outer_params


    def _evaluate(self, y_true, y_pred, best_estimator, X_test):
        # Calculate performance metrics
        mcc = matthews_corrcoef(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        f2 = fbeta_score(y_true, y_pred, beta=2)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        average_precision = average_precision_score(y_true, y_pred)
        y_scores = best_estimator.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_true, y_scores)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn)

        # Return dictionary with performance metrics
        return {
            'MCC': mcc,
            'Balanced Accuracy': bal_acc,
            'F1': f1,
            'F2': f2,
            'Recall': recall,
            'Precision': precision,
            'Average Precision': average_precision,
            'Specificity': specificity,
            'NPV': npv,
            'PR-AUC': pr_auc
        }

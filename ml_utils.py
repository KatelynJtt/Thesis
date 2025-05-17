import pandas as pd
from pathlib import Path
import numpy as np
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from itertools import cycle
import joblib

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit, GridSearchCV, RandomizedSearchCV,learning_curve, StratifiedKFold
from sklearn.svm import SVR,LinearSVC,NuSVC,SVC
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor,LinearRegression,PassiveAggressiveClassifier,RidgeClassifier,SGDClassifier,LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,BaggingRegressor,VotingRegressor,StackingRegressor
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor


from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,RandomForestClassifier,StackingClassifier,VotingClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB

from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, max_error, mean_absolute_percentage_error, classification_report, roc_curve,roc_auc_score,RocCurveDisplay,confusion_matrix, ConfusionMatrixDisplay

import shap
import lime
import lime.lime_tabular

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


param_tips = {
            'SVR': {
                'C': 'Regularization parameter. The strength of the regularization is inversely proportional to C. Higher values mean less regularization. Typical values: [0.1, 1, 10, 100].',
                'epsilon': 'Epsilon in the epsilon-SVR model. Specifies the epsilon-tube within which no penalty is associated in the training loss function. Typical values: [0.1, 0.2, 0.5, 1.0].',
                'gamma': 'Kernel coefficient for "rbf", "poly" and "sigmoid". Higher values lead to overfitting. Use "scale" or "auto" for default values. Typical values: [\'scale\', \'auto\', 1e-3, 1e-2, 1e-1, 1].',
                'kernel': 'Specifies the kernel type to be used in the algorithm. Options: ["linear", "poly", "rbf", "sigmoid"].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html'
                },
            'Ridge': {
                'alpha': 'Regularization strength; must be a positive float. Larger values specify stronger regularization. Typical values: [0.1, 1.0, 10.0, 100.0].',
                'solver': 'Solver to use in the computational routines. Options: ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html'
                },
            'Lasso': {
                'alpha': 'Constant that multiplies the L1 term, controlling regularization strength. Larger values specify stronger regularization. Typical values: [0.1, 1.0, 10.0, 100.0].',
                'max_iter': 'The maximum number of iterations for the solver to converge. Typical values: [1000, 2000, 3000].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html'
                },
            'SGD': {
                'alpha': 'Constant that multiplies the regularization term, controlling regularization strength. Typical values: [1e-6, 1e-4, 1e-2, 1.0].',
                'max_iter': 'The maximum number of passes over the training data (epochs). Typical values: [1000, 2000, 3000].',
                'learning_rate': 'The learning rate schedule. Options: ["constant", "optimal", "invscaling", "adaptive"].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html'
                },
            'ElasticNet': {
                'alpha': 'Constant that multiplies the penalty terms, controlling regularization strength. Typical values: [0.1, 1.0, 10.0, 100.0].',
                'l1_ratio': 'The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. A value of 0 corresponds to L2 penalty, a value of 1 to L1. Typical values: [0.1, 0.5, 0.7, 1.0].',
                'max_iter': 'The maximum number of iterations for the solver to converge. Typical values: [1000, 2000, 3000].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html'
                },
            'BayesianRidge': {
                'n_iter': 'Maximum number of iterations. Typical values: [100, 200, 300].',
                'alpha_1': 'Shape parameter for the Gamma distribution prior over the alpha parameter. Typical values: [1e-6, 1e-4, 1e-2].',
                'alpha_2': 'Inverse scale parameter for the Gamma distribution prior over the alpha parameter. Typical values: [1e-6, 1e-4, 1e-2].',
                'lambda_1': 'Shape parameter for the Gamma distribution prior over the lambda parameter. Typical values: [1e-6, 1e-4, 1e-2].',
                'lambda_2': 'Inverse scale parameter for the Gamma distribution prior over the lambda parameter. Typical values: [1e-6, 1e-4, 1e-2].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html'
                },
            'KNN': {
                'n_neighbors': 'Number of neighbors to use. Typical values: [3, 5, 7, 9].',
                'weights': 'Weight function used in prediction. Options: ["uniform", "distance"].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html'
                },
            'RandomForest': {
                'n_estimators': 'The number of trees in the forest. Typical values: [10, 50, 100, 200].',
                'max_features': 'The number of features to consider when looking for the best split. Options: ["auto", "sqrt", "log2"].',
                'max_depth': 'The maximum depth of the tree. Typical values: [None, 10, 20, 30].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html'
                },
            'GradientBoosting': {
                'n_estimators': 'The number of boosting stages to be run. Typical values: [100, 200, 300].',
                'learning_rate': 'Learning rate shrinks the contribution of each tree. Typical values: [0.01, 0.1, 0.2].',
                'max_depth': 'The maximum depth of the tree. Typical values: [3, 5, 7].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html'
                },
            'AdaBoost': {
                'n_estimators': 'The maximum number of estimators at which boosting is terminated. Typical values: [50, 100, 200].',
                'learning_rate': 'Learning rate shrinks the contribution of each classifier. Typical values: [0.01, 0.1, 1.0].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html'
                },
            'DecisionTree': {
                'max_depth': 'The maximum depth of the tree. Typical values: [None, 10, 20, 30].',
                'min_samples_split': 'The minimum number of samples required to split an internal node. Typical values: [2, 10, 20].',
                'min_samples_leaf': 'The minimum number of samples required to be at a leaf node. Typical values: [1, 5, 10].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html'
                },
            'LinearRegression': {
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html'
                },
            'RandomForestRegressor': {
                'n_estimators': 'The number of trees in the forest. Typical values: [10, 50, 100, 200].',
                'max_features': 'The number of features to consider when looking for the best split. Options: ["auto", "sqrt", "log2"].',
                'max_depth': 'The maximum depth of the tree. Typical values: [None, 10, 20, 30].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html'
                },
            'XGBoostRegressor': {
                'n_estimators': 'The number of boosting stages to be run. Typical values: [100, 200, 300].',
                'learning_rate': 'Learning rate shrinks the contribution of each tree. Typical values: [0.01, 0.1, 0.2].',
                'max_depth': 'The maximum depth of the tree. Typical values: [3, 5, 7].',
                'Web Link': 'https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor'
                },
            'LightGBMRegressor': {
                'n_estimators': 'The number of boosting stages to be run. Typical values: [100, 200, 300].',
                'learning_rate': 'Learning rate shrinks the contribution of each tree. Typical values: [0.01, 0.1, 0.2].',
                'num_leaves': 'The number of leaves in one tree. Typical values: [31, 50, 100].',
                'Web Link': 'https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html'
                },
            'CatBoostRegressor': {
                'iterations': 'The number of boosting stages to be run. Typical values: [100, 200, 300].',
                'learning_rate': 'Learning rate shrinks the contribution of each tree. Typical values: [0.01, 0.1, 0.2].',
                'depth': 'The maximum depth of the tree. Typical values: [3, 5, 7].',
                'Web Link': 'https://catboost.ai/docs/concepts/python-reference_catboostregressor.html'
                },
            'GaussianProcessRegressor': {
                'alpha': 'Value added to the diagonal of the kernel matrix during fitting. Typical values: [1e-10, 1e-8, 1e-6, 1e-4].',
                'optimizer': 'The optimizer to use for optimizing the kernel’s parameters. Options: ["fmin_l_bfgs_b", None].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html'
                },
            'MLPRegressor':{
                'hidden_layer_sizes': 'The number of neurons in the hidden layers. Adjust based on the complexity of your data and problem. Typical values: [(50,), (100,), (50, 50)].',
                'activation': 'The activation function for the hidden layers. Choose based on the problem. Typical choices: ["relu", "tanh", "logistic"].',
                'solver': 'The optimization algorithm to use. Choose based on problem size and characteristics. Typical choices: ["adam", "sgd", "lbfgs"].',
                'alpha': 'L2 penalty (regularization term) parameter. Helps prevent overfitting. Typical values: [0.0001, 0.001, 0.01].',
                'learning_rate_init': 'The initial learning rate used by the optimization algorithm. Experiment with different values to find optimal convergence speed. Typical values: [0.001, 0.01, 0.1].',
                'max_iter': 'The maximum number of iterations (epochs) for training. Adjust based on convergence and overfitting. Typical values: [100, 200, 300].',
                'batch_size': 'The size of mini-batches for gradient descent. Affects convergence speed and memory usage. Typical values: ["auto", 32, 64].',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html'
            },
            'AdaBoostClassifier': {
                'n_estimators': 'The maximum number of weak learners to train.',
                'learning_rate': 'Weighting factor for the weak learners. Lower values generally require more weak learners.',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html'
            },
            'BaggingClassifier': {
                'n_estimators': 'The number of base estimators in the ensemble.',
                'max_samples': 'The proportion of samples to draw from X to train each base estimator.',
                'max_features': 'The proportion of features to draw from X to train each base estimator.',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html'
            },
            'BernoulliNB': {
                'alpha': 'Additive (Laplace/Lidstone) smoothing parameter.',
                'fit_prior': 'Whether to learn class prior probabilities or not.',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html'
            },
            'CalibratedClassifierCV': {
                'method': 'The method to use for probability calibration. Options: "sigmoid", "isotonic".',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html'
            },
            'DecisionTreeClassifier': {
                'criterion': 'The function to measure the quality of a split. Options: "gini", "entropy".',
                'max_depth': 'The maximum depth of the tree.',
                'min_samples_split': 'The minimum number of samples required to split an internal node.',
                'min_samples_leaf': 'The minimum number of samples required to be at a leaf node.',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html'
            },
            'ExtraTreeClassifier': {
                'criterion': 'The function to measure the quality of a split. Options: "gini", "entropy".',
                'max_depth': 'The maximum depth of the tree.',
                'min_samples_split': 'The minimum number of samples required to split an internal node.',
                'min_samples_leaf': 'The minimum number of samples required to be at a leaf node.',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html'
            },
            'ExtraTreesClassifier': {
                'n_estimators': 'The number of trees in the forest.',
                'criterion': 'The function to measure the quality of a split. Options: "gini", "entropy".',
                'max_depth': 'The maximum depth of the tree.',
                'min_samples_split': 'The minimum number of samples required to split an internal node.',
                'min_samples_leaf': 'The minimum number of samples required to be at a leaf node.',
                'max_features': 'The number of features to consider when looking for the best split.',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html'
            },
            'GaussianNB': {
                # No hyperparameters
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html'
            },
            'GradientBoostingClassifier': {
                'n_estimators': 'The number of boosting stages to be run.',
                'learning_rate': 'The learning rate shrinks the contribution of each tree.',
                'max_depth': 'The maximum depth of the individual regression estimators.',
                'min_samples_split': 'The minimum number of samples required to split an internal node.',
                'min_samples_leaf': 'The minimum number of samples required to be at a leaf node.',
                'max_features': 'The number of features to consider when looking for the best split.',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
            },
            'KNeighborsClassifier': {
                'n_neighbors': 'The number of neighbors to use.',
                'weights': 'The weight function used in prediction. Options: "uniform", "distance".',
                'algorithm': 'Algorithm used to compute the nearest neighbors. Options: "auto", "ball_tree", "kd_tree", "brute".',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'
            },
            'LogisticRegression': {
                'penalty': 'The norm used in the penalization. Options: "l1", "l2".',
                'C': 'Inverse of regularization strength.',
                'solver': 'Algorithm to use in the optimization problem. Options: "liblinear".',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'
            },
            'LinearSVC': {
                'penalty': 'The norm used in the penalization. Options: "l1", "l2".',
                'C': 'Regularization parameter.',
                'loss': 'The loss function to be used. Options: "hinge", "squared_hinge".',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html'
            },
            'NuSVC': {
                'nu': 'An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the range (0, 1].',
                'kernel': 'The kernel function to be used in the algorithm. Options: "linear", "poly", "rbf", "sigmoid".',
                'gamma': 'Kernel coefficient for "rbf", "poly", and "sigmoid" kernels. If "auto", uses 1 / n_features.',
                'Web Link': 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html'
            }
            }

regressor_dict = {
    'LinearRegression': LinearRegression,
    'SVR': SVR,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'ElasticNet': ElasticNet,
    'BayesianRidge': BayesianRidge,
    'SGD': SGDRegressor,
    'RandomForest': RandomForestRegressor,
    'GradientBoosting': GradientBoostingRegressor,
    'AdaBoost': AdaBoostRegressor,
    'DecisionTree': DecisionTreeRegressor,
    'KNN': KNeighborsRegressor,
    'XGBoost': XGBRegressor,
    'LightGBM': LGBMRegressor,
    'CatBoost': CatBoostRegressor,
    'MLPRegressor': MLPRegressor,
    'RunAll': None
}

classifier_dict = {
    'AdaBoostClassifier': AdaBoostClassifier,
    'BaggingClassifier': BaggingClassifier,
    'BernoulliNB': BernoulliNB,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'GaussianNB': GaussianNB,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'KNeighborsClassifier': KNeighborsClassifier,
    'LogisticRegression': LogisticRegression,
    'LinearSVC': LinearSVC,
    'NuSVC': NuSVC,
    'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'RidgeClassifier': RidgeClassifier,
    'SGDClassifier': SGDClassifier,
    'SVC': SVC,
    'RunAll': None
}

default_param_grids = {
            'SVR': {'C': [0.1, 1, 10, 100],'epsilon': [0.1, 0.2, 0.5, 1.0],'gamma': ['scale', 'auto', 1e-3, 1e-2, 1e-1, 1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'] },
            'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0], 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
            'Lasso': {'alpha': [0.1, 1.0, 10.0, 100.0],'max_iter': [1000, 2000, 3000]},
            'SGD': { 'alpha': [1e-6, 1e-4, 1e-2, 1.0], 'max_iter': [1000, 2000, 3000],'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']},
            'ElasticNet': {'alpha': [0.1, 1.0, 10.0, 100.0],'l1_ratio': [0.1, 0.5, 0.7, 1.0],'max_iter': [1000, 2000, 3000]},
            'BayesianRidge': {'n_iter': [100, 200, 300],'alpha_1': [1e-6, 1e-4, 1e-2],'alpha_2': [1e-6, 1e-4, 1e-2],'lambda_1': [1e-6, 1e-4, 1e-2],'lambda_2': [1e-6, 1e-4, 1e-2]},
            'KNN': {'n_neighbors': [3, 5, 7, 9],'weights': ['uniform', 'distance']},
            'RandomForest': {'n_estimators': [10, 50, 100, 200],'max_features': ['auto', 'sqrt', 'log2'],'max_depth': [None, 10, 20, 30]},
            'GradientBoosting': {'n_estimators': [100, 200, 300],'learning_rate': [0.01, 0.1, 0.2],'max_depth': [3, 5, 7]},
            'AdaBoost': {'n_estimators': [50, 100, 200],'learning_rate': [0.01, 0.1, 1.0]},
            'DecisionTree': { 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20], 'min_samples_leaf': [1, 5, 10]},
            'LinearRegression': {},
            'XGBoost': { 'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
            'LightGBM': { 'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'num_leaves': [31, 50, 100]},
            'CatBoost': { 'iterations': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'depth': [3, 5, 7]},
            'GaussianProcess': { 'alpha': [1e-10, 1e-8, 1e-6, 1e-4], 'optimizer': ['fmin_l_bfgs_b', None]},
            'MLPRegressor':{'hidden_layer_sizes': [[50,],[100,], [50,50], [100,50]],'activation': ['relu', 'tanh', 'logistic'],'solver': ['adam', 'sgd', 'lbfgs'],'alpha': [0.0001, 0.001, 0.01],'learning_rate_init': [0.001, 0.01, 0.1],'max_iter': [100, 200, 300],'batch_size': ['auto', 32, 64]},
            'AdaBoostClassifier': {'n_estimators': [50, 100, 200],'learning_rate': [0.01, 0.1, 1.0]},
            'BaggingClassifier': {'n_estimators': [10, 50, 100],'max_samples': [0.5, 0.7, 1.0],'max_features': [0.5, 0.7, 1.0]},
            'BernoulliNB': {'alpha': [0.1, 0.5, 1.0],'fit_prior': [True, False]},
            'CalibratedClassifierCV': {'method': ['sigmoid', 'isotonic']},
            'DecisionTreeClassifier': {'criterion': ['gini', 'entropy'],'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]},
            'ExtraTreeClassifier': {'criterion': ['gini', 'entropy'],'max_depth': [None, 5, 10],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]},
            'ExtraTreesClassifier': {'n_estimators': [50, 100, 200],'criterion': ['gini', 'entropy'],'max_depth': [None, 5, 10],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'max_features': ['auto', 'sqrt', 'log2']},
            'GaussianNB': {},  # No hyperparameters to tune
            'GradientBoostingClassifier': {'n_estimators': [50, 100, 200],'learning_rate': [0.01, 0.1, 0.5],'max_depth': [3, 5, 7],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'max_features': ['auto', 'sqrt', 'log2']},
            'KNeighborsClassifier': {'n_neighbors': [3, 5, 10],'weights': ['uniform', 'distance'],'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
            'LogisticRegression': {'penalty': ['l1', 'l2'],'C': [0.1, 1.0, 10.0],'solver': ['liblinear']},
            'LinearSVC': {'penalty': ['l1', 'l2'],'C': [0.1, 1.0, 10.0],'loss': ['hinge', 'squared_hinge']},
            'NuSVC': {'nu': [0.25, 0.5, 0.75],'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],'gamma': ['scale', 'auto']},
            'PassiveAggressiveClassifier': {'C': [0.1, 1.0, 10.0],'loss': ['hinge', 'squared_hinge']},
            'RandomForestClassifier': {'n_estimators': [50, 100, 200],'criterion': ['gini', 'entropy'],'max_depth': [None, 5, 10],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'max_features': ['auto', 'sqrt', 'log2']},
            'RidgeClassifier': {'alpha': [0.1, 1.0, 10.0],'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
            'SGDClassifier': {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],'penalty': ['l1', 'l2', 'elasticnet'],'alpha': [0.0001, 0.001, 0.01],'max_iter': [1000, 2000],'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']},
            'SVC': {'C': [0.1, 1.0, 10.0],'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],'gamma': ['scale', 'auto']}
        }

metrics=['method','trained_time','R2','MAE','MSE','RMSE','MAX','MAPE']

##### DISPLAY FUNCTIONS ########################################################################################################################################
def delete_figure_agg(figure):
    """
    Close the given Matplotlib figure.

    Args:
        figure (matplotlib.figure.Figure): The Matplotlib figure to be closed.
    """
    plt.close(figure)

def draw_figure(figure):
    """
    Convert a Matplotlib figure to a base64-encoded PNG image.

    Args:
        figure (matplotlib.figure.Figure): The Matplotlib figure to be converted.

    Returns:
        str: The base64-encoded PNG image.
    """
    img = io.BytesIO()
    figure.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

def plot_regression_result(y_test, y_pred):
    """
    Plot the regression results using Matplotlib.

    Args:
        y_test (pandas.Series): The true labels for the test data.
        y_pred (pandas.Series): The predicted labels for the test data.

    Returns:
        str: The base64-encoded PNG image of the plot.
    """
# Create a Matplotlib figure and axis
    fig, ax = plt.subplots()
    
    # Plot the scatter plot
    ax.scatter(y_test, y_pred, color='g', label='Predictions')
    
    # Fit a line to the points
    estimator = LinearRegression()
    estimator.fit(y_test.values.reshape(-1, 1), y_pred)
    y_pred_line = estimator.predict(y_test.values.reshape(-1, 1))
    
    # Plot the fitted line
    ax.plot(y_test, y_pred_line, color='r', label='Fitted Line')
    
    # Set axis labels and title
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Regression Results')
    
    # Add a legend
    ax.legend()
    
    # Save the plot as a PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    # Close the figure to free up memory
    plt.close(fig)
    
    return plot_url

def explain_model_shap(model_data, models, X_train, X_test, model_dict):
    feature_names = X_train.columns.tolist()
    is_classifier = any('Classifier' in model for model in models)

    try:
        # Use the trained model directly
        model = joblib.load(model_data['model_path'])
        model_name = models[0] if model_data['ensemble_method'] == 'none' else f"{model_data['ensemble_method']} Ensemble of {', '.join(models)}"
        predict = lambda X: model.predict_proba(X) if is_classifier else model.predict(X)

        if any(m in model_name for m in ['DecisionTree', 'RandomForest', 'LightGBM', 'CatBoost', 'XGBoost']):
            explainer = shap.TreeExplainer(model)
        elif any(m in model_name for m in ['LinearRegression', 'LogisticRegression']):
            explainer = shap.LinearExplainer(model, X_train)
        else:
            explainer = shap.KernelExplainer(predict, X_train)

        shap_values = explainer(X_test)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False, max_display=20)
        plt.title(f'SHAP Explanation for {model_name}')
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return {'plot_url': plot_url}
    except Exception as e:
        print(f"Error in explain_model_shap: {str(e)}")
        print(f"Error type: {type(e)}")
        return {'error': str(e)}

def explain_model_lime(model_data, models, X_train, X_test, model_dict):
    feature_names = X_train.columns.tolist()
    is_classifier = any('Classifier' in model for model in models)

    try:
        # Use the trained model directly
        model = joblib.load(model_data['model_path'])
        model_name = models[0] if model_data['ensemble_method'] == 'none' else f"{model_data['ensemble_method']} Ensemble of {', '.join(models)}"
        predict = lambda X: model.predict_proba(X) if is_classifier else model.predict(X)

        # Create LIME explainer
        if is_classifier:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                class_names=['class_' + str(i) for i in range(2)],
                mode='classification'
            )
        else:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                mode='regression'
            )

        # Generate explanation for first test instance
        exp = explainer.explain_instance(
            X_test.iloc[0].values, 
            predict,
            num_features=len(feature_names)
        )

        # Create plot
        plt.figure(figsize=(12, 8))
        exp.as_pyplot_figure()
        plt.title(f'LIME Explanation for {model_name}')
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return {'plot_url': plot_url}
    except Exception as e:
        print(f"Error in explain_model_lime: {str(e)}")
        print(f"Error type: {type(e)}")
        return {'error': str(e)}


def calculate_scores(y_true, y_pred):
    return [
        ['R2', r2_score(y_true, y_pred)],
        ['MSE', root_mean_squared_error(y_true, y_pred) ** 2],
        ['MAE', mean_absolute_error(y_true, y_pred)],
        ['MAX', max_error(y_true, y_pred)],
        ['RMSE', root_mean_squared_error(y_true, y_pred)],
        ['MAPE', mean_absolute_percentage_error(y_true, y_pred)]
    ]

def method_runall(X_train, X_test, y_train, y_test):
    """
    Run all regression models and generate visualizations for their performance metrics.

    Args:
        X_train (pandas.DataFrame): The training data features.
        X_test (pandas.DataFrame): The test data features.
        y_train (pandas.Series): The training data labels.
        y_test (pandas.Series): The test data labels.

    Returns:
        tuple: A tuple containing two lists:
            - table_test (list): A list of base64-encoded PNG images for test performance metrics.
            - table_train (list): A list of base64-encoded PNG images for training performance metrics.
    """
    table_test = []
    table_train = []

    for name, est in regressor_dict.items():
        if est is not None:
            est.fit(X_train, y_train)

            y_test_pred = est.predict(X_test)

            # Calculate the metrics for test result
            r2_test = r2_score(y_test, y_test_pred)
            MAE_test = mean_absolute_error(y_test, y_test_pred)
            MSE_test = root_mean_squared_error(y_test, y_test_pred) ** 2
            RMSE_test = root_mean_squared_error(y_test, y_test_pred)
            MAX_test = max_error(y_test, y_test_pred)
            MAPE_test = mean_absolute_percentage_error(y_test, y_test_pred)

            # Create a Matplotlib figure for the test performance metrics
            fig, ax = plt.subplots(figsize=(8, 6))
            metrics = ['R-squared', 'MAE', 'MSE', 'RMSE', 'MAX Error', 'MAPE']
            values = [r2_test, MAE_test, MSE_test, RMSE_test, MAX_test, MAPE_test]
            ax.bar(metrics, values)
            ax.set_title(f'{name} - Test Performance Metrics')
            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')

            # Save the plot as a PNG image
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Close the figure to free up memory
            plt.close(fig)

            table_test.append(plot_url)

            y_train_pred = est.predict(X_train)
            r2_train = r2_score(y_train, y_train_pred)
            MAE_train = mean_absolute_error(y_train, y_train_pred)
            MSE_train = root_mean_squared_error(y_train, y_train_pred) ** 2
            RMSE_train = root_mean_squared_error(y_train, y_train_pred)
            MAX_train = max_error(y_train, y_train_pred)
            MAPE_train = mean_absolute_percentage_error(y_train, y_train_pred)

            # Create a Matplotlib figure for the training performance metrics
            fig, ax = plt.subplots(figsize=(8, 6))
            metrics = ['R-squared', 'MAE', 'MSE', 'RMSE', 'MAX Error', 'MAPE']
            values = [r2_train, MAE_train, MSE_train, RMSE_train, MAX_train, MAPE_train]
            ax.bar(metrics, values)
            ax.set_title(f'{name} - Training Performance Metrics')
            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')

            # Save the plot as a PNG image
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Close the figure to free up memory
            plt.close(fig)

            table_train.append(plot_url)

    return table_test, table_train

def generate_performance_metrics(y_true, y_pred, model_name, data_type):
    metrics = []

    r2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    MSE = root_mean_squared_error(y_true, y_pred) ** 2
    RMSE = root_mean_squared_error(y_true, y_pred)
    MAX = max_error(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true, y_pred)

    metric_names = ['R-squared', 'MAE', 'MSE', 'RMSE', 'MAX Error', 'MAPE']
    metric_values = [r2, MAE, MSE, RMSE, MAX, MAPE]

    # Create the metrics list
    metrics = list(zip(metric_names, metric_values))

    # Generate the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(metric_names, metric_values)
    ax.set_title(f'{model_name} - {data_type} Performance Metrics')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')

    plot_buffer = io.BytesIO()
    plt.savefig(plot_buffer, format='png')
    plot_buffer.seek(0)
    plot_url = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return {
        'metrics': metrics,
        'plot_url': plot_url
    }

def generate_classification_metrics(y_true, y_pred, model_name, data_type):
    metrics = []

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC-ROC calculation
    classes = np.unique(y_true)
    n_classes = len(classes)

    if n_classes == 2:
        auc_roc = roc_auc_score(y_true, y_pred)
    else:
        y_true_bin = label_binarize(y_true, classes=classes)
        y_pred_bin = label_binarize(y_pred, classes=classes)
        auc_roc = roc_auc_score(y_true_bin, y_pred_bin, average='weighted', multi_class='ovr')

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    metric_values = [accuracy, precision, recall, f1, auc_roc]

    # Create the metrics list
    metrics = list(zip(metric_names, metric_values))

    # Generate the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(metric_names[:4], metric_values[:4])  # Exclude AUC-ROC from the plot
    ax.set_title(f'{model_name} - {data_type} Performance Metrics')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_ylim(0, 1)  # Set y-axis limit from 0 to 1

    plot_buffer = io.BytesIO()
    plt.savefig(plot_buffer, format='png')
    plot_buffer.seek(0)
    plot_url = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return {
        'metrics': metrics,
        'plot_url': plot_url
    }


def singleML_regression(model_name, X_train, X_test, y_train, y_test, param_values, search_method):
    logger.debug(f"Starting singleML_regression with model: {model_name}, search method: {search_method}")
    logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.debug(f"Param values: {param_values}")

    try:
        if model_name == 'LinearRegression':
            est = regressor_dict[model_name]()
            est.fit(X_train, y_train)
            best_params = est.get_params()
        else:
            param = {}
            for key in default_param_grids[model_name].keys():
                prefixed_key = f'{model_name}-{key}'
                item = param_values.get(prefixed_key, '')
                item = item.strip()
                if item:
                    try:
                        item = [float(x) if '.' in x or 'e-' in x else int(x) for x in item.split(',')]
                    except ValueError:
                        item = item.split(',')
                param[key] = item
            
            logger.debug(f"Processed parameters: {param}")

            est = regressor_dict[model_name]()

            if search_method.lower() == 'none':
                est.set_params(**{k: v[0] if isinstance(v, list) else v for k, v in param.items()})
                est.fit(X_train, y_train)
                best_params = est.get_params()
            elif search_method.lower() in ['grid', 'random']:
                search_class = GridSearchCV if search_method.lower() == 'grid' else RandomizedSearchCV
                est = search_class(regressor_dict[model_name](), param, cv=5)
                est.fit(X_train, y_train)
                best_params = est.best_params_
                est = est.best_estimator_
            else:
                logger.warning(f"Unrecognized search_method: {search_method}. Defaulting to 'None'.")
                est.fit(X_train, y_train)
                best_params = est.get_params()
            
        logger.debug("Model fitting completed")

        y_test_pred = est.predict(X_test)
        y_train_pred = est.predict(X_train)

        logger.debug("Predictions generated")

        plot_url = plot_regression_result(y_test, y_test_pred)
        train_results = generate_performance_metrics(y_train, y_train_pred, model_name, 'Training')
        test_results = generate_performance_metrics(y_test, y_test_pred, model_name, 'Test')

        return {
            'model_name': model_name,
            'plot_url': plot_url,
            'train_metrics': train_results['metrics'],
            'test_metrics': test_results['metrics'],
            'train_metrics_plot': train_results['plot_url'],
            'test_metrics_plot': test_results['plot_url'],
            'best_params': best_params,
            'model': est
        }

    except Exception as e:
        logger.error(f"Error in singleML_regression: {str(e)}", exc_info=True)
        raise

def ensembleML_regression(ensemble_method, regressors, X_train, X_test, y_train, y_test):
    logging.debug(f"Starting ensembleML_regression with method: {ensemble_method}")
    logging.debug(f"Regressors: {regressors}")
    logging.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Check the ensemble method
    valid_methods = ['bagging', 'stacking', 'voting']
    if ensemble_method not in valid_methods:
        raise ValueError(f"Invalid ensemble method: {ensemble_method}. Must be one of {valid_methods}")

    # Check if regressors are valid
    invalid_regressors = [reg for reg in regressors if reg not in regressor_dict]
    if invalid_regressors:
        raise ValueError(f"Invalid regressor(s): {invalid_regressors}. Must be one of {list(regressor_dict.keys())}")

    try:
        if ensemble_method == 'bagging':
            base_model = regressor_dict[regressors[0]]()
            model = BaggingRegressor(estimator=base_model, n_estimators=10, random_state=0)
        elif ensemble_method == 'stacking':
            estimators = [(reg, regressor_dict[reg]()) for reg in regressors[:-1]]
            final_estimator = regressor_dict[regressors[-1]]()
            model = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=10)
        elif ensemble_method == 'voting':
            estimators = [(reg, regressor_dict[reg]()) for reg in regressors]
            model = VotingRegressor(estimators=estimators)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        train_metrics = calculate_scores(y_train, y_train_pred)
        test_metrics = calculate_scores(y_test, y_pred)

        plot_url = plot_regression_result(y_test, y_pred)
        train_metrics_plot = generate_performance_metrics(y_train, y_train_pred, ensemble_method, 'Train')
        test_metrics_plot = generate_performance_metrics(y_test, y_pred, ensemble_method, 'Test')

        return {
            'model': model,
            'model_name': ensemble_method,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'plot_url': plot_url,
            'train_metrics_plot': train_metrics_plot,
            'test_metrics_plot': test_metrics_plot
        }
    except Exception as e:
        logging.error(f"Error in ensembleML_regression: {str(e)}")
        return {
            'model': None,
            'model_name': ensemble_method,
            'train_metrics': [],
            'test_metrics': [],
            'plot_url': None,
            'train_metrics_plot': None,
            'test_metrics_plot': None,
            'error': str(e)
        }


def singleML_classification(model_name, X_train, X_test, y_train, y_test, param_values, search_method):
    logger.debug(f"Starting singleML_classification with model: {model_name}, search method: {search_method}")
    logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.debug(f"Param values: {param_values}")

    try:
        if model_name == 'LogisticRegression':
            est = classifier_dict[model_name]()
            est.fit(X_train, y_train)
            best_params = est.get_params()
        else:
            param = {}
            for key in default_param_grids[model_name].keys():
                prefixed_key = f'{model_name}-{key}'
                item = param_values.get(prefixed_key, '')
                item = item.strip()
                if item:
                    try:
                        item = [float(x) if '.' in x or 'e-' in x else int(x) for x in item.split(',')]
                    except ValueError:
                        item = item.split(',')
                param[key] = item
            
            logger.debug(f"Processed parameters: {param}")

            est = classifier_dict[model_name]()

            if search_method.lower() == 'none':
                est.set_params(**{k: v[0] if isinstance(v, list) else v for k, v in param.items()})
                est.fit(X_train, y_train)
                best_params = est.get_params()
            elif search_method.lower() in ['grid', 'random']:
                search_class = GridSearchCV if search_method.lower() == 'grid' else RandomizedSearchCV
                est = search_class(classifier_dict[model_name](), param, cv=5)
                est.fit(X_train, y_train)
                best_params = est.best_params_
                est = est.best_estimator_
            else:
                logger.warning(f"Unrecognized search_method: {search_method}. Defaulting to 'None'.")
                est.fit(X_train, y_train)
                best_params = est.get_params()
            
        logger.debug("Model fitting completed")

        y_test_pred = est.predict(X_test)
        y_train_pred = est.predict(X_train)

        train_results = generate_classification_metrics(y_train, y_train_pred, model_name, 'Training')
        test_results = generate_classification_metrics(y_test, y_test_pred, model_name, 'Test')

        classification_report = plot_classification_report(y_test, y_test_pred)
        cm_plot = plot_confusion_matrix(y_test, y_test_pred, est)
        roc_plot = plot_roc_curve(est, X_test, y_test)
        learning_curve_plot = plot_learning_curve(est, X_train, y_train)

        cm_plot_url = draw_figure(cm_plot)
        roc_plot_url = draw_figure(roc_plot)
        learning_curve_url = draw_figure(learning_curve_plot)

        return {
            'model_name': model_name,
            'train_metrics': train_results['metrics'],
            'test_metrics': test_results['metrics'],
            'train_metrics_plot': train_results['plot_url'],
            'test_metrics_plot': test_results['plot_url'],
            'classification_report': classification_report,
            'confusion_matrix_plot': cm_plot_url,
            'roc_curve_plot': roc_plot_url,
            'learning_curve_plot': learning_curve_url,
            'best_params': best_params,
            'model': est
        }

    except Exception as e:
        logger.error(f"Error in singleML_classification: {str(e)}", exc_info=True)
        raise

def ensembleML_classification(ensemble_method, classifiers, X_train, X_test, y_train, y_test):
    logger.debug(f"Starting ensembleML_classification with method: {ensemble_method}")
    logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    try:
        if ensemble_method == 'Bagging':
            base_model = classifier_dict[classifiers[0]]()
            model = BaggingClassifier(estimator=base_model, n_estimators=10, random_state=0)
        elif ensemble_method == 'Stacking':
            estimators = [(clf, classifier_dict[clf]()) for clf in classifiers[:-1]]
            final_estimator = classifier_dict[classifiers[-1]]()
            model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=10)
        elif ensemble_method == 'Voting':
            estimators = [(clf, classifier_dict[clf]()) for clf in classifiers]
            model = VotingClassifier(estimators=estimators)
        else:
            raise ValueError(f"Unsupported ensemble method: {ensemble_method}")

        model.fit(X_train, y_train)
        logger.debug("Model fitting completed")

        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        train_results = generate_classification_metrics(y_train, y_train_pred, ensemble_method, 'Training')
        test_results = generate_classification_metrics(y_test, y_test_pred, ensemble_method, 'Test')

        classification_report = plot_classification_report(y_test, y_test_pred)
        cm_plot = plot_confusion_matrix(y_test, y_test_pred, model)
        roc_plot = plot_roc_curve(model, X_test, y_test) if len(np.unique(y_test)) == 2 else None
        learning_curve_plot = plot_learning_curve(model, X_train, y_train)

        cm_plot_url = draw_figure(cm_plot)
        roc_plot_url = draw_figure(roc_plot) if roc_plot else None
        learning_curve_url = draw_figure(learning_curve_plot)

        return {
            'model_name': ensemble_method,
            'train_metrics': train_results['metrics'],
            'test_metrics': test_results['metrics'],
            'train_metrics_plot': train_results['plot_url'],
            'test_metrics_plot': test_results['plot_url'],
            'classification_report': classification_report,
            'confusion_matrix_plot': cm_plot_url,
            'roc_curve_plot': roc_plot_url,
            'learning_curve_plot': learning_curve_url,
            'model': model
        }

    except Exception as e:
        logger.error(f"Error in ensembleML_classification: {str(e)}", exc_info=True)
        raise

def calculate_classification_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def plot_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    return report

def plot_roc_curve(model, X_test, y_test):
    classes = np.unique(y_test)
    n_classes = len(classes)
    
    if n_classes == 2:
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        y_score = model.predict_proba(X_test)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, color in zip(range(n_classes), cycle(['aqua', 'darkorange', 'cornflowerblue'])):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0))
    fig.tight_layout()
    
    return fig

def plot_confusion_matrix(y_test, y_pred, model):
    cm = confusion_matrix(y_test, y_pred)
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    n_classes = len(unique_labels)
    
    fig, ax = plt.subplots(figsize=(max(6, n_classes/2), max(5, n_classes/2)))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=unique_labels, 
           yticklabels=unique_labels,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")

    fig.tight_layout()
    plt.title('Confusion Matrix')
    return fig

def plot_learning_curve(model, X_train, y_train):
    train_sizes = np.linspace(0.1, 1.0, 5)
    n_samples = X_train.shape[0]
    cv = ShuffleSplit(n_splits=min(5, n_samples // 2), test_size=0.2, random_state=42)
    
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train, train_sizes=train_sizes, cv=cv, n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    fig, ax = plt.subplots()
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                    valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.legend(loc="best")
    
    return fig


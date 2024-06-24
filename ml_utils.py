import PySimpleGUI as sg
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import time
import csv


from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold,cross_val_score,learning_curve
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

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, mean_absolute_percentage_error, classification_report, roc_curve,roc_auc_score,RocCurveDisplay,confusion_matrix, ConfusionMatrixDisplay

import shap
import lime
import lime.lime_tabular

# button_mapping_reg = { '-BUTTON1-':'Linear Regression','-BUTTON2-':'Ridge', '-BUTTON3-':'Muliple Regression','-BUTTON4-':'Bayesian Ridge', '-BUTTON5-': 'ElasticNet',
#                     '-BUTTON6-':'Polynomial Regression','-BUTTON7-':'Decision Tree','-BUTTON8-': 'SVM', '-BUTTON9-': 'K-nearest Neighbor', '-BUTTON10-': 'Neural Network',
#                     '-BUTTON11-':'Gradient Boosting','-BUTTON12-':'Random Forest','-BUTTON13-': 'Bagging', '-BUTTON14-': 'AdaBoost' , '-BUTTON15-':'SGD' ,
#                     '-BUTTON16-':'Voting','-BUTTON17-':'Stacking','-BUTTON18-': 'RunAll'
#                     }

# button_mapping_cls = { '-BUTTON19-':'Decision Tree','-BUTTON20-': 'SVM', '-BUTTON21-': 'K-nearest Neighbor', '-BUTTON22-': 'Neural Network',
#                     '-BUTTON23-':'Gradient Boosting','-BUTTON24-': 'RunAll'}

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

regressor_dict = {'LinearRegression': LinearRegression(),'SVR': SVR(),'Ridge': Ridge(),'Lasso': Lasso(),'ElasticNet': ElasticNet(),'BayesianRidge': BayesianRidge(),'SGD': SGDRegressor(),'RandomForest': RandomForestRegressor(),
                  'GradientBoosting': GradientBoostingRegressor(),'AdaBoost': AdaBoostRegressor(),'DecisionTree': DecisionTreeRegressor(),'KNN': KNeighborsRegressor(),'XGBoost': XGBRegressor(),
                  'LightGBM': LGBMRegressor(),'CatBoost': CatBoostRegressor(),'MLPRegressor': MLPRegressor(),'RunAll':None}

classifier_dict = {'AdaBoostClassifier': AdaBoostClassifier(),'BaggingClassifier': BaggingClassifier(),'BernoulliNB': BernoulliNB(),'DecisionTreeClassifier': DecisionTreeClassifier(),
                #    'CalibratedClassifierCV': CalibratedClassifierCV(), 'ExtraTreeClassifier': ExtraTreeClassifier(), 'ExtraTreesClassifier': ExtraTreesClassifier(),
                    'GaussianNB': GaussianNB(),'GradientBoostingClassifier': GradientBoostingClassifier(),'KNeighborsClassifier': KNeighborsClassifier(),'LogisticRegression': LogisticRegression(),
                    'LinearSVC': LinearSVC(),'NuSVC': NuSVC(),'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),'RandomForestClassifier': RandomForestClassifier(),'RidgeClassifier': RidgeClassifier(),
                    'SGDClassifier': SGDClassifier(),'SVC': SVC(),'RunAll':None}


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

############### Canvas related functions used in windows ############
def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    try:
        draw_figure.canvas_packed.pop(figure_agg.get_tk_widget())
    except Exception as e:
        print(f'Error removing {figure_agg} from list', e)
    plt.close('all')

def draw_figure(canvas, figure):
    if not hasattr(draw_figure, 'canvas_packed'):
        draw_figure.canvas_packed = {}
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    widget = figure_canvas_agg.get_tk_widget()
    if widget not in draw_figure.canvas_packed:
        draw_figure.canvas_packed[widget] = figure
        widget.pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def plot_regression_result(y_test, y_pred):
    plt.close('all')
    # num_outputs = y_pred.shape[1]
    # if num_outputs >1:
    #     fig,axs = plt.subplots(num_outputs,1,figsize=(6,6))
    #     label_name = y_test.columns
    #     for i in range(num_outputs):
    #         test = y_test.iloc[:,i]
    #         test = test.values.reshape(-1,1)
    #         pred = y_pred[:,i]
    #         estimator = LinearRegression()
    #         estimator.fit(test,pred)
    #         y_pred1 = estimator.predict(test)
    #         axs[i].scatter(test, pred,color='g')
    #         axs[i].plot(test,y_pred1,color = 'r')
    #         axs[i].set_xlabel("Test")
    #         axs[i].set_ylabel("Predict")
    #         axs[i].set_title(f'result for {label_name[i]}')
    #     plt.tight_layout()
    #     return fig
    # else:
    estimator = LinearRegression()
    estimator.fit(y_test,y_pred)
    y_pred1 = estimator.predict(y_test)
    plt.scatter(y_test, y_pred,color='g')
    plt.plot(y_test,y_pred1, color = 'r')
    plt.xlabel("Test")
    plt.ylabel("Predict")
    fig = plt.gcf()  # get the figure to show
    return fig

def explain_model_shap(model,model_name, X_train, X_test):
    plt.close("all")
    feature_names = X_test.columns
    if model_name in ['DecisionTree','RandomForest','LightGBM','CatBoost','XGBoost']:
        explainer = shap.TreeExplainer(model, X_train)
    elif model_name in ['LinearRegression','LogicalRegression']:
        explainer = shap.LinearExplainer(model.predict, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train)

    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test,feature_names)

def explain_model_lime(model, X_train, X_test):
    feature_name = X_train.columns
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,feature_names=feature_name,verbose=True,mode='regression')
    explanation = explainer.explain_instance(X_test.values[0], model.predict)
    feature_importances = explanation.as_list()
    features, importances= zip(*feature_importances)
    plt.close('all')
    plt.barh(features,importances)
    plt.xlabel('Importance')
    plt.title('LIME Feature Importances')
    plt.ylabel('Features')
    plt.gca().invert_yaxis()
    plt.show()
    # explanation.show_in_notebook(show_table=True)

def method_runall(X_train,X_test,y_train,y_test):
    table_train= []
    table_test = []

    for name,est in regressor_dict.items():
        start_time = time.time()
        print(name,est)
        if est != None:
            est.fit(X_train, y_train)
            estimated_time = time.time() - start_time

            y_test_pred = est.predict(X_test)

            # calculate the metricx for test result
            r2_test = r2_score(y_test, y_test_pred)
            MAE_test = mean_absolute_error(y_test, y_test_pred)
            MSE_test =  mean_squared_error(y_test, y_test_pred)
            RMSE_test = mean_squared_error(y_test, y_test_pred, squared=False)
            MAX_test = max_error(y_test, y_test_pred)
            MAPE_test =  mean_absolute_percentage_error(y_test,y_test_pred)

            table_t =[name,estimated_time,r2_test,MAE_test,MSE_test,RMSE_test,MAX_test,MAPE_test]
            table_test.append(table_t)

            y_train_pred = est.predict(X_train)
            r2_train = r2_score(y_train,y_train_pred)
            MAE_train = mean_absolute_error(y_train,y_train_pred)
            MSE_train = mean_squared_error(y_train,y_train_pred)
            RMSE_train = mean_squared_error(y_train,y_train_pred, squared=False)
            MAX_train = max_error(y_train,y_train_pred)
            MAPE_train = mean_absolute_percentage_error(y_train,y_train_pred)
            table =[name,estimated_time,r2_train,MAE_train,MSE_train,RMSE_train,MAX_train,MAPE_train]
            table_train.append(table)

    return table_test,table_train

# def setModel_singleML(regressor,param,search_method):

#     if search_method =='None':
#         model = regressor_dict[regressor]
#         model.set_params(**param)
#     elif search_method == 'Grid':
#         model = GridSearchCV(regressor_dict[regressor],param, cv=5)
#     else:
#         model == None

#     return model

def create_param_input_fields(param_grid,search_method):
    fields = []
    if search_method == 0:
        for param, default_values in param_grid.items():
            if any(isinstance(item,str) for item in default_values):
                fields.append([sg.Text(param,size=(5,1)),sg.Combo(values=default_values, default_value= default_values[0], key=f'-Param_None_{param}-')])
            else:
                fields.append([sg.Text(param,size=(5,1)),sg.Input( default_text = default_values[0], key=f'-Param_None_{param}-')])

    elif search_method==1:
        for param, default_values in param_grid.items():
            fields.append([sg.Text(param,size=(5,1)),sg.Input(', '.join(map(str, default_values)), key=f'-Param_Grid_{param}-')])
    elif search_method==2:
        for param, default_values in param_grid.items():
            fields.append([sg.Text(param,size=(5,1)),sg.Input(', '.join(map(str, default_values)), key=f'-Param_Random_{param}-')])
    return fields

def singleML_None_search_layout(regressor):
    param_grid = default_param_grids[regressor]
    layout = create_param_input_fields(param_grid,0)
    frame_catego = sg.pin(sg.Col(layout,expand_x=True, expand_y=True,visible=True,key='-COL_NONE-',pad =(0,0)),expand_x=True, expand_y=True, shrink=True)

    return frame_catego

def singleML_GridCV_search_layout(regressor):
    param_grid = default_param_grids[regressor]
    layout = create_param_input_fields(param_grid,1)
    frame_catego = sg.pin(sg.Col(layout,expand_x=True, expand_y=True,visible=False,key='-COL_GRID-',pad =(0,0)),expand_x=True, expand_y=True, shrink=True)
    return frame_catego

def singleML_RandomCV_search_layout(regressor):
    param_grid = default_param_grids[regressor]
    layout = create_param_input_fields(param_grid,2)
    frame_catego = sg.pin(sg.Col(layout,expand_x=True, expand_y=True,visible=False,key='-COL_RANDOM-',pad =(0,0)),expand_x=True, expand_y=True, shrink=True)
    return frame_catego

def singleML_window_regression(main_windows,event,X_train,X_test,y_train,y_test):

    regressor = event
    search_method ='None'

    if regressor == 'RunAll':
        table_test,table_train = method_runall(X_train,X_test,y_train,y_test)
        table_layout_train =sg.Table(values=table_train,headings=metrics,auto_size_columns=True, hide_vertical_scroll=True, expand_x=True,expand_y=True,key='-TABLEALL-')
        table_layout_test =sg.Table(values=table_test,headings=metrics,auto_size_columns=True, hide_vertical_scroll=True, expand_x=True,expand_y=True,key='-TABLEALLTEST-')
        layout = [[sg.Column([[sg.Text('Train result')],[table_layout_train],[sg.Text('Test result')],[ table_layout_test],[sg.pin(sg.Column([[sg.Push(),sg.Button('Back',button_color='red',pad=(10,10)), sg.Button('Exit',button_color='red',pad=(10,10)),sg.Push()]],expand_x= True,pad =(0,0)),expand_x=True)]],expand_x=True,expand_y=True,key='-COLRUNALL-')]]
    else:
        if event == 'LinearRegression':
           layout = [[sg.pin(sg.Frame('',layout=[[sg.Button('Train and Predict', size=(20,2))]]))]]
        else:

            layout_none= singleML_None_search_layout(regressor)
            layout_grid= singleML_GridCV_search_layout(regressor)
            layout_random = singleML_RandomCV_search_layout(regressor)
            layout = [[sg.Text('Parameter Optimization Method:'),sg.Radio('None', 'OPTIMIZATION_METHOD', key='-NONE_SEARCH-', enable_events=True, default=True),sg.Radio('Grid Search', 'OPTIMIZATION_METHOD', key='-GRID_SEARCH-',enable_events=True), sg.Radio('Random Search', 'OPTIMIZATION_METHOD', key='-RANDOM_SEARCH-',enable_events=True)]]
            layout += [[sg.pin(sg.Frame('',layout=[[layout_none],[layout_grid],[layout_random]],expand_x=True, expand_y=True,pad=(0,0)),expand_x=True,expand_y=True,shrink=True),sg.Frame('',layout=[[sg.Button('Train and Predict', size=(20,2))]])]]

        layout +=[[ sg.Frame('',layout=[[
                            sg.Column([[sg.Text('Result')],[sg.Canvas(key='-CANVAS-',size=(600,400),background_color='white')]],expand_x=True,expand_y=True),
                            sg.Column([[sg.Text('Train Score')],[sg.Table(values=[],headings=['Metric', 'Score'],justification='left',auto_size_columns=True,hide_vertical_scroll=True, key='-TABLETRAIN-',expand_x=True)],
                                        [sg.VPush()], [sg.Text('Test Score')],[sg.Table(values=[],headings=['Metric', 'Score'],justification='left',auto_size_columns=True, hide_vertical_scroll=True, expand_x=True,key='-TABLETEST-')],
                                        ],expand_x=True,expand_y=True)]],
                                        expand_x=True, expand_y=True,visible=True,key = '-CONTROL-')],
                                        [sg.Frame('',layout=[[sg.pin(sg.Column([[sg.Button('Explain with SHAP'),sg.Button('Explain with LIME'), sg.Button('View Optimal Parameters'), sg.Button('Save Trained Model'),sg.Button('Save Displayed Result'),sg.Button('Help',button_color='red',pad=(10,10)),sg.Button('Back',button_color='red',pad=(10,10)), sg.Button('Exit',button_color='red',pad=(10,10)),sg.Push()]],expand_x= True,pad =(0,0)),expand_x=True)]])]]

    window = sg.Window(f'{regressor}',layout=layout,finalize=True)
    figure_agg = None
    best_params = None
    model_explain = None
    is_saved = False

    while True:
        event,values = window.read()
        print(event,values)

        if event ==sg.WIN_CLOSED:
            main_windows.un_hide()
            break

        if event == '-GRID_SEARCH-':
            search_method = 'Grid'
            window['-COL_NONE-'].update(visible = False)
            window['-COL_RANDOM-'].update(visible = False)
            window['-COL_GRID-'].update(visible = True)

        if event =='-RANDOM_SEARCH-':
            search_method = 'Random'
            window['-COL_NONE-'].update(visible = False)
            window['-COL_GRID-'].update(visible = False)
            window['-COL_RANDOM-'].update(visible = True)

        if event == '-NONE_SEARCH-':
            search_method = 'None'
            window['-COL_GRID-'].update(visible = False)
            window['-COL_RANDOM-'].update(visible = False)
            window['-COL_NONE-'].update(visible = True)

        if event =='Train and Predict':
            if figure_agg:
                delete_figure_agg(figure_agg)

            if event =='LinearRegression':
                est = regressor_dict[regressor]
                est.fit(X_train, y_train)
            else:
                param ={}
                if search_method == 'None':
                    for key in default_param_grids[regressor].keys():
                        item = values[f'-Param_None_{key}-']
                        item = item.strip()
                        if isinstance(item,int):
                            item = int(item)
                        else:
                            try:
                                if '.' in item or 'e-' in item:
                                    item = float(item)
                                else:
                                    item = int(item)
                            except ValueError:
                                item = item
                        param[key] = item
                else:
                    for key in default_param_grids[regressor].keys():
                        new_values_str = values[f'-Param_Grid_{key}-']
                        print(new_values_str)
                        new_values_str = new_values_str.strip()
                        new_list=[]
                        if new_values_str.startswith('[') and new_values_str.endswith(']'):
                            new_values_str= new_values_str.split('[')[1:]
                            for  item in new_values_str:
                                item_list = item.split(']')[0].split(',')
                                input_list = [int(x.strip()) for x in item_list if x.strip()]
                                new_list.append(input_list)
                            print(new_list)
                        else:
                            for item in new_values_str.split(','):
                                item = item.strip()
                                try:
                                    if '.' in item or 'e-' in item:
                                        new_list.append(float(item))
                                    else:
                                        new_list.append(int(item))
                                except ValueError:
                                    new_list.append(item)
                        param[key] = new_list

                if search_method =='None':
                    est = regressor_dict[regressor]
                    est.set_params(**param)
                elif search_method == 'Grid':
                    est = GridSearchCV(regressor_dict[regressor],param, cv=5)
                else:
                    est == None

                est.fit(X_train, y_train)

                if search_method !='None':
                    best_params = est.best_params_
                    est = est.best_estimator_
                else:
                    best_params = est.get_params()

            model_explain = est
            y_test_pred = est.predict(X_test)
            y_train_pred = est.predict(X_train)
            fig = plot_regression_result(y_test,y_test_pred)
            figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

            train_score = [['R2', r2_score(y_train,y_train_pred)],
                            ['MSE',mean_squared_error(y_train,y_train_pred)],
                            ['MAE',mean_absolute_error(y_train,y_train_pred)],
                            ['MAX',max_error(y_train,y_train_pred)],
                            ['RMSE',mean_squared_error(y_train,y_train_pred, squared=False)],
                            ['MAPE',mean_absolute_percentage_error(y_train,y_train_pred)]]

            test_score =[['R2', r2_score(y_test,y_test_pred)],
                        ['MSE',mean_squared_error(y_test,y_test_pred)],
                        ['MAE',mean_absolute_error(y_test,y_test_pred)],
                        ['MAX',max_error(y_test,y_test_pred)],
                        ['RMSE',mean_squared_error(y_test, y_test_pred, squared=False)],
                        ['MAPE',mean_absolute_percentage_error(y_test, y_test_pred)]]

            window['-TABLETRAIN-'].update(values = train_score)
            window['-TABLETEST-'].update(values = test_score)
            window['-CONTROL-'].update(visible = True)

        # save image.
        if event == 'Save Trained Model':
            is_saved = True

            # save model to user_model dic.
            filename = 'users_model/'+ f'{regressor}.joblib'
            joblib.dump(est, filename)
            sg.popup('Your model have been saved in Train-ML!')

        if event == 'Save Displayed Result':
            filepath =sg.popup_get_file('Open',no_window=True,save_as=True,file_types=[("PDF","*.pdf"),("PNG","*.png"),("JPG","*.jpg")])
            if filepath:
                fig.savefig(fname = filepath)
                sg.popup('Your model have been saved in Train-ML!')
            else:
                sg.popup_error('The filepath is invalid!')
            filepath = sg.popup_get_file('Save as', no_window=True,save_as= True,default_extension='.csv')
            if filepath:
                with open(filepath,"w",newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Train result'])
                    csvwriter.writerow(['Metric', 'Score'])
                    for row in train_score:
                        csvwriter.writerow(row)
                    csvwriter.writerow(['Test result'])
                    csvwriter.writerow(['Metric', 'Score'])
                    for row in test_score:
                        csvwriter.writerow(row)

        if event == 'View Optimal Parameters':
            sg.popup(f'Best Parameters:{best_params}')

        if event == 'Explain with SHAP':
            explain_model_shap(model_explain,regressor, X_train,X_test)

        if event == 'Explain with LIME':
            explain_model_lime(model_explain, X_train, X_test)

        if event == 'Help':
            para_text = param_tips[regressor]
            help_text = '\n\n'.join([f'{key}:{value}' for key,value in para_text.items()])
            sg.popup_scrolled(help_text,title=f'Parameter Tips for {regressor}',text_color='blue')

        if event == 'Back':
            if is_saved == False:
                is_yes = sg.popup('You do not save your model. Do you want to save your trained model first?',button_type=sg.POPUP_BUTTONS_YES_NO)
                if is_yes == 'Yes':
                    continue
            window.close()
            main_windows.un_hide()
            main_windows['-NONE_ENSEMBLE-'].update(True)

        if event == 'Exit':
            if is_saved == False:
                is_yes = sg.popup('You do not save your model. Do you want to save your trained model first?',button_type=sg.POPUP_BUTTONS_YES_NO)
                if is_yes == 'Yes':
                    continue

            is_yes = sg.popup('Are you sure to exit the ML Module?',button_type=sg.POPUP_BUTTONS_YES_NO)
            if is_yes == 'Yes':
                window.close()
                main_windows.close()

    window.close()

def ensembelML_window_regression(main_windows,event,para_list,X_train,X_test,y_train,y_test):
    ensemble_method = event
    model_list =[]
    field =[]
    figure_agg = None
    is_saved = False

    if ensemble_method == '-BAG-':
        base_model = para_list[0][0]
        base_model_param = para_list[0][1]
        layout = [[sg.pin(sg.Frame('',layout=[[sg.Text(f'{base_model}'),sg.Text(f"{base_model_param}")]],expand_x=True, expand_y=True,pad=(0,0)),expand_x=True,expand_y=True,shrink=True),sg.Frame('',layout=[[sg.Button('Train and Predict', size=(20,2))]])]]
        regressor = regressor_dict[base_model]
        regressor.set_params(**base_model_param)
        ensemble_model = BaggingRegressor(estimator=regressor,n_estimators=10, random_state=0)
    elif ensemble_method =='-STACK-':
        for i in range(len(para_list)):
            base_model = para_list[i][0]
            base_model_param = para_list[i][1]
            field += [[sg.Text(f'{base_model}'),sg.Text(f"{base_model_param}")]]
            regressor = regressor_dict[base_model]
            regressor.set_params(**base_model_param)
            model_list.append((base_model,regressor))
        layout = [[sg.pin(sg.Frame('',layout=field,expand_x=True, expand_y=True,pad=(0,0)),expand_x=True,expand_y=True,shrink=True),sg.Frame('',layout=[[sg.Button('Train and Predict', size=(20,2))]])]]
        ensemble_model = StackingRegressor(estimators=model_list[:-1],final_estimator=model_list[-1][1],cv=10)
    elif ensemble_method == '-VOTE-':
        for i in range(len(para_list)):
            base_model = para_list[i][0]
            base_model_param = para_list[i][1]
            field += [[sg.Text(f'{base_model}'),sg.Text(f"{base_model_param}")]]
            regressor = regressor_dict[base_model]
            regressor.set_params(**base_model_param)
            model_list.append((base_model,regressor))
        layout = [[sg.pin(sg.Frame('',layout=field,expand_x=True, expand_y=True,pad=(0,0)),expand_x=True,expand_y=True,shrink=True),sg.Frame('',layout=[[sg.Button('Train and Predict', size=(20,2))]])]]
        ensemble_model = VotingRegressor(estimators=model_list)


    layout +=[[ sg.Frame('',layout=[[
                sg.Column([[sg.Text('Result')],[sg.Canvas(key='-CANVAS-',size=(400,400),background_color='white')]],expand_x=True,expand_y=True),
                sg.Column([[sg.Text('Train Score')],[sg.Table(values=[],headings=['Metric', 'Score'],auto_size_columns=True,hide_vertical_scroll=True, key='-TABLETRAIN-',expand_x=True)],
                            [sg.VPush()], [sg.Text('Test Score')],[sg.Table(values=[],headings=['Metric', 'Score'],auto_size_columns=True, hide_vertical_scroll=True, expand_x=True,key='-TABLETEST-')],
                            ],expand_x=True,expand_y=True)]],
                            expand_x=True, expand_y=True,visible=True,key = '-CONTROL-')],
                            [sg.Frame('',layout=[[sg.pin(sg.Column([[sg.Button('Explain with SHAP'),sg.Button('Explain with LIME'), sg.Button('View Optimal Parameter'), sg.Button('Save Trained Model'),sg.Button('Save Displayed Result'),sg.Button('Back',button_color='red',pad=(10,10)), sg.Button('Exit',button_color='red',pad=(10,10)),sg.Push()]],expand_x= True,pad =(0,0)),expand_x=True)]])]]

    window = sg.Window(f'{ensemble_method}',layout=layout)

    while True:
        event,values = window.read()

        if event ==sg.WIN_CLOSED:
            main_windows.un_hide()
            break

        if event =='Train and Predict':
            if figure_agg:
                delete_figure_agg(figure_agg)

            ensemble_model.fit(X_train, y_train)


            model_explain = ensemble_model
            y_test_pred = ensemble_model.predict(X_test)
            y_train_pred = ensemble_model.predict(X_train)
            fig = plot_regression_result(y_test,y_test_pred)
            figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

            train_score = [['R2', r2_score(y_train,y_train_pred)],
                            ['MSE',mean_squared_error(y_train,y_train_pred)],
                            ['MAE',mean_absolute_error(y_train,y_train_pred)],
                            ['MAX',max_error(y_train,y_train_pred)],
                            ['RMSE',mean_squared_error(y_train,y_train_pred, squared=False)],
                            ['MAPE',mean_absolute_percentage_error(y_train,y_train_pred)]]

            test_score =[['R2', r2_score(y_test,y_test_pred)],
                        ['MSE',mean_squared_error(y_test,y_test_pred)],
                        ['MAE',mean_absolute_error(y_test,y_test_pred)],
                        ['MAX',max_error(y_test,y_test_pred)],
                        ['RMSE',mean_squared_error(y_test, y_test_pred, squared=False)],
                        ['MAPE',mean_absolute_percentage_error(y_test, y_test_pred)]]

            window['-TABLETRAIN-'].update(values=train_score)
            window['-TABLETEST-'].update(values=test_score)
            window['-CONTROL-'].update(visible=True)

        if event == 'Save Trained Model':
            is_saved = True

            # save model to user_model dic.
            filename = 'users_model/'+ f'{regressor}.joblib'
            joblib.dump(ensemble_model, filename)
            sg.popup('Your model have been saved in Train-ML!')

        if event == 'Save Displayed Result':
            filepath =sg.popup_get_file('Open',no_window=True,save_as=True,file_types=[("PDF","*.pdf"),("PNG","*.png"),("JPG","*.jpg")])
            if filepath:
                fig.savefig(fname = filepath)
                sg.popup('Your model have been saved in Train-ML!')
            else:
                sg.popup_error('The filepath is invalid!')
            filepath = sg.popup_get_file('Save as', no_window=True,save_as= True,default_extension='.csv')
            if filepath:
                with open(filepath,"w",newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Train result'])
                    csvwriter.writerow(['Metric', 'Score'])
                    for row in train_score:
                        csvwriter.writerow(row)
                    csvwriter.writerow(['Test result'])
                    csvwriter.writerow(['Metric', 'Score'])
                    for row in test_score:
                        csvwriter.writerow(row)

        if event == 'View Optimal Parameter':
            param_text = '\n\n'.join(f'{item}' for item in para_list)
            # help_text = '\n\n'.join([f'{key}:{value}' for key,value in para_list.items()])
            sg.popup_scrolled(param_text,title=f'Parameter Tips for {regressor}',text_color='blue')
            # sg.popup(f'Best Parameters:{para_list}')

        if event == 'Explain with SHAP':
            explain_model_shap(model_explain,regressor, X_train,X_test)

        if event == 'Explain with LIME':
            explain_model_lime(model_explain, X_train, X_test)

        if event == 'Help':
            para_text = param_tips[regressor]
            help_text = '\n\n'.join([f'{key}:{value}' for key,value in para_text.items()])
            sg.popup_scrolled(help_text,title=f'Parameter Tips for {regressor}',text_color='blue')

        if event == 'Back':
            if is_saved == False:
                is_yes = sg.popup('You do not save your model. Do you want to save your trained model first?',button_type=sg.POPUP_BUTTONS_YES_NO)
                if is_yes == 'Yes':
                    continue
            window.close()
            main_windows.un_hide()
            main_windows['-NONE_ENSEMBLE-'].update(True)

        if event == 'Exit':
            if is_saved == False:
                is_yes = sg.popup('You do not save your model. Do you want to save your trained model first?',button_type=sg.POPUP_BUTTONS_YES_NO)
                if is_yes == 'Yes':
                    continue

            is_yes = sg.popup('Are you sure to exit the ML Module?',button_type=sg.POPUP_BUTTONS_YES_NO)
            if is_yes == 'Yes':
                window.close()
                main_windows.close()

    window.close()

def singleML_window_classification(main_windows,event,X_train,X_test,y_train,y_test):

    classfier = event
    search_method ='None'

    if classfier == 'RunAll':
        table_test,table_train = method_runall(X_train,X_test,y_train,y_test)
        table_layout_train =sg.Table(values=table_train,headings=metrics,auto_size_columns=True, hide_vertical_scroll=True, expand_x=True,expand_y=True,key='-TABLEALL-')
        table_layout_test =sg.Table(values=table_test,headings=metrics,auto_size_columns=True, hide_vertical_scroll=True, expand_x=True,expand_y=True,key='-TABLEALLTEST-')
        layout = [[sg.Column([[sg.Text('Train result')],[table_layout_train],[sg.Text('Test result')],[ table_layout_test],[sg.pin(sg.Column([[sg.Push(),sg.Button('Back',button_color='red',pad=(10,10)), sg.Button('Exit',button_color='red',pad=(10,10)),sg.Push()]],expand_x= True,pad =(0,0)),expand_x=True)]],expand_x=True,expand_y=True,key='-COLRUNALL-')]]
    else:
        layout = [[sg.Text('Parameter Optimization Method:'),sg.Radio('None', 'OPTIMIZATION_METHOD', key='-NONE_SEARCH-', enable_events=True, default=True),sg.Radio('Grid Search', 'OPTIMIZATION_METHOD', key='-GRID_SEARCH-',enable_events=True), sg.Radio('Random Search', 'OPTIMIZATION_METHOD', key='-RANDOM_SEARCH-',enable_events=True)]]
        layout_none= singleML_None_search_layout(classfier)
        layout_grid= singleML_GridCV_search_layout(classfier)
        layout_random = singleML_RandomCV_search_layout(classfier)

        layout += [[sg.pin(sg.Frame('',layout=[[layout_none],[layout_grid],[layout_random]],expand_x=True, expand_y=True,pad=(0,0)),expand_x=True,expand_y=True,shrink=True),sg.Frame('',layout=[[sg.Button('Train and Predict', size=(20,2))]])]]
        layout +=[[ sg.Frame('',layout=[[
                            sg.Column([[sg.Text('Result')],[sg.Canvas(key='-CANVAS-',size=(600,400),background_color='white')]],expand_x=True,expand_y=True),
                            sg.Column([[sg.Text('Train Score')],[sg.Table(values=[],headings=['Metric', 'Score'],justification='left',auto_size_columns=True,hide_vertical_scroll=True, key='-TABLETRAIN-',expand_x=True)],
                                        [sg.VPush()], [sg.Text('Test Score')],[sg.Table(values=[],headings=['Metric', 'Score'],justification='left',auto_size_columns=True, hide_vertical_scroll=True, expand_x=True,key='-TABLETEST-')],
                                        ],expand_x=True,expand_y=True)]],
                                        expand_x=True, expand_y=True,visible=True,key = '-CONTROL-')],
                                        [sg.Frame('',layout=[[sg.pin(sg.Column([[sg.Button('Explain with SHAP'),sg.Button('Explain with LIME'), sg.Button('View Optimal Parameters'), sg.Button('Save Trained Model'),sg.Button('Save Displayed Result'),sg.Button('Help',button_color='red',pad=(10,10)),sg.Button('Back',button_color='red',pad=(10,10)), sg.Button('Exit',button_color='red',pad=(10,10)),sg.Push()]],expand_x= True,pad =(0,0)),expand_x=True)]])]]

    window = sg.Window(f'{classfier}',layout=layout,finalize=True)
    figure_agg = None
    best_params = None
    model_explain = None
    is_saved = False

    while True:
        event,values = window.read()
        print(event,values)

        if event ==sg.WIN_CLOSED:
            main_windows.un_hide()
            break

        if event == '-GRID_SEARCH-':
            search_method = 'Grid'
            window['-COL_NONE-'].update(visible = False)
            window['-COL_RANDOM-'].update(visible = False)
            window['-COL_GRID-'].update(visible = True)

        if event =='-RANDOM_SEARCH-':
            search_method = 'Random'
            window['-COL_NONE-'].update(visible = False)
            window['-COL_GRID-'].update(visible = False)
            window['-COL_RANDOM-'].update(visible = True)

        if event == '-NONE_SEARCH-':
            search_method = 'None'
            window['-COL_GRID-'].update(visible = False)
            window['-COL_RANDOM-'].update(visible = False)
            window['-COL_NONE-'].update(visible = True)

        if event =='Train and Predict':
            if figure_agg:
                delete_figure_agg(figure_agg)

            param ={}
            if search_method == 'None':
                for key in default_param_grids[classfier].keys():
                    item = values[f'-Param_None_{key}-']
                    item = item.strip()
                    if isinstance(item,int):
                        item = int(item)
                    else:
                        try:
                            if '.' in item or 'e-' in item:
                                item = float(item)
                            else:
                                item = int(item)
                        except ValueError:
                            item = item
                    param[key] = item
            else:
                for key in default_param_grids[classfier].keys():
                    new_values_str = values[f'-Param_Grid_{key}-']
                    new_values_str = new_values_str.strip()
                    new_list=[]
                    if new_values_str.startswith('[') and new_values_str.endswith(']'):
                        new_values_str= new_values_str.split('[')[1:]
                        for  item in new_values_str:
                            item_list = item.split(']')[0].split(',')
                            input_list = [int(x.strip()) for x in item_list if x.strip()]
                            new_list.append(input_list)
                    else:
                        for item in new_values_str.split(','):
                            item = item.strip()
                            try:
                                if '.' in item or 'e-' in item:
                                    new_list.append(float(item))
                                else:
                                    new_list.append(int(item))
                            except ValueError:
                                new_list.append(item)
                    param[key] = new_list

            if search_method =='None':
                clf = classifier_dict[classfier]
                clf.set_params(**param)
            elif search_method == 'Grid':
                clf = GridSearchCV(classifier_dict[classfier],param, cv=5)
            else:
                clf == None

            clf.fit(X_train, y_train)

            if search_method !='None':
                best_params = clf.best_params_
                clf = clf.best_estimator_
            else:
                best_params = clf.get_params()

            model_explain = clf
            y_test_pred = clf.predict(X_test)
            report = classification_report(y_test,y_test_pred)
            sg.popup(report)

            y_score = clf.decision_function(X_test)
            fpr, tpr,_ = roc_curve(y_test,y_score,pos_label=2)

            roc_display=  RocCurveDisplay(fpr=fpr, tpr=tpr)
            roc_display.plot()
            plt.show()

            cm = confusion_matrix(y_test,y_test_pred)
            disp = ConfusionMatrixDisplay(cm,display_labels=clf.classes_)
            disp.plot()
            plt.show()

            train_sizes, train_scores, valid_scores = learning_curve(clf,X_train,y_train,train_sizes=np.linspace(0.1,1.0,10),cv=5)
            train_scores_mean = np.mean(train_scores,axis=1)
            train_scores_std = np.std(train_scores,axis=1)
            valid_scores_mean = np.mean(valid_scores,axis=1)
            valid_scores_std =  np.std(valid_scores,axis=1)

            # plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_scores_mean, label='Training Score', color='blue')
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='blue')
            plt.plot(train_sizes, valid_scores_mean, label='Cross-Validation Score', color='orange')
            plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='orange')
            plt.xlabel('Training Examples')
            plt.ylabel('Score')
            plt.title('Learning Curve')
            plt.legend(loc='best')
            plt.grid()
            plt.show()



            # fig = plot_regression_result(y_test,y_test_pred)
            # figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

            # train_score = [['R2', r2_score(y_train,y_train_pred)],
            #                 ['MSE',mean_squared_error(y_train,y_train_pred)],
            #                 ['MAE',mean_absolute_error(y_train,y_train_pred)],
            #                 ['MAX',max_error(y_train,y_train_pred)],
            #                 ['RMSE',mean_squared_error(y_train,y_train_pred, squared=False)],
            #                 ['MAPE',mean_absolute_percentage_error(y_train,y_train_pred)]]

            # test_score =[['R2', r2_score(y_test,y_test_pred)],
            #             ['MSE',mean_squared_error(y_test,y_test_pred)],
            #             ['MAE',mean_absolute_error(y_test,y_test_pred)],
            #             ['MAX',max_error(y_test,y_test_pred)],
            #             ['RMSE',mean_squared_error(y_test, y_test_pred, squared=False)],
            #             ['MAPE',mean_absolute_percentage_error(y_test, y_test_pred)]]

            # window['-TABLETRAIN-'].update(values = train_score)
            # window['-TABLETEST-'].update(values = test_score)
            # window['-CONTROL-'].update(visible = True)

        # save image.
        if event == 'Save Trained Model':
            is_saved = True

            # save model to user_model dic.
            filename = 'users_model/'+ f'{classfier}.joblib'
            joblib.dump(clf, filename)
            sg.popup('Your model have been saved in Train-ML!')

        if event == 'Save Displayed Result':
            filepath =sg.popup_get_file('Open',no_window=True,save_as=True,file_types=[("PDF","*.pdf"),("PNG","*.png"),("JPG","*.jpg")])
            # if filepath:
            #     fig.savefig(fname = filepath)
            #     sg.popup('Your model have been saved in Train-ML!')
            # else:
            #     sg.popup_error('The filepath is invalid!')
            # filepath = sg.popup_get_file('Save as', no_window=True,save_as= True,default_extension='.csv')
            # if filepath:
            #     with open(filepath,"w",newline="") as csvfile:
            #         csvwriter = csv.writer(csvfile)
            #         csvwriter.writerow(['Train result'])
            #         csvwriter.writerow(['Metric', 'Score'])
            #         for row in train_score:
            #             csvwriter.writerow(row)
            #         csvwriter.writerow(['Test result'])
            #         csvwriter.writerow(['Metric', 'Score'])
            #         for row in test_score:
            #             csvwriter.writerow(row)

        if event == 'View Optimal Parameters':
            sg.popup(f'Best Parameters:{best_params}')

        if event == 'Explain with SHAP':
            explain_model_shap(model_explain,classfier, X_train,X_test)

        if event == 'Explain with LIME':
            explain_model_lime(model_explain, X_train, X_test)

        if event == 'Help':
            para_text = param_tips[classfier]
            help_text = '\n\n'.join([f'{key}:{value}' for key,value in para_text.items()])
            sg.popup_scrolled(help_text,title=f'Parameter Tips for {classfier}',text_color='blue')

        if event == 'Back':
            if is_saved == False:
                is_yes = sg.popup('You do not save your model. Do you want to save your trained model first?',button_type=sg.POPUP_BUTTONS_YES_NO)
                if is_yes == 'Yes':
                    continue
            window.close()
            main_windows.un_hide()
            main_windows['-NONE_ENSEMBLE-'].update(True)

        if event == 'Exit':
            if is_saved == False:
                is_yes = sg.popup('You do not save your model. Do you want to save your trained model first?',button_type=sg.POPUP_BUTTONS_YES_NO)
                if is_yes == 'Yes':
                    continue

            is_yes = sg.popup('Are you sure to exit the ML Module?',button_type=sg.POPUP_BUTTONS_YES_NO)
            if is_yes == 'Yes':
                window.close()
                main_windows.close()

    window.close()


def ensembelML_window_classification(main_windows,event,para_list,X_train,X_test,y_train,y_test):
    ensemble_method = event
    model_list =[]
    field =[]
    figure_agg = None
    is_saved = False

    if ensemble_method == '-BAG-':
        base_model = para_list[0][0]
        base_model_param = para_list[0][1]
        layout = [[sg.pin(sg.Frame('',layout=[[sg.Text(f'{base_model}'),sg.Text(f"{base_model_param}")]],expand_x=True, expand_y=True,pad=(0,0)),expand_x=True,expand_y=True,shrink=True),sg.Frame('',layout=[[sg.Button('Train and Predict', size=(20,2))]])]]
        classifier = classifier_dict[base_model]
        classifier.set_params(**base_model_param)
        ensemble_model = BaggingClassifier(estimator=classifier,
                        n_estimators=10, random_state=0)
    elif ensemble_method =='-STACK-':
        for i in range(len(para_list)):
            base_model = para_list[i][0]
            base_model_param = para_list[i][1]
            field += [[sg.Text(f'{base_model}'),sg.Text(f"{base_model_param}")]]
            classifier = classifier_dict[base_model]
            classifier.set_params(**base_model_param)
            model_list.append((base_model,classifier))
        layout = [[sg.pin(sg.Frame('',layout=field,expand_x=True, expand_y=True,pad=(0,0)),expand_x=True,expand_y=True,shrink=True),sg.Frame('',layout=[[sg.Button('Train and Predict', size=(20,2))]])]]
        ensemble_model = StackingClassifier(estimators=model_list[:-1],final_estimator=model_list[-1][1],cv=10)
    elif ensemble_method == '-VOTE-':
        for i in range(len(para_list)):
            base_model = para_list[i][0]
            base_model_param = para_list[i][1]
            field += [[sg.Text(f'{base_model}'),sg.Text(f"{base_model_param}")]]
            classifier = classifier_dict[base_model]
            classifier.set_params(**base_model_param)
            model_list.append((base_model,classifier))
        layout += [[sg.pin(sg.Frame('',layout=field,expand_x=True, expand_y=True,pad=(0,0)),expand_x=True,expand_y=True,shrink=True),sg.Frame('',layout=[[sg.Button('Train and Predict', size=(20,2))]])]]
        ensemble_model = VotingClassifier(estimators=model_list)

    layout +=[[ sg.Frame('',layout=[[
            sg.Column([[sg.Text('Result')],[sg.Canvas(key='-CANVAS-',size=(400,400),background_color='white')]],expand_x=True,expand_y=True),
            sg.Column([[sg.Text('Train Score')],[sg.Table(values=[],headings=['Metric', 'Score'],auto_size_columns=True,hide_vertical_scroll=True, key='-TABLETRAIN-',expand_x=True)],
                        [sg.VPush()], [sg.Text('Test Score')],[sg.Table(values=[],headings=['Metric', 'Score'],auto_size_columns=True, hide_vertical_scroll=True, expand_x=True,key='-TABLETEST-')],
                        ],expand_x=True,expand_y=True)]],
                        expand_x=True, expand_y=True,visible=True,key = '-CONTROL-')],
                        [sg.Frame('',layout=[[sg.pin(sg.Column([[sg.Button('Explain with SHAP'),sg.Button('Explain with LIME'), sg.Button('View Optimal Parameter'), sg.Button('Save Trained Model'),sg.Button('Save Displayed Result'),sg.Button('Back',button_color='red',pad=(10,10)), sg.Button('Exit',button_color='red',pad=(10,10)),sg.Push()]],expand_x= True,pad =(0,0)),expand_x=True)]])]]

    window = sg.Window(f'{ensemble_method}',layout=layout)

    while True:
        event,values = window.read()

        if event ==sg.WIN_CLOSED:
            main_windows.un_hide()
            break

        if event =='Train and Predict':
            if figure_agg:
                delete_figure_agg(figure_agg)

            ensemble_model.fit(X_train, y_train)


            model_explain = ensemble_model

            y_test_pred = ensemble_model.predict(X_test)
            report = classification_report(y_test,y_test_pred)
            sg.popup(report)

            y_score = ensemble_model.decision_function(X_test)
            fpr, tpr,_ = roc_curve(y_test,y_score,pos_label=2)

            roc_display=  RocCurveDisplay(fpr=fpr, tpr=tpr)
            roc_display.plot()
            plt.show()

            cm = confusion_matrix(y_test,y_test_pred)
            disp = ConfusionMatrixDisplay(cm,display_labels=ensemble_model.classes_)
            disp.plot()
            plt.show()

            train_sizes, train_scores, valid_scores = learning_curve(ensemble_model,X_train,y_train,train_sizes=np.linspace(0.1,1.0,10),cv=5)
            train_scores_mean = np.mean(train_scores,axis=1)
            train_scores_std = np.std(train_scores,axis=1)
            valid_scores_mean = np.mean(valid_scores,axis=1)
            valid_scores_std =  np.std(valid_scores,axis=1)

            # plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_scores_mean, label='Training Score', color='blue')
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='blue')
            plt.plot(train_sizes, valid_scores_mean, label='Cross-Validation Score', color='orange')
            plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='orange')
            plt.xlabel('Training Examples')
            plt.ylabel('Score')
            plt.title('Learning Curve')
            plt.legend(loc='best')
            plt.grid()
            plt.show()
            # y_test_pred = ensemble_model.predict(X_test)
            # y_train_pred = ensemble_model.predict(X_train)

            # fig = plot_regression_result(y_test,y_test_pred)
            # figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

            # train_score = [['R2', r2_score(y_train,y_train_pred)],
            #                 ['MSE',mean_squared_error(y_train,y_train_pred)],
            #                 ['MAE',mean_absolute_error(y_train,y_train_pred)],
            #                 ['MAX',max_error(y_train,y_train_pred)],
            #                 ['RMSE',mean_squared_error(y_train,y_train_pred, squared=False)],
            #                 ['MAPE',mean_absolute_percentage_error(y_train,y_train_pred)]]

            # test_score =[['R2', r2_score(y_test,y_test_pred)],
            #             ['MSE',mean_squared_error(y_test,y_test_pred)],
            #             ['MAE',mean_absolute_error(y_test,y_test_pred)],
            #             ['MAX',max_error(y_test,y_test_pred)],
            #             ['RMSE',mean_squared_error(y_test, y_test_pred, squared=False)],
            #             ['MAPE',mean_absolute_percentage_error(y_test, y_test_pred)]]

            # window['-TABLETRAIN-'].update(values=train_score)
            # window['-TABLETEST-'].update(values=test_score)
            # window['-CONTROL-'].update(visible=True)

        if event == 'Save Trained Model':
            is_saved = True

            # save model to user_model dic.
            filename = 'users_model/'+ f'{ensemble_method}.joblib'
            joblib.dump(ensemble_model, filename)
            sg.popup('Your model have been saved in Train-ML!')

        if event == 'Save Displayed Result':
            pass
            # filepath =sg.popup_get_file('Open',no_window=True,save_as=True,file_types=[("PDF","*.pdf"),("PNG","*.png"),("JPG","*.jpg")])
            # if filepath:
            #     fig.savefig(fname = filepath)
            #     sg.popup('Your model have been saved in Train-ML!')
            # else:
            #     sg.popup_error('The filepath is invalid!')
            # filepath = sg.popup_get_file('Save as', no_window=True,save_as= True,default_extension='.csv')
            # if filepath:
            #     with open(filepath,"w",newline="") as csvfile:
            #         csvwriter = csv.writer(csvfile)
            #         csvwriter.writerow(['Train result'])
            #         csvwriter.writerow(['Metric', 'Score'])
            #         for row in train_score:
            #             csvwriter.writerow(row)
            #         csvwriter.writerow(['Test result'])
            #         csvwriter.writerow(['Metric', 'Score'])
            #         for row in test_score:
            #             csvwriter.writerow(row)

        if event == 'View Optimal Parameter':
            param_text = '\n\n'.join(f'{item}' for item in para_list)
            # help_text = '\n\n'.join([f'{key}:{value}' for key,value in para_list.items()])
            sg.popup_scrolled(param_text,title=f'Parameter Tips for {classifier}',text_color='blue')
            # sg.popup(f'Best Parameters:{para_list}')

        if event == 'Explain with SHAP':
            explain_model_shap(model_explain,classifier, X_train,X_test)

        if event == 'Explain with LIME':
            explain_model_lime(model_explain, X_train, X_test)

        if event == 'Help':
            para_text = param_tips[classifier]
            help_text = '\n\n'.join([f'{key}:{value}' for key,value in para_text.items()])
            sg.popup_scrolled(help_text,title=f'Parameter Tips for {classifier}',text_color='blue')

        if event == 'Back':
            if is_saved == False:
                is_yes = sg.popup('You do not save your model. Do you want to save your trained model first?',button_type=sg.POPUP_BUTTONS_YES_NO)
                if is_yes == 'Yes':
                    continue
            window.close()
            main_windows.un_hide()
            main_windows['-NONE_ENSEMBLE-'].update(True)

        if event == 'Exit':
            if is_saved == False:
                is_yes = sg.popup('You do not save your model. Do you want to save your trained model first?',button_type=sg.POPUP_BUTTONS_YES_NO)
                if is_yes == 'Yes':
                    continue

            is_yes = sg.popup('Are you sure to exit the ML Module?',button_type=sg.POPUP_BUTTONS_YES_NO)
            if is_yes == 'Yes':
                window.close()
                main_windows.close()

    window.close()



######### Main windows related functions ##############

def create_main_layout():

    regression_layout = []
    row = []
    for button_name in regressor_dict.keys():
        if button_name == 'RunAll':
            row.append(sg.Button(button_name,key='-RUNALL_R-', size =(20,2), button_color='green',expand_x=True))
        else:
            row.append(sg.Button(button_name, size =(20,2),expand_x =True))
        if len(row) == 6 :
            regression_layout.append(row)
            row =[]
    if row:
        extra_space = 6-len(row)
        for i in range(extra_space):
            row.append(sg.Text('',size =(20,2),expand_x =True))
        regression_layout.append(row)

    class_layout = []
    row = []
    for button_name in classifier_dict.keys():
        if button_name == 'RunAll':
            row.append(sg.Button(button_name, key='-RUNALL_C-', size =(20,2), button_color='green',expand_x=True))
        else:
            row.append(sg.Button(button_name, size =(20,2),expand_x=True))
        if len(row) == 6 :
            class_layout.append(row)
            row =[]
    if row:
        extra_space = 6-len(row)
        for i in range(extra_space):
            row.append(sg.Text('',size =(20,2),expand_x =True))
        class_layout.append(row)


    layout = [
              [sg.Text('Dataset:',justification='right'),sg.Input(size=(50, 1), key='-FILEPATH-',expand_x= True,readonly=True), sg.B('Load Data')],
              [sg.Text('Select Features:',justification='right'),sg.Listbox(values=[], expand_x=True, key='-LISTBOXFEA-',select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,size=(30,4)),
               sg.Text('Select Label:',justification='left'),sg.Listbox(values=[], expand_x=True, key='-LISTBOXLAB-',select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,size=(30,4))],
            #  [sg.Text('Select Evaluation method:'),sg.Radio('Single Train-Test Split', group_id='-EVAL_METHOD-',default=True,key='-SPLIT-',),sg.Radio('K-fold Cross-Validation',group_id='-EVAL_METHOD-',key='-K_FOLD-')],
             [sg.Text('Test Size:'),sg.Slider(range=(0,1),default_value=0.3,resolution=0.1,orientation='h', expand_x=True, key='-SKIDER-')],
              [sg.Text('Select Ensemble Method:',justification='right'),sg.Radio('None', group_id='ENSEMBLE_METHOD', key='-NONE_ENSEMBLE-', default=True),
               sg.Radio('Stacking',group_id='ENSEMBLE_METHOD', key = '-STACK-',enable_events=True), sg.Radio('Voting', group_id='ENSEMBLE_METHOD', key = '-VOTE-',enable_events=True),
               sg.Radio('Bagging',group_id='ENSEMBLE_METHOD', key = '-BAG-',enable_events=True)],
              [sg.Frame('Regression', layout=regression_layout,expand_x= True, expand_y= True, key='-FRAMEREGR-')],
              [sg.Frame('Classfication', layout=class_layout,expand_x= True, expand_y= True, key='-FRAMEREGR2-')],
              ]
    window= sg.Window('Machine Learning:', layout = layout,finalize=True,  resizable=True, use_default_focus=False)
    return window

def main():
    window = create_main_layout()
    n_base_estimitors = 1
    filepath =''
    pre_event = None

    while True:
        event,values = window.read()
        print(event,values)

        if event ==sg.WIN_CLOSED:
            break

        if event == 'Load Data':
            window['-FILEPATH-'].update(value='')
            window['-LISTBOXFEA-'].update(values = [])
            window['-LISTBOXLAB-'].update(values=[])
            window['-NONE_ENSEMBLE-'].update(True)

            filepath = sg.popup_get_file('Select a CSV file',file_types=[("CVS File","*.csv")],no_window=True)
            # check if the data path is valid.
            if filepath !='':
                pre_fea =[]
                current_fea =[]
                pre_label =[]
                current_label =[]
                pre_split =[]
                current_split =[]
                window['-FILEPATH-'].update(value = filepath)
                ##---------------- read data-----------------##
                # load dataset
                df_raw = pd.DataFrame(pd.read_csv(filepath))
                # read column head
                df_colname = list(df_raw.columns)

                ##---------------- Update the features and labels in layout-----------------##
                # df_colname.insert(0,'All')
                window['-LISTBOXFEA-'].update(values = df_colname)
                window['-LISTBOXLAB-'].update(values= df_colname)
                sg.popup('Data Loaded Successfully')
            else:
                sg.popup('Filepath is incorrect!')
        else:
            if filepath =='':
                window['-NONE_ENSEMBLE-'].update(True)
                sg.popup('Please upload dataset!')
                continue

            current_label = values['-LISTBOXLAB-']
            current_fea = values['-LISTBOXFEA-']
            current_split = values['-SKIDER-']

            if (current_split == pre_split and current_fea == pre_fea and current_label == pre_label): # If nothing change, the dataset does not change
                    pass
            else:
                if current_label != [] and current_fea !=[] :
                    pre_fea = current_fea
                    pre_label = current_label
                    pre_split = current_split
                    X = df_raw[current_fea]
                    y = df_raw[current_label]
                    # Check if there is NA and String
                    missing_value = y.isna().sum().sum() + X.isna().sum().sum()
                    if  missing_value>0:
                        sg.popup(f'There are {missing_value} missing values in this dataset, please upload dataset without misisng values.')

                        continue

                    constring = X.select_dtypes(include=['object']).shape[1]>0
                    if constring>0:
                        sg.popup('There are non-numeric data in this dataset.')
                        continue

                    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=current_split,random_state=42)
                else:
                    sg.popup_error('Please select label and feature')
                    window['-NONE_ENSEMBLE-'].update(True)
                    continue


            if event == '-STACK-' or event =='-VOTE-' or event == '-BAG-':
                num_model = sg.popup_get_text('How many base models in this ensemble ML? ')
                if num_model == None:
                    window['-NONE_ENSEMBLE-'].update(True)
                    window[event].update(False)
                else:
                    num_model = int(num_model)
                    if num_model>1:
                        n_base_estimitors = num_model
                        model_list =[]
                    else:
                        window['-NONE_ENSEMBLE-'].update(True)
                        window[event].update(False)
                        sg.popup('please re-input!')

            if event in regressor_dict:
                if values['-NONE_ENSEMBLE-']:
                        method = event
                        window.hide()
                        singleML_window_regression(window,method,X_train,X_test,y_train,y_test)
                elif values['-VOTE-']:
                    # if pre_event == None:
                    #     pre_event = 'Regression'
                    # elif pre_event == 'Classification':
                    #     sg.popup('If you ')
                    #     continue

                    if event == 'LinearRegression':
                        layout =[]
                    else:
                        param_grid = default_param_grids[event]
                        layout = create_param_input_fields(param_grid,0)
                    layout += [[sg.Button('Add'),sg.Button('Cancel')]]
                    window_model = sg.Window(f'{event}',layout)
                    event_model,value_model = window_model.read(close=True)

                    if event_model == 'Add':
                        if len(model_list) == n_base_estimitors:
                            sg.popup(f'You have {n_base_estimitors}.')
                            model_text = '\n\n'.join(f'{item}' for item in model_list)
                            ensemble_ok = sg.popup_scrolled(model_text,title='Ensemble_Stack models',yes_no=True)
                            if ensemble_ok == 'Yes':
                                ensembelML_window_regression(window,'-VOTE-',model_list,X_train,X_test,y_train,y_test)
                            else:
                                model_list = []
                                sg.popup(f'All your base-models have been removed, please select{n_base_estimitors}!')
                        else:
                            param={}
                            if event != 'LinearRegression':
                                for key in default_param_grids[event].keys():
                                    item = value_model[f'-Param_None_{key}-']
                                    try:
                                        if '.' in item:
                                            item = float(item)
                                        else:
                                            item = int(item)
                                    except ValueError:
                                        item = item
                                    param[key] = item


                            model_list.append([event,param])
                            if len(model_list) == n_base_estimitors:
                                model_text = '\n\n'.join(f'{item}' for item in model_list)
                                ensemble_ok = sg.popup_scrolled(model_text,title='Ensemble_Stack models',yes_no=True)
                                if ensemble_ok == 'Yes':
                                    ensembelML_window_regression(window,'-VOTE-',model_list,X_train,X_test,y_train,y_test)
                                else:
                                    model_list = []
                                    sg.popup(f'All your base-models have been removed, please select{n_base_estimitors}!')

                elif values['-STACK-']:
                    if event == 'LinearRegression':
                        layout =[]
                    else:
                        param_grid = default_param_grids[event]
                        layout = create_param_input_fields(param_grid,0)
                    layout += [[sg.Button('Add'),sg.Button('Cancel')]]
                    window_model = sg.Window(f'{event}',layout)
                    event_model,value_model = window_model.read(close=True)
                    if event_model == 'Add':
                        if len(model_list) == n_base_estimitors:
                            sg.popup(f'You have {n_base_estimitors} models added, this one cannot be added.')
                            model_text = '\n\n'.join(f'{item}' for item in model_list)
                            ensemble_ok = sg.popup_scrolled(model_text,title='Ensemble_Stack models',yes_no=True)
                            if ensemble_ok == 'Yes':
                                ensembelML_window_regression(window, '-STACK-',model_list,X_train,X_test,y_train,y_test)
                            else:
                                model_list = []
                                sg.popup(f'All your base-models have been removed, please select{n_base_estimitors}!')
                        else:
                            param={}
                            if event != 'LinearRegression':
                                for key in default_param_grids[event].keys():
                                    item = value_model[f'-Param_None_{key}-']
                                    try:
                                        if '.' in item or 'e-' in item:
                                            item = float(item)
                                        else:
                                            item = int(item)
                                    except ValueError:
                                        item = item
                                    param[key] = item

                            model_list.append([event,param])
                            if len(model_list) == n_base_estimitors:
                                model_text = '\n\n'.join(f'{item}' for item in model_list)
                                ensemble_ok = sg.popup_scrolled(model_text,title='Ensemble_Stack models',yes_no=True)
                                if ensemble_ok == 'Yes':
                                    ensembelML_window_regression(window, '-STACK-',model_list,X_train,X_test,y_train,y_test)
                                else:
                                    model_list = []
                                    sg.popup(f'All your base-models have been removed, please select{n_base_estimitors}!')

                elif values['-BAG-']:
                    model_list =[]
                    if event == 'LinearRegression':
                        layout =[]
                    else:
                        param_grid = default_param_grids[event]
                        layout = create_param_input_fields(param_grid,0)
                    layout += [[sg.Button('Add'),sg.Button('Cancel')]]
                    window_model = sg.Window(f'{event}',layout)
                    event_model,value_model = window_model.read(close=True)

                    param={}
                    if event != 'LinearRegression':
                        for key in default_param_grids[event].keys():
                            item = value_model[f'-Param_None_{key}-']
                            try:
                                if '.' in item or 'e-' in item:
                                    item = float(item)
                                else:
                                    item = int(item)
                            except ValueError:
                                item = item
                            param[key] = item

                    model_list.append([event,param])
                    model_text = '\n\n'.join(f'{item}' for item in model_list)
                    ensemble_ok = sg.popup_scrolled(model_text,title='Ensemble_Stack models',yes_no=True)
                    if ensemble_ok == 'Yes':
                        ensembelML_window_regression(window, '-BAG-',model_list,X_train,X_test,y_train,y_test,True)
                    else:
                        model_list = []
                        sg.popup(f'All your base-models have been removed, please select{n_base_estimitors}!')


            if event in classifier_dict:
                if values['-NONE_ENSEMBLE-']:
                        method = event
                        window.hide()
                        singleML_window_classification(window,method,X_train,X_test,y_train,y_test)
                elif values['-VOTE-']:
                    param_grid = default_param_grids[event]
                    layout = create_param_input_fields(param_grid,0)
                    layout += [[sg.Button('Add'),sg.Button('Cancel')]]
                    window_model = sg.Window(f'{event}',layout)
                    event_model,value_model = window_model.read(close=True)
                    if event_model == 'Add':
                        if len(model_list) == n_base_estimitors:
                            sg.popup(f'You have {n_base_estimitors}.')
                            model_text = '\n\n'.join(f'{item}' for item in model_list)
                            ensemble_ok = sg.popup_scrolled(model_text,title='Ensemble_Stack models',yes_no=True)
                            if ensemble_ok == 'Yes':
                                ensembelML_window_classification(window,'-VOTE-',model_list,X_train,X_test,y_train,y_test)
                            else:
                                model_list = []
                                sg.popup(f'All your base-models have been removed, please select{n_base_estimitors}!')
                        else:
                            param={}
                            for key in default_param_grids[event].keys():
                                item = value_model[f'-Param_None_{key}-']
                                try:
                                    if '.' in item:
                                        item = float(item)
                                    else:
                                        item = int(item)
                                except ValueError:
                                    item = item
                                param[key] = item


                            model_list.append([event,param])
                            if len(model_list) == n_base_estimitors:
                                model_text = '\n\n'.join(f'{item}' for item in model_list)
                                ensemble_ok = sg.popup_scrolled(model_text,title='Ensemble_Stack models',yes_no=True)
                                if ensemble_ok == 'Yes':
                                    ensembelML_window_classification(window,'-VOTE-',model_list,X_train,X_test,y_train,y_test)
                                else:
                                    model_list = []
                                    sg.popup(f'All your base-models have been removed, please select{n_base_estimitors}!')

                elif values['-STACK-']:
                    param_grid = default_param_grids[event]
                    layout = create_param_input_fields(param_grid,0)
                    layout += [[sg.Button('Add'),sg.Button('Cancel')]]
                    window_model = sg.Window(f'{event}',layout)
                    event_model,value_model = window_model.read(close=True)
                    if event_model == 'Add':
                        if len(model_list) == n_base_estimitors:
                            sg.popup(f'You have {n_base_estimitors} models added, this one cannot be added.')
                            model_text = '\n\n'.join(f'{item}' for item in model_list)
                            ensemble_ok = sg.popup_scrolled(model_text,title='Ensemble_Stack models',yes_no=True)
                            if ensemble_ok == 'Yes':
                                ensembelML_window_classification(window, '-STACK-',model_list,X_train,X_test,y_train,y_test)
                            else:
                                model_list = []
                                sg.popup(f'All your base-models have been removed, please select{n_base_estimitors}!')
                        else:
                            param={}
                            for key in default_param_grids[event].keys():
                                item = value_model[f'-Param_None_{key}-']
                                try:
                                    if '.' in item or 'e-' in item:
                                        item = float(item)
                                    else:
                                        item = int(item)
                                except ValueError:
                                    item = item
                                param[key] = item

                            model_list.append([event,param])
                            if len(model_list) == n_base_estimitors:
                                model_text = '\n\n'.join(f'{item}' for item in model_list)
                                ensemble_ok = sg.popup_scrolled(model_text,title='Ensemble_Stack models',yes_no=True)
                                if ensemble_ok == 'Yes':
                                    ensembelML_window_classification(window, '-STACK-',model_list,X_train,X_test,y_train,y_test)
                                else:
                                    model_list = []
                                    sg.popup(f'All your base-models have been removed, please select{n_base_estimitors}!')

                elif values['-BAG-']:
                    model_list =[]
                    param_grid = default_param_grids[event]
                    layout = create_param_input_fields(param_grid,0)
                    layout += [[sg.Button('Add'),sg.Button('Cancel')]]
                    window_model = sg.Window(f'{event}',layout)
                    event_model,value_model = window_model.read(close=True)

                    param={}
                    for key in default_param_grids[event].keys():
                        item = value_model[f'-Param_None_{key}-']
                        try:
                            if '.' in item or 'e-' in item:
                                item = float(item)
                            else:
                                item = int(item)
                        except ValueError:
                            item = item
                        param[key] = item

                    model_list.append([event,param])
                    model_text = '\n\n'.join(f'{item}' for item in model_list)
                    ensemble_ok = sg.popup_scrolled(model_text,title='Ensemble_Stack models',yes_no=True)
                    if ensemble_ok == 'Yes':
                        ensembelML_window_classification(window, '-BAG-',model_list,X_train,X_test,y_train,y_test,False)
                    else:
                        model_list = []
                        sg.popup(f'All your base-models have been removed, please select{n_base_estimitors}!')


    window.close()


if __name__ == "__main__":
    main()
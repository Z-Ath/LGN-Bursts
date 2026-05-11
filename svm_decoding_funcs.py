#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:52:36 2026

@author: loganbecker
"""

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# --- Functions used to build and fit LDAs ---
def build_lda(X, labels, svm_type, metric_type='acc', params=None,
              run_search=True, dual='auto', random_state=6895):
    
    """
    Builds a LDA model given criteria such as linearity, metric type, dual use. Will run a hyperparameter search to get 
    the control parameter C
    
    Parameters
    -----------
    X: float array of size [TxN] where T is the number of trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a trial associate with each neuron)
    labels: float array of size T where T is the number of trials 
        Design Matrix labels (one value per row, used as the targets)
    svm_type: string ('linear','nonlinear','logistic')
        Designates the type of LDA to build
    metric_type: string (DEFAULT - 'acc' or 'roc')
        Designates the type of scoring system to use when hyperparamter testing    
    params: dict from hyperparam_search (e.g. {'C': 1.0, 'gamma': 0.01}) Default = None
        Parameters to use for hyperparameter fitting
    run_serach: Bool - DEFAULT TRUE
        Boolean to flag if to do a hyperparameter search
    dual: str
        Used to dictate method for the LinearSVC. May want True for SPARSE data 
        
    Return
    -------
    Model: LDA model to use for fitting
    
    """
    
    # --- Hyperparameter Fitting ---
    if params is None:
        if run_search:
            params = hyperparam_search(X, labels, metric_type=metric_type, svm_type=svm_type)
        else:
            params = {}  # fall back to defaults

    # Defaults or estimated
    C = params.get('C', 1.0)
    gamma = params.get('gamma', 'scale')

    # --- Choose model and build LDA ---
    # Linear
    if svm_type == 'linear':
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(C=C, dual=dual,max_iter=5000,tol=1e-3,random_state=random_state)
            )

    # Nonlinear
    elif svm_type == 'nonlinear':
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf',C=C,gamma=gamma,random_state=random_state)
            )

    # Logistic
    elif svm_type == 'logistic':
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C,solver='saga',max_iter=10000,
                class_weight='balanced',tol=1e-3,random_state=random_state)
            )

    # Raise error if none
    else:
        raise ValueError("svm_type must be 'linear', 'nonlinear', or 'logistic'")

    return model

def hyperparam_search(X, labels, metric_type,svm_type,
                          C_grid=[0.001, 0.01, 0.1, 1.0],
                          gamma_grid=['scale', 'auto', 0.001, 0.01, 0.1],
                          n_splits=5, n_classes=8, dual = 'auto', n_jobs = -1, random_state=6895):
    
    """
    Builds and runs a LDA hyperparameter Grid Search given the model type and a range of C and gamma (if needed) values
    
    Paramters
    ----------
    X: float array of size [TxN] where T is the number of trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a trial associate with each neuron)
    labels: float array of size T where T is the number of trials
        Design Matrix labels (one value per row, used as the targets)
    metric_type: string (DEFAULT - 'acc' or 'roc')
        Designates the type of scoring system to use when hyperparamter testing    
    svm_type: string ('linear','nonlinear','logistic')
        Designates the type of LDA to build
    C_grid: list of floats 
        LDA C-hyperparameter values to test (DEFAULT [0.001,0.01,0.1,1])
    gamma_grid: list of strings and floats
        nonlinear gamma-hyperparameter values to test (DEFAULT ['scale','auto',0.001,0.01,0.1])
    n_splits: int
        Number of Cross-validation fits
    n_classes: int
        Number of unique labels (default to 8 for the paper)
    dual: str
        Used to dictate method for the LinearSVC. May want True for SPARSE data 
    n_jobs: int
        Used for CPU use allocation and speed (-1 = use all available drives)
        
    Return
    ------
    output: Dictionary with hyperparameters (depends on model type but atleast C)
    
    """
    
    # ---Equalize the training set --- 
    labels = labels.astype(int) # Make sure labels are integers
    min_class_count = np.min(np.bincount(labels)) # Get min class size
    n_splits = min(n_splits, min_class_count) # Make sure splits >= min class
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # --- Choose scoring ---
    if metric_type == 'acc': scoring = 'accuracy'
    elif metric_type == 'roc': scoring = roc_auc_decision_scorer  # ROC scorer 
    else:
        raise ValueError("metric_type must be 'acc' or 'roc'")
        
    # --- Build PIPELINE
    # Linear
    if svm_type == 'linear':
        pipeline = make_pipeline(
            StandardScaler(),
            LinearSVC(dual=dual, max_iter=5000, random_state=6895))
        param_grid = {'linearsvc__C': C_grid}

    # Nonlinaer
    elif svm_type == 'nonlinear':
        pipeline = make_pipeline(
            StandardScaler(),SVC(kernel='rbf', random_state=6895))

        param_grid = {'svc__C': C_grid,'svc__gamma': gamma_grid}
        
    # Logistic
    elif svm_type == 'logistic':
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver='saga',max_iter=10000,class_weight='balanced',random_state=6895))
        param_grid = {'logisticregression__C': C_grid}
    else:
        raise ValueError("svm_type must be 'linear', 'nonlinear', or 'logistic'")

    # For ROC specifically, want a 1 vs rest classifier
    if metric_type == 'roc':
        pipeline = make_pipeline(StandardScaler(),OneVsRestClassifier(pipeline))

        # update param grid keys
        param_grid = {f"onevsrestclassifier__estimator__{k}": v for k, v in param_grid.items()}
        
    # Build grid search and fit
    grid = GridSearchCV(pipeline,param_grid,cv=cv,scoring=scoring,n_jobs=n_jobs)
    grid.fit(X, labels)
    best_params = grid.best_params_

    # Build output
    output = {}
    for key, val in best_params.items():
        if key.endswith('__C'):
            output['C'] = val
        elif key.endswith('__gamma'):
            output['gamma'] = val

    return output

def roc_auc_decision_scorer(estimator, X, y):
    """
    Create a scorere for ROC AUC and decides choice. Used as function input to GridSearchCV

    Parameters
    ----------
    estimator: Function
    X: float array of size [TxN] where T is the number of trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a trial associate with each neuron)
    Y: float array of size T where T is the number of trials
        Design Matrix labels (one value per row, used as the targets)
        
    Returns
    -------
    roc_auc_score: array float
       ROC AUC SCOREs

    """
    
    classes = np.unique(y) # Get unique classes
    y_bin = label_binarize(y, classes=classes) # Convert to binarization
    y_scores = estimator.decision_function(X) # Get scores given estimator
    return roc_auc_score(y_bin, y_scores, average='macro', multi_class='ovr')



def run_lda(lda, x_train, x_test, y_train, y_test, x_transfer = None, y_transfer = None):
    """
    Function to to fit LDA. Trains on x_train, tests on x_test. Will also test on some other data set such as x_transfer
    based on the LDA trained to x_train
    
    Parameters
    ----------
    lda: function
        lda model (build from build_lda)
    x_train: float array of size [T_trainxN] where T_train is the number of training trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a training trial associate with each neuron)
    x_test: float array of size [T_testxN] where T_test is the number of test trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a test trial associate with each neuron)
    y_train: float array of size T_train where T_train is the number of training trials
        Design Matrix labels (one value per row, used as the targets)
    y_test: float array of size T_test where T_test is the number of test trials
        Design Matrix labels (one value per row, used as the targets)
    x_transfer: float array of size [T_newxN] where T_new is the number of new trials and N is the number of neurons,
        Design matrix for a seperate condition to test model on (for example, train on familiar, but test on deviants)
        Default = None. Only need if testing on a different condition
    y_transfer: float array of size T_new where T_new is the number of new trials
        Design Matrix labels seperate condition to test model on (for example, train on familiar, but test on deviants)
        Default = None. Only need if testing on a different condition
        
    Return
    ------
    acc_held: Held out test accuracy (%)
    acc_transfer: Accuracy on transfer set (if applicable) (%)
    ypred: Predicated labels (list of size T_test)
    """
    
    clf = clone(lda) # Clone lda
    clf.fit(x_train,y_train) # Fit the lda
    ypred = clf.predict(x_test) # Predictions on test set
    acc_held = accuracy_score(y_test, ypred) * 100 # ACC of held out (test) set
    acc_transfer = (
        accuracy_score(y_transfer, clf.predict(x_transfer))*100 # Acc of alternative data set
        if x_transfer is not None else None
        )
    
    return acc_held, acc_transfer, ypred

def run_shuffle(lda, x_train, x_test, y_train, y_test):
    """
    Trains and tests LDA on shuffled ytrain data (CONTROL CASE)
    
    Parameters
    -----------
    lda: function
        lda model (build from build_lda)
    x_train: float array of size [T_trainxN] where T_train is the number of training trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a training trial associate with each neuron)
    x_test: float array of size [T_testxN] where T_test is the number of test trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a test trial associate with each neuron)
    y_train: float array of size T_train where T_train is the number of training trials
        Design Matrix labels (one value per row, used as the targets)
    y_test: float array of size T_test where T_test is the number of test trials
        Design Matrix labels (one value per row, used as the targets)

    Return
    ------
    shuffle accuracy score (%)
    
    """    
    
    clf = clone(lda) # Clone lda
    clf.fit(x_train,np.random.permutation(y_train)) # Train on shuffled ytrain
    return accuracy_score(y_test,clf.predict(x_test))*100 # Return acc (%)

def split_burst_tonic(data1, data2, labels, **kwargs):
    """
    Splits the burst and tonics into training and test sets under the same events for both tonic and burst.
    data1 and data2 are interchangeable. Just note that if data1 = Burst/Tonic, then x_train1 is burst/tonic, 
    
    
    Parameters
    ----------
    data1 : float array of size [TxN] where T is the number of trials and N is the number of neurons,
        Design matrix for either tonic or busrt (for example, each col would be the rate of a given neuron, 
                      and each row is a training trial associate with each neuron)
    data2 : float array of size [TxN] where T_train is the number of training trials and N is the number of neurons,
        Design matrix for the other data type (for example, each col would be the rate of a given neuron, 
                      and each row is a training trial associate with each neuron)
    labels : float array of size T where T is the number of trials
        Design Matrix labels (one value per row, used as the targets)
    **kwargs : functional arguments
        Additional arguments for train_test_split if needed

    Returns
    -------
    x_train1 : float array of size [T_trainxN] where T_train is the number of training trials and N is the number of neurons,
        Training set for data1
    x_test1 : float array of size [T_testxN] where T_test is the number of training trials and N is the number of neurons,
        Test set for data 1
    x_train2 : float array of size [T_trainxN] where T_train is the number of training trials and N is the number of neurons,
        Training set for data2
    x_test2 : float array of size [T_testxN] where T_test is the number of training trials and N is the number of neurons,
        Training set for data2
    y_train : float array of size T_train where T_train is the number of training trials
        Training Labels (same for both data1 and data2)
    y_test : float array of size T_test where T_test is the number of test trials
        Test Labels (same for both data1 and data2)
    """
    
    indices = np.arange(len(labels)) # Label indecies
    x_train1, x_test1, y_train, y_test, tr_idx, tst_idx = train_test_split(data1,labels,indices,**kwargs)
    return (x_train1, x_test1, data2[tr_idx], data2[tst_idx], y_train, y_test)

def run_roc(x_train, x_test, y_train, y_test, 
            x_transfer=None, y_transfer=None, 
            C=0.01, n_classes=8):
    """
    Fits a One vs Rest SVC and computes averaged ROC AUC.
    Trains on x_train, tests on x_test.
    Will also test on x_transfer if provided (transfer generalization).
    Chance is 0.5 for each binary OvR comparison.
    
    Parameters
    ----------
    x_train: float array of size [T_trainxN] where T_train is the number of training trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a training trial associate with each neuron)
    x_test: float array of size [T_testxN] where T_test is the number of test trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a test trial associate with each neuron)
    y_train: float array of size T_train where T_train is the number of training trials
        Design Matrix labels (one value per row, used as the targets)
    y_test: float array of size T_test where T_test is the number of test trials
        Design Matrix labels (one value per row, used as the targets)
    x_transfer: float array of size [T_newxN] where T_new is the number of new trials and N is the number of neurons,
        Design matrix for a seperate condition to test model on (for example, train on familiar, but test on deviants)
        Default = None. Only need if testing on a different condition
    y_transfer: float array of size T_new where T_new is the number of new trials
        Design Matrix labels seperate condition to test model on (for example, train on familiar, but test on deviants)
        Default = None. Only need if testing on a different condition
    C: float scalar
        LDA linear svm control paramters (defualt = 0.01)
    n_classes: int scalar
        Number of classes or unique y_train labels (defualt = 8)
        
    Return
    ------
    roc_held: float scalar,
        OvR ROC return value of held out test set
    roc_transfer: float scalar,
        OvR ROC return value of transfer set (if generalizing to another set)
    """
    
    # Build One vs Rest classifier
    clf = OneVsRestClassifier(
        make_pipeline(
            StandardScaler(),SVC(kernel='linear', probability=False, C=C, random_state=6895))
        )
    
    # Binarize labels for multiclass ROC
    classes = np.arange(n_classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Fit lda
    clf.fit(x_train, y_train)
    
    # Get decision scores for test set
    y_scores = clf.decision_function(x_test)
    roc_held = roc_auc_score(y_test_bin, y_scores, 
                              average='macro', 
                              multi_class='ovr')
    
    # Transfer set if provided
    roc_transfer = None
    if x_transfer is not None:
        y_transfer_bin = label_binarize(y_transfer, classes=classes)
        y_scores_transfer = clf.decision_function(x_transfer)
        roc_transfer = roc_auc_score(y_transfer_bin, y_scores_transfer,
                                      average='macro',
                                      multi_class='ovr')
    
    return roc_held, roc_transfer


def run_roc_shuffle(lda,x_train, x_test, y_train, y_test, n_classes=8):
    """
    Permutation control for ROC AUC - shuffles y_train before fitting
    
    Parameters
    ----------
    x_train: float array of size [T_trainxN] where T_train is the number of training trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a training trial associate with each neuron)
    x_test: float array of size [T_testxN] where T_test is the number of test trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a test trial associate with each neuron)
    y_train: float array of size T_train where T_train is the number of training trials
        Design Matrix labels (one value per row, used as the targets)
    y_test: float array of size T_test where T_test is the number of test trials
        Design Matrix labels (one value per row, used as the targets)
    n_classes: int scalar
        Number of classes or unique y_train labels (defualt = 8)
        
    Return
    ------
    roc_auc_score: float scalar,
        OvR ROC return value of shuffled set
    """
    

    classes = np.arange(n_classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    lda = clone(lda)
    lda.fit(x_train, np.random.permutation(y_train))
    y_scores = lda.decision_function(x_test)
    
    return roc_auc_score(y_test_bin, y_scores, average='macro', multi_class='ovr')

def run_roc_per_image(lda,x_train, x_test, y_train, y_test,
                      x_transfer=None, y_transfer=None,
                      C=0.01, n_classes=8):
    """
    Same as run_roc but also returns per-image ROC curves
    for both test and transfer sets
    
    Parameters
    ----------
    lda: function
        lda model (build from build_lda)
    x_train: float array of size [T_trainxN] where T_train is the number of training trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a training trial associate with each neuron)
    x_test: float array of size [T_testxN] where T_test is the number of test trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a test trial associate with each neuron)
    y_train: float array of size T_train where T_train is the number of training trials
        Design Matrix labels (one value per row, used as the targets)
    y_test: float array of size T_test where T_test is the number of test trials
        Design Matrix labels (one value per row, used as the targets)
    x_transfer: float array of size [T_newxN] where T_new is the number of new trials and N is the number of neurons,
        Design matrix for a seperate condition to test model on (for example, train on familiar, but test on deviants)
        Default = None. Only need if testing on a different condition
    y_transfer: float array of size T_new where T_new is the number of new trials
        Design Matrix labels seperate condition to test model on (for example, train on familiar, but test on deviants)
        Default = None. Only need if testing on a different condition
    C: float scalar
        LDA linear svm control paramters (defualt = 0.01)
    n_classes: int scalar
        Number of classes or unique y_train labels (defualt = 8)
        
    Return
    ------
    roc_held: float scalar,
        OvR ROC return value of held out test set
    roc_transfer: float scalar,
        OvR ROC return value of transfer set (if generalizing to another set)
    curves_held: float array,
        OvR ROC for each image in the held out test set
    curves_transfer: float array,
        OvR ROC for each image in the held out transfer set
        
    """

    # Binarize labels for multiclass ROC
    classes = np.arange(n_classes) # vector of classes
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Copy, fit, and get results from LDA
    lda = clone(lda)
    lda.fit(x_train, y_train)
    y_scores = lda.decision_function(x_test)
    
    # Overall macro ROC AUC
    roc_held = roc_auc_score(y_test_bin, y_scores,
                              average='macro',
                              multi_class='ovr')
    
    # Per image ROC curves and AUC
    fpr_per_img = {}
    tpr_per_img = {}
    auc_per_img = np.zeros(n_classes)
    for i in range(n_classes):
        fpr_per_img[i], tpr_per_img[i], _ = roc_curve(y_test_bin[:, i],y_scores[:, i]) # ROC Curve
        auc_per_img[i] = roc_auc_score(y_test_bin[:, i], y_scores[:, i]) # ROC SCore
    
    # Transfer set
    roc_transfer = None
    fpr_transfer = {}
    tpr_transfer = {}
    auc_transfer = np.zeros(n_classes)
    if x_transfer is not None:
        y_transfer_bin = label_binarize(y_transfer, classes=classes)
        y_scores_transfer = lda.decision_function(x_transfer)
        roc_transfer = roc_auc_score(y_transfer_bin, y_scores_transfer,
                                      average='macro',
                                      multi_class='ovr')
        for i in range(n_classes):
            fpr_transfer[i], tpr_transfer[i], _ = roc_curve(
                y_transfer_bin[:, i], y_scores_transfer[:, i])
            auc_transfer[i] = roc_auc_score(y_transfer_bin[:, i], 
                                             y_scores_transfer[:, i])
    
    curves_held = {'fpr': fpr_per_img, 'tpr': tpr_per_img, 'auc': auc_per_img}
    curves_transfer = {'fpr': fpr_transfer, 'tpr': tpr_transfer, 'auc': auc_transfer}
    
    return roc_held, roc_transfer, curves_held, curves_transfer

def run_roc_per_behavior(lda,x_train, x_test, y_train, y_test,
                              x_transfer=None, y_transfer=None):
    """
    Same as run_roc but also returns per-image ROC curves
    for both test and transfer sets
    
    Parameters
    ----------
    lda: function
        lda model (build from build_lda)
    x_train: float array of size [T_trainxN] where T_train is the number of training trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a training trial associate with each neuron)
    x_test: float array of size [T_testxN] where T_test is the number of test trials and N is the number of neurons,
        Design matrix (for example, each col would be the rate of a given neuron, 
                      and each row is a test trial associate with each neuron)
    y_train: float array of size T_train where T_train is the number of training trials
        Design Matrix labels (one value per row, used as the targets)
    y_test: float array of size T_test where T_test is the number of test trials
        Design Matrix labels (one value per row, used as the targets)
    x_transfer: float array of size [T_newxN] where T_new is the number of new trials and N is the number of neurons,
        Design matrix for a seperate condition to test model on (for example, train on familiar, but test on deviants)
        Default = None. Only need if testing on a different condition
    y_transfer: float array of size T_new where T_new is the number of new trials
        Design Matrix labels seperate condition to test model on (for example, train on familiar, but test on deviants)
        Default = None. Only need if testing on a different condition
    C: float scalar
        LDA linear svm control paramters (defualt = 0.01)
    n_classes: int scalar
        Number of classes or unique y_train labels (defualt = 8)
        
    Return
    ------
    roc_held: float scalar,
        OvR ROC return value of held out test set
    roc_transfer: float scalar,
        OvR ROC return value of transfer set (if generalizing to another set)
    curves_held: float array,
        OvR ROC for each image in the held out test set
    curves_transfer: float array,
        OvR ROC for each image in the held out transfer set
        
    """
    
    # Copy, fit, and get results from LDA
    lda = clone(lda)
    lda.fit(x_train, y_train)
    y_scores = lda.decision_function(x_test)

    roc_held = roc_auc_score(y_test, y_scores)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    curves_held = {'fpr': {0: fpr}, 'tpr': {0: tpr}, 'auc': np.array([roc_held])}

    roc_transfer = None
    curves_transfer = {'fpr': {}, 'tpr': {}, 'auc': np.array([])}

    if x_transfer is not None:
        y_scores_transfer = lda.decision_function(x_transfer)
        roc_transfer = roc_auc_score(y_transfer, y_scores_transfer)
        fpr_t, tpr_t, _ = roc_curve(y_transfer, y_scores_transfer)
        curves_transfer = {'fpr': {0: fpr_t}, 'tpr': {0: tpr_t}, 'auc': np.array([roc_transfer])}

    return roc_held, roc_transfer, curves_held, curves_transfer


def pval_to_star(p):
    """
    Convert a p-value to a star symbol when plotting a significance. alpha = 0.05 for min criteria of sig.
    p < 0.001 = ***
    p < 0.01 = **
    p < 0.05 = *
    p > 0.05 = n.s
    
    Parameters
    ----------
    p : float or int,
        p-value to convert to strng symbol

    Returns
    -------
    string

    """
    
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'
    
def boxsig(p,pos1,pos2,ymax,yshift, ax):
    """
    Custom function to easily plot a box plot with significance lines and appropriate star placements

    Parameters
    ----------
    p : float or int,
        p-value.
    pos1 : float or int,
          x_1 aligment for start of line connecting 2 box plots
    pos2 : float or int,
        x_2 aligment for end of line connecting 2 box plots
    ymax : float or int,
        Max y-location for line
    yshift : float or int,
        y-shift to put the text above the ymax line
    ax : axis handle

    Returns
    -------
    None.

    """
    
    ax.plot([pos1, pos1,pos2, pos2], [ymax, ymax+yshift, ymax+yshift, ymax], lw=1.2, c="black")
    ax.text(np.mean([pos1,pos2]), ymax+yshift, pval_to_star(p),
             ha="center", va="bottom", fontsize=10)

# ===========================================
# 1. Configuration & Seed Setup
# ===========================================
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import random
import pickle
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Dict, Any


import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve,
    classification_report, roc_curve
)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

import torch
import torch.nn as nn
from torch import optim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../libs/'))
import def_function as func

from sklearn.linear_model import LogisticRegression


# Reproducibility
SEED = 47
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ===========================================
# 2. Utility Functions
# ===========================================
def get_common_path(relative_path: str) -> str:
    """Get absolute path from relative path."""
    cur_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cur_path, relative_path)

def time_consumption_since(start: float) -> str:
    """Calculate time elapsed since start time."""
    elapsed = time.time() - start
    m = int(elapsed // 60)
    s = int(elapsed % 60)
    return f"{m}m {s}s"

def load_input_pkl(common_path: str):
    """Load train, test, validation and type mapping data from pickle files."""
    names = ['.combined.train', '.combined.test', '.combined.valid', '.types']
    return [
        pickle.load(open(common_path + "lt" + nm, 'rb'), encoding='bytes')
        for nm in names
    ]

def prepare_train_and_test_features(train_sl: List, valid_sl: List, test_sl: List, input_size_1: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare features and labels for model training and testing."""
    # Process training and validation data
    pts_tr, labels_tr, features_tr = [], [], []
    for pt in train_sl + valid_sl:
        pts_tr.append(pt[0])
        labels_tr.append(pt[1])
        features_tr.append([code for v in pt[-1] for code in v[-1]])
    
    # Process test data
    pts_t, labels_t, features_t = [], [], []
    for pt in test_sl:                  
        pts_t.append(pt[0])
        labels_t.append(pt[1])
        features_t.append([code for v in pt[-1] for code in v[-1]])

    mlb = MultiLabelBinarizer(classes=range(input_size_1[0])[1:])
    return mlb.fit_transform(features_tr), np.array(labels_tr), mlb.transform(features_t), np.array(labels_t)


# ===========================================
# 3. Evaluation Functions
# ===========================================
def calculate_metrics(y_real, y_pred_proba, y_pred_class):
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        'auc': roc_auc_score(y_real, y_pred_proba),
        'auprc': average_precision_score(y_real, y_pred_proba)
    }
    
    # Precision, recall, f1-score
    precision, recall, fscore, _ = precision_recall_fscore_support(y_real, y_pred_class, average='binary')
    precision_w, recall_w, fscore_w, _ = precision_recall_fscore_support(y_real, y_pred_class, average='weighted')
    
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1': fscore,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': fscore_w,
        'accuracy': accuracy_score(y_real, y_pred_class),
        'balanced_accuracy': balanced_accuracy_score(y_real, y_pred_class)
    })
    
    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred_class).ravel()
    metrics['specificity'] = tn / (tn + fp)
    
    return metrics

def save_results_table(y_real, y_pred_proba, results_path, window_days):
    """Save ROC curve data and metrics to CSV."""
    fpr, tpr, thresholds = roc_curve(y_real, y_pred_proba)
    
    # Find optimal threshold
    th_optimal = thresholds[np.argmax(tpr - fpr)]
    gmeans = np.sqrt(tpr * (1 - fpr))
    print(f'Best Threshold={th_optimal:.4f}, G-Mean={gmeans.max():.3f}')
    
    # Create results table
    result_table = pd.DataFrame({
        'classifiers': ['LR'],
        'fpr': [fpr.tolist()],
        'tpr': [tpr.tolist()],
        'auc': [roc_auc_score(y_real, y_pred_proba)],
        'auprc': [average_precision_score(y_real, y_pred_proba)]
    }).set_index('classifiers')
    
    result_table.to_csv(os.path.join(results_path, f"result_table_LR_{window_days}d.csv"))
    return result_table

def perform_bootstrap_analysis(nfeatures_t, labels_t, results_path, window_days, n_trials=500, alpha=0.05):
    """Perform bootstrap confidence interval analysis for LR model."""
    # Load saved LR model
    model_path = os.path.join(
        results_path, 
        'optuna_tuning', 
        'objective_LR_LT', 
        f'0d_{window_days}d', 
        'test_model.p'
    )
    clf = pickle.load(open(model_path, 'rb'))
    
    # Initialize bootstrap results
    all_CI = defaultdict(list)
    
    print("Starting bootstrap trials...")
    for trial_num in range(n_trials):
        # Resample test data
        X_b, y_b = resample(nfeatures_t, labels_t)
        
        # Generate predictions
        y_score = clf.predict_proba(X_b)[:,1]
        y_pred = clf.predict(X_b)
        
        # Calculate metrics for this trial
        metrics = calculate_metrics(y_b, y_score, y_pred)
        
        # Store metrics
        for key, value in metrics.items():
            all_CI[key].append(value)
        
        print(f"Completed bootstrap trial {trial_num + 1}/{n_trials}")
    
    # Calculate confidence intervals
    df_ci = pd.DataFrame(columns=["lower", "upper", "95%_CI", "stat", "model"])
    
    for key in all_CI.keys():
        p_lower = (alpha/2) * 100
        p_upper = (1-alpha/2) * 100
        lower = round(max(0, np.percentile(all_CI[key], p_lower)), 4)
        upper = round(min(1, np.percentile(all_CI[key], p_upper)), 4)
        
        df_ci.loc[len(df_ci)] = [
            lower, upper, f"({lower},{upper})", key, f"LR_0d_{window_days}d"
        ]
    
    # Save confidence intervals
    ci_filename = f"confidence_interval_LR_0d_{window_days}d.csv"
    df_ci.to_csv(os.path.join(results_path, ci_filename), index=False)
    
    # Save PR curve data (using full test set, not bootstrap sample)
    y_score = clf.predict_proba(nfeatures_t)[:,1]
    precision, recall, _ = precision_recall_curve(labels_t, y_score)
    
    # Save precision and recall curves
    for data, name in [(precision, 'precision'), (recall, 'recall')]:
        filename = f'{name}_LR_0d_{window_days}d.pkl'
        with open(os.path.join(results_path, filename), 'wb') as f:
            pickle.dump(data.tolist(), f)
    
    print(f"Bootstrap analysis completed. Results saved to {ci_filename}")
    return df_ci


# ===========================================
# 4. Main Execution
# ===========================================
if __name__ == "__main__":
    start_tm = datetime.now()
    
    # Configuration
    output_folder_name = 'Results_15_final_new'
    days_before_index_date = 1095
    part_common_path = get_common_path('../../../Results/') + output_folder_name + '/'
    results_output_path = func.get_common_path('../../Test_results') + '/' + str(days_before_index_date) + '/'
    
    # Selected time windows
    selected_list = ['Results_0d_30d_window']
    #selected_list = ['Results_0d_365d_window']
    #selected_list = ['Results_0d_1095d_window']
    #selected_list = ['Results_0d_1825d_window']
    
    # Process each time window
    for idx_folder, specific_folder_name in enumerate(selected_list):
        common_path = part_common_path + str(days_before_index_date) + "/" + specific_folder_name + '/'
        print(f'{idx_folder+1}. Running ML for file from {specific_folder_name}:\n')
        
        # Extract window days from folder name (e.g., 'Results_0d_365d_window' -> 365)
        window_days = int(specific_folder_name.split('_')[2].replace('d', ''))
        
        # ----------------------------
        # Data Loading & Preparation
        # ----------------------------
        train_sl, test_sl, valid_sl, types_d = load_input_pkl(common_path)
        types_d_rev = dict(zip(types_d.values(), types_d.keys()))
        input_size = [len(types_d_rev) + 1]
        
        print(f'len(train_sl) is {len(train_sl)}, len(valid_sl) is {len(valid_sl)}, len(test_sl) is {len(test_sl)}, input_size is {input_size}', '\n')

        nfeatures_tr, labels_tr, nfeatures_t, labels_t = prepare_train_and_test_features(train_sl, valid_sl, test_sl, input_size)
        
        # ----------------------------                                                                    
        # Model Training
        # ----------------------------
        to_train_model = True
        if to_train_model:
            #Fit the best model on the test data and save the model in the test_results folder
            #create the .csv file to generate ['classifiers', 'fpr','tpr','auc', 'auprc']

            # Define optimal parameters
            logreg_c = 0.006684444
            
            # Initialize and train model
            clf = LogisticRegression(C=logreg_c, random_state = SEED, class_weight='balanced')
            clf.fit(nfeatures_tr, labels_tr)

            
            # Save model
            ckpt_dir = os.path.join(results_output_path, 'optuna_tuning', 'objective_LR_LT', f'0d_{window_days}d')
            if not os.path.exists(ckpt_dir):
                print("Creating checkpoint directory")
                os.makedirs(ckpt_dir)
            
            model_path = os.path.join(ckpt_dir, 'test_model.p')
            func.save_pkl(clf, model_path)
            
            # ----------------------------
            # Metrics Calculation & Saving
            # ----------------------------
            # Generate predictions
            y_score = clf.predict_proba(nfeatures_t)[:,1]
            y_pred = clf.predict(nfeatures_t)
            
            # Calculate comprehensive metrics
            metrics = calculate_metrics(labels_t, y_score, y_pred)
            
            # Print results
            print("\nFinal Metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            
            print('\nClassification Report:')
            print(classification_report(labels_t, y_pred))
            
            # Save results table
            save_results_table(labels_t, y_score, results_output_path, window_days)
        

        to_calculate_bootstrapped_performance = True
        if to_calculate_bootstrapped_performance:
            print("\nStarting bootstrap confidence interval analysis...")
            df_ci = perform_bootstrap_analysis(nfeatures_t,labels_t, results_output_path, window_days)
            print("Bootstrap analysis completed.")
    
    print(f'\nTotal elapsed time: {datetime.now() - start_tm}')
    print('Training complete!')

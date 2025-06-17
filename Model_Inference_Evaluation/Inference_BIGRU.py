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

import torch
import torch.nn as nn
from torch import optim

# Add custom libs to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../libs/'))

# Reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# GPU Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# ===========================================
# 2. Custom Imports
# ===========================================
import models as model
from EHRDataloader import EHRdataloader, EHRdataFromLoadedPickles as EHRDataset
import utils as ut
import def_function as func

# ===========================================
# 3. Utility Functions
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

# ===========================================
# 4. Training Functions
# ===========================================
def train_batches(mbs_list, model, optimizer, shuffle=True, loss_fn = nn.BCELoss()):
    """Train model on batches with weighted loss."""
    current_loss = 0.0
    all_losses = []
    plot_every = 5
    
    if shuffle:
        random.shuffle(mbs_list)
    
    for i, batch in enumerate(mbs_list, 1):
        sample, label_tensor, seq_l, mtd = batch
        
        # Set class weights
        weight = torch.zeros_like(label_tensor)
        weight[label_tensor == 0] = 0.5378951502061653
        weight[label_tensor == 1] = 7.097150259067358
        
        loss_fn = nn.BCELoss(weight.squeeze())
        
        output, loss = ut.trainsample(sample, label_tensor, seq_l, mtd, model, optimizer, criterion=loss_fn)
        current_loss += loss
        
        if i % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0.0
    
    return current_loss, all_losses

def run_dl_model(ehr_model, train_mbs, valid_mbs, model_name, epochs=100,
                 lr=5e-4, l2=1e-4, eps=1e-4, opt_name='Adagrad', patience=5, window_days = 30,
                 results_output_path=''):
    """Train deep learning model with early stopping."""
    
    # Optimizer selection
    optimizers = {
        'Adadelta': lambda: optim.Adadelta(ehr_model.parameters(), lr=lr, weight_decay=l2, eps=eps),
        'Adagrad':  lambda: optim.Adagrad(ehr_model.parameters(), lr=lr, weight_decay=l2),
        'Adam':     lambda: optim.Adam(ehr_model.parameters(), lr=lr, weight_decay=l2, eps=eps),
        'Adamax':   lambda: optim.Adamax(ehr_model.parameters(), lr=lr, weight_decay=l2, eps=eps),
        'RMSprop':  lambda: optim.RMSprop(ehr_model.parameters(), lr=lr, weight_decay=l2, eps=eps),
        'ASGD':     lambda: optim.ASGD(ehr_model.parameters(), lr=lr, weight_decay=l2),
        'SGD':      lambda: optim.SGD(ehr_model.parameters(), lr=lr, weight_decay=l2, momentum=0.9),
    }
    
    optimizer = optimizers[opt_name]()
    
    # Training loop with early stopping
    best_valid_auc = 0.0
    best_valid_epoch = 0
    
    for epoch in range(1, epochs + 1):
        # Training phase
        start = time.time()
        current_loss, train_loss = train_batches(train_mbs, model=ehr_model, optimizer=optimizer, loss_fn = nn.BCELoss())
        avg_loss = np.mean(train_loss)
        train_time = time_consumption_since(start)
        
        # Evaluation phase
        train_auc, y_t_real, y_t_hat = ut.calculate_auc(ehr_model, train_mbs, which_model=model_name)
        valid_auc, y_v_real, y_v_hat = ut.calculate_auc(ehr_model, valid_mbs, which_model=model_name)
        
        # Log progress
        print(f"Epoch {epoch:3d} | Train AUC: {train_auc:.4f} | Valid AUC: {valid_auc:.4f} | "
              f"Avg Loss: {avg_loss:.4f} | Time: {train_time}")
        
        # Check for improvement
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_valid_epoch = epoch
            y_real = y_v_real
            y_hat = y_v_hat
            
            # Display detailed metrics for best model
            y_pred = (np.array(y_hat) > 0.5)
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_real, y_pred))
            print('\nClassification Report:')
            print(classification_report(y_real, y_pred))
            
            # Save best model
            ckpt_dir = os.path.join(results_output_path, 'optuna_tuning', 'objective_BiGRU_LT', f'0d_{window_days}d')
            os.makedirs(ckpt_dir, exist_ok=True)
            
            model_path = os.path.join(ckpt_dir, 'test_model.p')
            param_path = os.path.join(ckpt_dir, 'test_parameters.p')
            torch.save(ehr_model, model_path)
            torch.save(ehr_model.state_dict(), param_path)
            print(f"→ Saved best model at epoch {epoch} (AUC={best_valid_auc:.4f})")
        
        # Early stopping check
        if epoch - best_valid_epoch > patience:
            break
    
    # Final results
    print(f'Best validation AUC {best_valid_auc:.4f} at epoch {best_valid_epoch}')
    return y_real, y_hat

# ===========================================
# 5. Evaluation Functions
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
        'classifiers': ['BiGRU'],
        'fpr': [fpr.tolist()],
        'tpr': [tpr.tolist()],
        'auc': [roc_auc_score(y_real, y_pred_proba)],
        'auprc': [average_precision_score(y_real, y_pred_proba)]
    }).set_index('classifiers')
    
    result_table.to_csv(os.path.join(results_path, f"result_table_BiGRU_{window_days}d.csv"))
    return result_table

def perform_bootstrap_analysis(test_sl, results_path, window_days, n_trials=500, alpha=0.05):
    """Perform bootstrap confidence interval analysis."""
    # Load best model
    ckpt_dir = os.path.join(
        results_path,
        'optuna_tuning',
        'objective_BiGRU_LT',  # Keep original naming
        f'0d_{window_days}d'
    )
    best_model = torch.load(os.path.join(ckpt_dir, "test_model.p"))
    best_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "test_parameters.p")))
    
    if USE_CUDA:
        best_model.cuda()
    best_model.eval()
    
    # Initialize bootstrap results
    all_CI = defaultdict(list)
    
    print("Starting bootstrap trials...")
    for trial_num in range(n_trials):
        # Resample test data
        test_resample = resample(test_sl)
        test_dataset = EHRDataset(test_resample, sort=True, model='RNN')
        test_mbs = list(EHRdataloader(test_dataset, batch_size=128, packPadMode=True))
        
        # Calculate predictions
        _, y_real, y_hat = ut.calculate_auc(best_model, test_mbs, which_model='RNN')
        y_pred = np.where(np.array(y_hat) > 0.5, 1, 0).tolist()
        
        # Calculate metrics for this trial
        metrics = calculate_metrics(y_real, y_hat, y_pred)
        
        # Store metrics
        for key, value in metrics.items():
            all_CI[key].append(value)
        
        
        print(f"Bootstrap trial {trial_num} completed")
    
    # Calculate confidence intervals
    df_ci = pd.DataFrame(columns=["lower", "upper", "95%_CI", "stat", "model"])
    
    for key in all_CI.keys():
        p_lower = (alpha/2) * 100
        p_upper = (1-alpha/2) * 100
        lower = round(max(0, np.percentile(all_CI[key], p_lower)), 4)
        upper = round(min(1, np.percentile(all_CI[key], p_upper)), 4)
        
        df_ci.loc[len(df_ci)] = [
            lower, upper, f"({lower},{upper})", key, f"BiGRU{window_days}d"
        ]
    
    # Save confidence intervals
    df_ci.to_csv(os.path.join(results_path, f"CI_BiGRU_{window_days}d.csv"), index=False)   
    
    # Save PR curve data
    test_dataset = EHRDataset(test_sl, sort=True, model='RNN')
    test_mbs = list(EHRdataloader(test_dataset, batch_size=128, packPadMode=True))
    _, labels_t, y_score = ut.calculate_auc(best_model, test_mbs, which_model='RNN')
    
    precision, recall, _ = precision_recall_curve(labels_t, y_score)
    with open(os.path.join(results_path, f'precision_BiGRU_0d_{window_days}d.pkl'), 'wb') as f:
        pickle.dump(precision.tolist(), f)
    with open(os.path.join(results_path, f'recall_BiGRU_0d_{window_days}d.pkl'), 'wb') as f:
        pickle.dump(recall.tolist(), f)
    
    return df_ci


# ===========================================
# 6. Main Execution
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
        print(f'{idx_folder+1}. Running DL for file from {specific_folder_name}:\n')
        
        # Extract window days from folder name (e.g., 'Results_0d_365d_window' -> 365)
        window_days = int(specific_folder_name.split('_')[2].replace('d', ''))
        
        # ----------------------------
        # Data Loading & Preparation
        # ----------------------------
        train_sl, test_sl, valid_sl, types_d = load_input_pkl(common_path)
        types_d_rev = dict(zip(types_d.values(), types_d.keys()))
        input_size = [len(types_d_rev) + 1]
        
        # Combine train and validation sets
        train_list = train_sl + valid_sl
        
        # ----------------------------                                                                    
        # Model Training
        # ----------------------------
        to_train_model = True
        if to_train_model:
            #Fit the best model on the test data and save the model in the test_results folder
            #create the .csv file to generate ['classifiers', 'fpr','tpr','auc', 'auprc']

            # Define optimal parameters
            params = {
                'lr_expo': -1,
                'l2_expo': -4,
                'eps_expo': -4,
                'embed_dim_expo': 7,
                'hidden_size_expo': 6,
                'optimizer_name': 'Adagrad'
            }
            
            # Initialize model
            ehr_model = model.EHR_RNN(
                input_size,
                embed_dim=2**params['embed_dim_expo'],
                hidden_size=2**params['hidden_size_expo'],
                n_layers=2,
                dropout_r=0.1,
                cell_type='GRU',
                bii=True,
                time=True
            )
            
            if USE_CUDA:
                ehr_model = ehr_model.cuda()
            
            # Prepare data loaders
            train_dataset = EHRDataset(train_list, sort=True, model='RNN')
            train_mbs = list(EHRdataloader(train_dataset, batch_size=128, packPadMode=True))
            
            test_dataset = EHRDataset(test_sl, sort=True, model='RNN')
            test_mbs = list(EHRdataloader(test_dataset, batch_size=128, packPadMode=True))
            
            # Train model
            y_real, y_pred_proba = run_dl_model(
                ehr_model, train_mbs, test_mbs, 'RNN',
                epochs=100,
                lr=10**params['lr_expo'],
                l2=10**params['l2_expo'],
                eps=10**params['eps_expo'],
                opt_name=params['optimizer_name'],
                patience=5, 
                window_days = window_days,
                results_output_path=results_output_path
            )
            
            # ----------------------------
            # Metrics Calculation & Saving
            # ----------------------------
            y_pred_class = np.where(np.array(y_pred_proba) > 0.5, 1, 0).tolist()
            
            # Calculate comprehensive metrics
            metrics = calculate_metrics(y_real, y_pred_proba, y_pred_class)
            
            # Print results
            print("\nFinal Metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            
            print('\nClassification Report:')
            print(classification_report(y_real, y_pred_class))
            
            # Save results table
            save_results_table(y_real, y_pred_proba, results_output_path, window_days)
        

        to_calculate_bootstrapped_performance = True
        if to_calculate_bootstrapped_performance:
            print("\nStarting bootstrap confidence interval analysis...")
            df_ci = perform_bootstrap_analysis(test_sl, results_output_path, window_days)
            print("Bootstrap analysis completed.")
    
    print(f'\nTotal elapsed time: {datetime.now() - start_tm}')
    print('Training complete!')

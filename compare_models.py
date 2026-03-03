#!/usr/bin/env python3
"""
EEG Attention Classifier - Model Comparison & Training
=======================================================
Compares RandomForest, LDA, SVM, and GradBoost classifiers for EEG-based
attention detection. Trains the best model and reports feature importance.

Usage:
    python compare_models.py --data_root data/rsvp
    python compare_models.py --subject_dir data/rsvp/sub-01
"""

import argparse
import glob
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as spsig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SAMPLING_RATE = 250  # Hz
EPOCH_PRE_S = 0.200
EPOCH_PRE_SAMP = round(EPOCH_PRE_S * SAMPLING_RATE)
TARGET_LETTER = 'X'

# EEG channel names (8-channel Muse headset)
CHANNEL_NAMES = ["T5", "P3", "Pz", "P4", "T6", "O1", "O2", "REF"]


# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
def bandpower(signal_1ch, sfreq, band):
    """Compute band power using Welch's method."""
    f, psd = spsig.welch(signal_1ch, fs=sfreq, nperseg=min(256, len(signal_1ch)))
    idx = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx):
        return 0.0
    return np.trapz(psd[idx], f[idx])


def extract_features(epochs, sfreq=SAMPLING_RATE, pre_samp=EPOCH_PRE_SAMP):
    """
    Extract EEG features from epochs for classification.
    
    Features include:
    - Band powers (theta, alpha, beta, gamma) per channel
    - ERP components (N200, P300, LPP amplitudes)
    - Statistical features (variance, skewness, kurtosis)
    - Pre-stimulus alpha power
    """
    n_trials, n_ch, _ = epochs.shape
    features = []
    
    for ep in epochs:
        if np.any(np.isnan(ep)):
            features.append([np.nan] * (13 + 4 * n_ch))
            continue
        
        # Post-stimulus epoch (after stimulus onset)
        post_epoch = ep[:, pre_samp:]
        n_post = post_epoch.shape[1]
        
        # Band powers per channel
        theta_pows = [bandpower(post_epoch[ch], sfreq, (4, 8)) for ch in range(n_ch)]
        alpha_pows = [bandpower(post_epoch[ch], sfreq, (8, 13)) for ch in range(n_ch)]
        beta_pows = [bandpower(post_epoch[ch], sfreq, (13, 30)) for ch in range(n_ch)]
        gamma_pows = [bandpower(post_epoch[ch], sfreq, (30, 45)) for ch in range(n_ch)]
        
        # Mean band powers across channels
        mean_theta = float(np.mean(theta_pows))
        mean_alpha = float(np.mean(alpha_pows))
        mean_beta = float(np.mean(beta_pows))
        mean_gamma = float(np.mean(gamma_pows))
        
        # ERP components
        # N200: negative deflection ~150-250ms post-stimulus
        n200_start = round(0.150 * sfreq)
        n200_end = round(0.250 * sfreq)
        n200_amp = float(np.mean(post_epoch[:, n200_start:n200_end])) if n200_end <= n_post else 0.0
        
        # P300: positive deflection ~300-500ms post-stimulus
        p300_start = round(0.300 * sfreq)
        p300_end = round(0.500 * sfreq)
        p300_amp = float(np.mean(post_epoch[:, p300_start:p300_end])) if p300_end <= n_post else 0.0
        
        # LPP: Late Positive Potential ~400-800ms
        lpp_start = round(0.400 * sfreq)
        lpp_end = round(0.800 * sfreq)
        lpp_amp = float(np.mean(post_epoch[:, lpp_start:lpp_end])) if lpp_end <= n_post else 0.0
        
        # Statistical features
        epoch_var = float(np.mean(np.var(post_epoch, axis=1)))
        epoch_std = float(np.mean(np.std(post_epoch, axis=1)))
        mean_amp = float(np.mean(post_epoch))
        max_amp = float(np.max(np.abs(post_epoch)))
        
        # Pre-stimulus alpha (baseline)
        pre_epoch = ep[:, :pre_samp]
        pre_alpha = [bandpower(pre_epoch[ch], sfreq, (8, 13)) for ch in range(n_ch)]
        mean_pre_alpha = float(np.mean(pre_alpha))
        
        # Alpha change (post - pre)
        alpha_change = mean_alpha - mean_pre_alpha
        
        # Build feature vector
        feat_vec = [
            mean_theta, mean_alpha, mean_beta, mean_gamma,
            n200_amp, p300_amp, lpp_amp,
            epoch_var, epoch_std, mean_amp, max_amp,
            mean_pre_alpha, alpha_change,
        ] + alpha_pows + theta_pows + beta_pows + gamma_pows
        
        features.append(feat_vec)
    
    return np.array(features, dtype=np.float32)


def get_feature_names(n_ch=8):
    """Get human-readable feature names."""
    ch = CHANNEL_NAMES[:n_ch]
    return [
        "theta_mean", "alpha_mean", "beta_mean", "gamma_mean",
        "n200_amp", "p300_amp", "lpp_amp",
        "epoch_var", "epoch_std", "mean_amp", "max_amp",
        "pre_alpha_mean", "alpha_change",
    ] + [f"alpha_{c}" for c in ch] + \
        [f"theta_{c}" for c in ch] + \
        [f"beta_{c}" for c in ch] + \
        [f"gamma_{c}" for c in ch]


# ─────────────────────────────────────────────────────────────
#  LABELING
# ─────────────────────────────────────────────────────────────
def assign_labels(metadata_df, target_letter=TARGET_LETTER):
    """
    Assign focused (0) vs zoned-out (1) labels based on behavioral data.
    
    Only target letter trials are used:
    - Focused (0): Hit with RT below threshold (mean + 1.5*std)
    - Zoned-out (1): Miss OR hit with slow RT
    """
    target_df = metadata_df[metadata_df['letter'] == target_letter].copy()
    
    if len(target_df) == 0:
        return pd.DataFrame()
    
    hit_rts = target_df.loc[target_df['response'] == 'hit', 'rt'].dropna()
    
    if len(hit_rts) == 0:
        target_df['label'] = 1
        target_df['label_name'] = 'zoned_out'
        return target_df
    
    rt_mean = hit_rts.mean()
    rt_std = hit_rts.std() if len(hit_rts) > 1 else 0.0
    slow_threshold = rt_mean + 1.5 * rt_std
    
    def label_row(row):
        if row['response'] == 'miss':
            return 1
        elif row['response'] == 'hit':
            if pd.isna(row['rt']):
                return np.nan
            return 1 if row['rt'] > slow_threshold else 0
        return np.nan
    
    target_df['label'] = target_df.apply(label_row, axis=1)
    target_df['label_name'] = target_df['label'].map({0: 'focused', 1: 'zoned_out'})
    
    return target_df


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
def discover_sessions(subject_dir):
    """Find all session directories within a subject folder."""
    pattern = os.path.join(subject_dir, 'ses-*')
    sessions = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    return sessions


def load_session(session_dir):
    """Load EEG epochs and metadata from a single session."""
    epochs_path = os.path.join(session_dir, 'eeg_epochs.npy')
    if not os.path.exists(epochs_path):
        return None, None
    
    epochs = np.load(epochs_path, allow_pickle=True)
    
    meta_path = os.path.join(session_dir, 'metadata.csv')
    if not os.path.exists(meta_path):
        return None, None
    
    metadata = pd.read_csv(meta_path)
    labeled_df = assign_labels(metadata)
    
    if len(labeled_df) == 0:
        return None, None
    
    labeled_df = labeled_df.dropna(subset=['label'])
    
    if len(labeled_df) == 0:
        return None, None
    
    trial_indices = labeled_df['index'].astype(int).values
    labels = labeled_df['label'].astype(int).values
    
    labeled_epochs = epochs[trial_indices]
    
    valid_mask = ~np.any(np.isnan(labeled_epochs.reshape(len(labeled_epochs), -1)), axis=1)
    labeled_epochs = labeled_epochs[valid_mask]
    labels = labels[valid_mask]
    
    if len(labels) == 0:
        return None, None
    
    X = extract_features(labeled_epochs)
    
    feat_valid = ~np.any(np.isnan(X), axis=1)
    X = X[feat_valid]
    y = labels[feat_valid]
    
    return X, y


def load_data(subject_dir=None, data_root=None):
    """Load EEG data from subject folder or data root."""
    if data_root:
        # Discover all subject/session directories
        pattern = os.path.join(data_root, 'sub-*', 'ses-*')
        sessions = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
        
        if not sessions:
            raise FileNotFoundError(f"No session directories found in {data_root}")
        
        print(f"\n{'='*60}")
        print(f"LOADING DATA FROM: {data_root}")
        print(f"Found {len(sessions)} session(s)")
        print('='*60)
        
        all_X = []
        all_y = []
        
        for session in sessions:
            print(f"\n[LOADING] {session}")
            X_sess, y_sess = load_session(session)
            if X_sess is not None and len(y_sess) > 0:
                all_X.append(X_sess)
                all_y.append(y_sess)
                unique, counts = np.unique(y_sess, return_counts=True)
                class_str = ', '.join([f"{'Focused' if u == 0 else 'Zoned-out'}: {c}" for u, c in zip(unique, counts)])
                print(f"  Valid trials: {len(y_sess)} ({class_str})")
        
        if not all_X:
            raise ValueError(f"No valid data found in {data_root}")
        
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
    elif subject_dir:
        sessions = discover_sessions(subject_dir)
        
        if not sessions:
            raise FileNotFoundError(f"No session directories found in {subject_dir}")
        
        print(f"\n{'='*60}")
        print(f"LOADING SUBJECT DATA: {subject_dir}")
        print(f"Found {len(sessions)} session(s)")
        print('='*60)
        
        all_X = []
        all_y = []
        
        for session in sessions:
            print(f"\n[LOADING] {session}")
            X_sess, y_sess = load_session(session)
            if X_sess is not None and len(y_sess) > 0:
                all_X.append(X_sess)
                all_y.append(y_sess)
                unique, counts = np.unique(y_sess, return_counts=True)
                class_str = ', '.join([f"{'Focused' if u == 0 else 'Zoned-out'}: {c}" for u, c in zip(unique, counts)])
                print(f"  Valid trials: {len(y_sess)} ({class_str})")
        
        if not all_X:
            raise ValueError(f"No valid data found in {subject_dir}")
        
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
    else:
        raise ValueError("Either subject_dir or data_root must be provided")
    
    print(f"\n{'='*60}")
    print("COMBINED DATA SUMMARY")
    print('='*60)
    print(f"  Total samples: {len(y)}")
    print(f"  Feature matrix: {X.shape}")
    
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        label_name = 'Focused' if u == 0 else 'Zoned-out'
        print(f"  {label_name} (class {u}): {c} trials ({c/len(y)*100:.1f}%)")
    
    return X, y


# ─────────────────────────────────────────────────────────────
#  MODEL COMPARISON
# ─────────────────────────────────────────────────────────────
def compare_models(X, y):
    """Compare RandomForest, LDA, SVM, and GradBoost classifiers."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Check class distribution
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("\n[ERROR] Only one class present in the data!")
        return None, None, None
    
    # Define models to compare
    models = {
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', 
                                          random_state=42, n_jobs=-1))
        ]),
        'LDA': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearDiscriminantAnalysis())
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                       probability=True, random_state=42))
        ]),
        'GradBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    # Cross-validation setup
    min_class_count = min(np.bincount(y))
    n_splits = min(5, min_class_count)
    
    if n_splits < 2:
        print(f"\n[WARN] Not enough samples for CV (min class has {min_class_count} samples)")
        n_splits = 2
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\n[CV] Using {n_splits}-fold stratified cross-validation")
    print(f"[CV] Evaluating {len(models)} models...\n")
    
    # Evaluate each model
    results = {}
    best_score = -np.inf
    best_model_name = None
    best_pipeline = None
    
    for name, pipeline in models.items():
        print(f"[{name}]")
        
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_balanced = cross_val_score(pipeline, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        
        results[name] = {
            'accuracy': cv_scores,
            'balanced_accuracy': cv_balanced,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'mean_balanced_accuracy': cv_balanced.mean(),
            'std_balanced_accuracy': cv_balanced.std()
        }
        
        print(f"  Accuracy:          {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Balanced Accuracy: {cv_balanced.mean():.3f} ± {cv_balanced.std():.3f}")
        print()
        
        # Track best model based on balanced accuracy
        if cv_balanced.mean() > best_score:
            best_score = cv_balanced.mean()
            best_model_name = name
            best_pipeline = pipeline
    
    print("="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"  Balanced Accuracy: {best_score:.3f} ± {results[best_model_name]['std_balanced_accuracy']:.3f}")
    print("="*60)
    
    return best_model_name, best_pipeline, results


# ─────────────────────────────────────────────────────────────
#  TRAINING & FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
def train_best_model(X, y, model_name, pipeline, save_path=None):
    """Train the best model on full dataset and extract feature importance."""
    print("\n" + "="*60)
    print(f"TRAINING FINAL MODEL: {model_name}")
    print("="*60)
    
    # Train on full dataset
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    
    # Evaluation
    accuracy = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    
    print("\n[TRAINING SET EVALUATION]")
    print(classification_report(y, y_pred, target_names=['Focused', 'Zoned-out']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"                 Predicted")
    print(f"                 Focused  Zoned-out")
    print(f"  Actual Focused    {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"  Actual Zoned-out  {cm[1,0]:3d}       {cm[1,1]:3d}")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    feature_names = get_feature_names(n_ch=8)
    feature_importance = None
    
    clf = pipeline.named_steps['clf']
    
    if hasattr(clf, 'feature_importances_'):
        # RandomForest or GradBoost
        feature_importance = clf.feature_importances_
        
        # Get top 20 features
        indices = np.argsort(feature_importance)[::-1][:20]
        
        print("\nTop 20 Most Important Features:")
        print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12} {'Description'}")
        print("-" * 70)
        
        for rank, idx in enumerate(indices, 1):
            feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            importance = feature_importance[idx]
            
            # Add description
            if 'Pz' in feat_name:
                desc = "(Parietal midline - attention)"
            elif 'P3' in feat_name or 'P4' in feat_name:
                desc = "(Parietal - cognitive processing)"
            elif 'alpha' in feat_name:
                desc = "(8-13 Hz - relaxation/attention)"
            elif 'theta' in feat_name:
                desc = "(4-8 Hz - drowsiness/meditation)"
            elif 'beta' in feat_name:
                desc = "(13-30 Hz - active thinking)"
            elif 'p300' in feat_name:
                desc = "(P300 ERP - attention to target)"
            else:
                desc = ""
            
            print(f"{rank:<6} {feat_name:<25} {importance:<12.4f} {desc}")
    
    elif hasattr(clf, 'coef_'):
        # LDA or SVM (linear)
        coef = np.abs(clf.coef_[0])
        feature_importance = coef
        
        indices = np.argsort(coef)[::-1][:20]
        
        print("\nTop 20 Most Important Features (by coefficient magnitude):")
        print(f"{'Rank':<6} {'Feature':<25} {'|Coefficient|':<15} {'Description'}")
        print("-" * 75)
        
        for rank, idx in enumerate(indices, 1):
            feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            importance = coef[idx]
            
            if 'Pz' in feat_name:
                desc = "(Parietal midline - attention)"
            elif 'P3' in feat_name or 'P4' in feat_name:
                desc = "(Parietal - cognitive processing)"
            elif 'alpha' in feat_name:
                desc = "(8-13 Hz - relaxation/attention)"
            elif 'theta' in feat_name:
                desc = "(4-8 Hz - drowsiness/meditation)"
            elif 'beta' in feat_name:
                desc = "(13-30 Hz - active thinking)"
            elif 'p300' in feat_name:
                desc = "(P300 ERP - attention to target)"
            else:
                desc = ""
            
            print(f"{rank:<6} {feat_name:<25} {importance:<15.4f} {desc}")
    else:
        print("\n[INFO] Feature importance not available for this model type")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Model: {model_name}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(y)} ({sum(y==0)} focused, {sum(y==1)} zoned-out)")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Balanced Accuracy: {balanced_acc:.1%}")
    
    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': pipeline,
                'model_name': model_name,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'n_features': X.shape[1],
                'n_samples': len(y),
                'feature_names': feature_names,
                'feature_importance': feature_importance,
            }, f)
        print(f"\n[SAVED] Model -> {save_path}")
    
    return pipeline, feature_importance


def plot_comparison(results, best_model_name, feature_importance, feature_names, save_dir):
    """Plot model comparison and feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Model comparison
    ax = axes[0]
    model_names = list(results.keys())
    balanced_accs = [results[name]['mean_balanced_accuracy'] for name in model_names]
    balanced_stds = [results[name]['std_balanced_accuracy'] for name in model_names]
    
    colors = ['#2ecc71' if name == best_model_name else '#3498db' for name in model_names]
    bars = ax.bar(model_names, balanced_accs, yerr=balanced_stds, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Chance Level')
    ax.set_ylabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison (Cross-Validation)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, balanced_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Feature importance
    ax2 = axes[1]
    if feature_importance is not None:
        indices = np.argsort(feature_importance)[::-1][:15]
        top_features = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in indices]
        top_importance = feature_importance[indices]
        
        y_pos = np.arange(len(top_features))
        ax2.barh(y_pos, top_importance, color='teal', alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_features, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax2.set_title(f'Top 15 Features ({best_model_name})', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, f'Feature importance\nnot available for {best_model_name}',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.axis('off')
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Saved -> {plot_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Compare EEG attention classifiers and train best model'
    )
    parser.add_argument('--subject_dir', '-s', type=str, default=None,
                       help='Path to subject folder (trains on all sessions)')
    parser.add_argument('--data_root', '-r', type=str, default='data/rsvp',
                       help='Path to data root containing sub-*/ses-* directories')
    parser.add_argument('--output', '-o', type=str, default='cache/best_model.pkl',
                       help='Output path for trained model')
    parser.add_argument('--plot_dir', '-p', type=str, default='cache',
                       help='Directory to save plots')
    args = parser.parse_args()
    
    # Load data
    X, y = load_data(subject_dir=args.subject_dir, data_root=args.data_root)
    
    # Compare models
    best_model_name, best_pipeline, results = compare_models(X, y)
    
    if best_model_name is None:
        print("\n[FAILED] Could not compare models - check data quality")
        return 1
    
    # Train best model
    trained_model, feature_importance = train_best_model(
        X, y, best_model_name, best_pipeline, save_path=args.output
    )
    
    # Plot results
    feature_names = get_feature_names(n_ch=8)
    plot_comparison(results, best_model_name, feature_importance, 
                   feature_names, save_dir=args.plot_dir)
    
    print("\n[SUCCESS] Model comparison and training complete!")
    return 0


if __name__ == '__main__':
    exit(main())

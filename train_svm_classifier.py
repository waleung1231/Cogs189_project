#!/usr/bin/env python3
"""
SVM Classifier for RSVP Attention Detection
============================================
Trains an SVM to classify focused vs zoned-out states based on EEG epochs.

Labels:
- Focused (0): Hit with fast reaction time on target letter 'X'
- Zoned-out (1): Miss on target letter 'X' OR hit with slow reaction time

Usage:
    python train_svm_classifier.py                              # Default: all sessions in data/sub-01
    python train_svm_classifier.py --subject_dir data/sub-01    # Train on entire subject folder
    python train_svm_classifier.py --session_dir data/sub-01/ses-01  # Single session only
"""

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy import signal as spsig
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

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SAMPLING_RATE = 250  # Hz
EPOCH_PRE_S = 0.200
EPOCH_PRE_SAMP = round(EPOCH_PRE_S * SAMPLING_RATE)
TARGET_LETTER = 'X'  # The action letter used in the experiment


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
    # Filter to target letter trials only
    target_df = metadata_df[metadata_df['letter'] == target_letter].copy()
    
    if len(target_df) == 0:
        print(f"[ERROR] No trials found for target letter '{target_letter}'")
        return pd.DataFrame()
    
    # Calculate RT threshold from hits
    hit_rts = target_df.loc[target_df['response'] == 'hit', 'rt'].dropna()
    
    if len(hit_rts) == 0:
        # No hits - all targets are zoned-out (misses)
        target_df['label'] = 1
        target_df['label_name'] = 'zoned_out'
        return target_df
    
    rt_mean = hit_rts.mean()
    rt_std = hit_rts.std() if len(hit_rts) > 1 else 0.0
    slow_threshold = rt_mean + 1.5 * rt_std
    
    print(f"\n[LABELING] Target letter: '{target_letter}'")
    print(f"  RT mean: {rt_mean*1000:.0f} ms")
    print(f"  RT std:  {rt_std*1000:.0f} ms")
    print(f"  Slow threshold: {slow_threshold*1000:.0f} ms")
    
    def label_row(row):
        if row['response'] == 'miss':
            return 1  # Zoned-out: missed the target
        elif row['response'] == 'hit':
            if pd.isna(row['rt']):
                return np.nan
            return 1 if row['rt'] > slow_threshold else 0  # Slow = zoned-out
        return np.nan  # Other responses (shouldn't happen for targets)
    
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
    """Load EEG epochs and metadata from a single session, extract features and labels."""
    print(f"\n[LOADING] {session_dir}")
    
    # Load epochs
    epochs_path = os.path.join(session_dir, 'eeg_epochs.npy')
    if not os.path.exists(epochs_path):
        print(f"  [SKIP] No eeg_epochs.npy found")
        return None, None
    
    epochs = np.load(epochs_path, allow_pickle=True)
    print(f"  Epochs shape: {epochs.shape}")
    
    # Load metadata
    meta_path = os.path.join(session_dir, 'metadata.csv')
    if not os.path.exists(meta_path):
        print(f"  [SKIP] No metadata.csv found")
        return None, None
    
    metadata = pd.read_csv(meta_path)
    print(f"  Total trials: {len(metadata)}")
    
    # Assign labels (only for target letter trials)
    labeled_df = assign_labels(metadata)
    
    if len(labeled_df) == 0:
        print(f"  [SKIP] No labeled trials found")
        return None, None
    
    # Drop NaN labels
    labeled_df = labeled_df.dropna(subset=['label'])
    
    if len(labeled_df) == 0:
        print(f"  [SKIP] No valid labels after filtering")
        return None, None
    
    # Get indices of labeled trials
    trial_indices = labeled_df['index'].astype(int).values
    labels = labeled_df['label'].astype(int).values
    
    # Extract corresponding epochs
    labeled_epochs = epochs[trial_indices]
    
    # Remove epochs with NaN values
    valid_mask = ~np.any(np.isnan(labeled_epochs.reshape(len(labeled_epochs), -1)), axis=1)
    labeled_epochs = labeled_epochs[valid_mask]
    labels = labels[valid_mask]
    
    if len(labels) == 0:
        print(f"  [SKIP] No valid epochs after NaN removal")
        return None, None
    
    # Class distribution for this session
    unique, counts = np.unique(labels, return_counts=True)
    class_str = ', '.join([f"{'Focused' if u == 0 else 'Zoned-out'}: {c}" for u, c in zip(unique, counts)])
    print(f"  Valid trials: {len(labels)} ({class_str})")
    
    # Extract features
    X = extract_features(labeled_epochs)
    
    # Remove any trials with NaN features
    feat_valid = ~np.any(np.isnan(X), axis=1)
    X = X[feat_valid]
    y = labels[feat_valid]
    
    return X, y


def load_data(subject_dir=None, session_dir=None):
    """
    Load EEG data from either a subject folder (all sessions) or a single session.
    
    Args:
        subject_dir: Path to subject folder containing ses-* subdirectories
        session_dir: Path to a single session directory
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
    """
    if session_dir:
        # Load single session
        X, y = load_session(session_dir)
        if X is None:
            raise ValueError(f"No valid data found in {session_dir}")
        return X, y
    
    if subject_dir:
        # Discover and load all sessions
        sessions = discover_sessions(subject_dir)
        
        if not sessions:
            raise FileNotFoundError(f"No session directories (ses-*) found in {subject_dir}")
        
        print(f"\n{'='*60}")
        print(f"LOADING SUBJECT DATA: {subject_dir}")
        print(f"Found {len(sessions)} session(s)")
        print('='*60)
        
        all_X = []
        all_y = []
        
        for session in sessions:
            X_sess, y_sess = load_session(session)
            if X_sess is not None and len(y_sess) > 0:
                all_X.append(X_sess)
                all_y.append(y_sess)
        
        if not all_X:
            raise ValueError(f"No valid data found in any session under {subject_dir}")
        
        # Concatenate all sessions
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        # Print combined summary
        print(f"\n{'='*60}")
        print("COMBINED DATA SUMMARY")
        print('='*60)
        print(f"  Sessions loaded: {len(all_X)}")
        print(f"  Total samples: {len(y)}")
        print(f"  Feature matrix: {X.shape}")
        
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            label_name = 'Focused' if u == 0 else 'Zoned-out'
            print(f"  {label_name} (class {u}): {c} trials ({c/len(y)*100:.1f}%)")
        
        return X, y
    
    raise ValueError("Either subject_dir or session_dir must be provided")


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────
def train_svm(X, y, save_path=None):
    """Train SVM classifier with cross-validation."""
    print("\n" + "="*60)
    print("TRAINING SVM CLASSIFIER")
    print("="*60)
    
    # Check class distribution
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("\n[ERROR] Only one class present in the data!")
        print("        Need both focused and zoned-out trials for training.")
        return None
    
    # Build pipeline: StandardScaler + SVM with RBF kernel
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', 
                   probability=True, random_state=42))
    ])
    
    # Cross-validation
    min_class_count = min(np.bincount(y))
    n_splits = min(5, min_class_count)
    
    if n_splits < 2:
        print(f"\n[WARN] Not enough samples for cross-validation (min class has {min_class_count} samples)")
        print("       Training on full dataset without CV...")
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        accuracy = accuracy_score(y, y_pred)
        balanced_acc = balanced_accuracy_score(y, y_pred)
    else:
        print(f"\n[CROSS-VALIDATION] {n_splits}-fold stratified CV")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Compute CV scores
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        cv_balanced = cross_val_score(pipeline, X, y, cv=cv, scoring='balanced_accuracy')
        
        print(f"\n[CV RESULTS]")
        print(f"  Accuracy:          {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Balanced Accuracy: {cv_balanced.mean():.3f} ± {cv_balanced.std():.3f}")
        print(f"  Per-fold accuracy: {[f'{s:.3f}' for s in cv_scores]}")
        
        accuracy = cv_scores.mean()
        balanced_acc = cv_balanced.mean()
        
        # Train final model on all data
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
    
    # Final evaluation on training set (sanity check)
    print("\n[TRAINING SET EVALUATION]")
    print(classification_report(y, y_pred, target_names=['Focused', 'Zoned-out']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"                 Predicted")
    print(f"                 Focused  Zoned-out")
    print(f"  Actual Focused    {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"  Actual Zoned-out  {cm[1,0]:3d}       {cm[1,1]:3d}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Model: SVM with RBF kernel")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(y)} ({sum(y==0)} focused, {sum(y==1)} zoned-out)")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Balanced Accuracy: {balanced_acc:.1%}")
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': pipeline,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'n_features': X.shape[1],
                'n_samples': len(y),
            }, f)
        print(f"\n[SAVED] Model -> {save_path}")
    
    return pipeline


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train SVM classifier for attention detection')
    parser.add_argument('--subject_dir', '-s', type=str, default='data/rsvp/sub-01',
                       help='Path to subject folder (trains on all sessions)')
    parser.add_argument('--session_dir', '-d', type=str, default=None,
                       help='Path to single session directory (overrides subject_dir)')
    parser.add_argument('--output', '-o', type=str, default='cache/svm_model.pkl',
                       help='Output path for trained model')
    args = parser.parse_args()
    
    # Load data from subject folder (all sessions) or single session
    X, y = load_data(subject_dir=args.subject_dir, session_dir=args.session_dir)
    
    # Train SVM
    model = train_svm(X, y, save_path=args.output)
    
    if model is None:
        print("\n[FAILED] Could not train model - check data quality")
        return 1
    
    print("\n[SUCCESS] Training complete!")
    return 0


if __name__ == '__main__':
    exit(main())

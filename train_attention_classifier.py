# ============================================================
#  RSVP Attention Classifier — Training Script
#  Trains a model to predict focused vs. zoned-out trials
#  from EEG epochs saved by run_rsvp.py
# ============================================================
#
#  LABELLING STRATEGY
#  ------------------
#  "Zoned out" (label=1) is inferred from behaviour:
#    • Miss  : target appeared but no spacebar press
#    • Slow RT: reaction time > mean + 1.5 × SD  (sluggish attention)
#  "Focused" (label=0):
#    • Hit with RT within the normal range
#
#  Only TARGET letter epochs are labelled; non-target epochs
#  are not directly labelled (no ground truth) but can be
#  used for unsupervised / semi-supervised extensions later.
#
#  FEATURES
#  --------
#  Per epoch (8 ch × 1000 samples @ 250 Hz):
#    1. P300 amplitude window mean  (300–500 ms post-onset, Pz-like ch)
#    2. Mean power in theta band    (4–8 Hz)
#    3. Mean power in alpha band    (8–13 Hz)
#    4. Mean power in beta band     (13–30 Hz)
#    5. Frontal alpha asymmetry     (FAA)
#    6. ERP peak latency            (max abs amplitude 200–600 ms)
#    7. Epoch variance              (artefact proxy)
#
#  MODEL
#  -----
#  RandomForest + optional SVM/LDA for comparison.
#  Evaluated with leave-one-subject-out or stratified k-fold.
#
#  USAGE
#  -----
#    python train_attention_classifier.py \
#        --data_dir data/rsvp/sub-01/ses-01 \
#        --out_model cache/attention_model.pkl
# ============================================================

import os, argparse, pickle, warnings
import numpy as np
import pandas as pd
from scipy import signal as spsig
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SAMPLING_RATE   = 250
EPOCH_PRE_S     = 0.200
EPOCH_PRE_SAMP  = round(EPOCH_PRE_S * SAMPLING_RATE)  # 50 samples

# ──────────────────────────────────────────────
#  FEATURE EXTRACTION
# ──────────────────────────────────────────────

def bandpower(epoch_1ch, sfreq, band):
    """Mean PSD power in a frequency band for a single channel epoch."""
    f, psd = spsig.welch(epoch_1ch, fs=sfreq, nperseg=min(256, len(epoch_1ch)))
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(psd[idx], f[idx])


def extract_features(epochs, sfreq=SAMPLING_RATE, pre_samp=EPOCH_PRE_SAMP):
    """
    epochs : (n_trials, n_channels, n_times)
    Returns feature matrix of shape (n_trials, n_features)
    """
    n_trials, n_ch, n_times = epochs.shape
    features = []

    # Time axis relative to stimulus onset (in samples)
    # epoch layout: [pre_samp samples | post_samp samples]
    post_start = pre_samp

    for ep in epochs:
        if np.any(np.isnan(ep)):
            features.append([np.nan] * 21)  # placeholder; filtered later
            continue

        post_epoch = ep[:, post_start:]          # post-stimulus portion

        # ── 1-3: Band power (averaged over all channels) ──────────────
        theta_pows = [bandpower(post_epoch[ch], sfreq, (4, 8))  for ch in range(n_ch)]
        alpha_pows = [bandpower(post_epoch[ch], sfreq, (8, 13)) for ch in range(n_ch)]
        beta_pows  = [bandpower(post_epoch[ch], sfreq, (13, 30)) for ch in range(n_ch)]
        mean_theta = np.mean(theta_pows)
        mean_alpha = np.mean(alpha_pows)
        mean_beta  = np.mean(beta_pows)

        # ── 4: P300 amplitude (ch 0-7 mean, 300-500 ms post-onset) ────
        p300_start = round(0.300 * sfreq)
        p300_end   = round(0.500 * sfreq)
        if p300_end <= post_epoch.shape[1]:
            p300_amp = np.mean(post_epoch[:, p300_start:p300_end])
        else:
            p300_amp = np.nan

        # ── 5: ERP peak latency (max abs amplitude 200-600 ms) ────────
        lat_start = round(0.200 * sfreq)
        lat_end   = round(0.600 * sfreq)
        if lat_end <= post_epoch.shape[1]:
            mean_erp = np.mean(post_epoch[:, lat_start:lat_end], axis=0)
            peak_lat = np.argmax(np.abs(mean_erp)) / sfreq  # in seconds
        else:
            peak_lat = np.nan

        # ── 6: Frontal alpha asymmetry (channels 0 vs 1 as proxy) ─────
        # Assumes ch0 = left frontal, ch1 = right frontal (check your montage)
        faa = np.log(alpha_pows[0] + 1e-10) - np.log(alpha_pows[1] + 1e-10) if n_ch >= 2 else 0.0

        # ── 7: Epoch variance (artefact / alertness proxy) ────────────
        epoch_var = np.mean(np.var(post_epoch, axis=1))

        # ── 8-15: Per-channel alpha power (8 features) ────────────────
        per_ch_alpha = alpha_pows  # list of n_ch values

        # ── 16-20: Pre-stimulus alpha (baseline alertness) ────────────
        pre_epoch = ep[:, :pre_samp]
        pre_alpha = [bandpower(pre_epoch[ch], sfreq, (8, 13)) for ch in range(n_ch)]
        mean_pre_alpha = np.mean(pre_alpha)

        feat_vec = [
            mean_theta,
            mean_alpha,
            mean_beta,
            p300_amp if not np.isnan(p300_amp) else 0.0,
            peak_lat  if not np.isnan(peak_lat)  else 0.5,
            faa,
            epoch_var,
            mean_pre_alpha,
        ] + per_ch_alpha + pre_alpha  # +16 channel-wise features

        features.append(feat_vec)

    return np.array(features, dtype=np.float32)


# ──────────────────────────────────────────────
#  LABELLING
# ──────────────────────────────────────────────

def assign_labels(metadata_df):
    """
    Returns a boolean Series: True = zoned_out, False = focused.
    Only rows where is_target==1 get meaningful labels.
    """
    df = metadata_df.copy()
    # Use only target trials
    target = df[df['is_target'] == 1].copy()

    # Compute RT stats from hits only
    hit_rts = target.loc[target['response'] == 'hit', 'rt'].dropna()
    rt_mean = hit_rts.mean()
    rt_std  = hit_rts.std()
    slow_threshold = rt_mean + 1.5 * rt_std

    def label_row(row):
        if row['response'] == 'miss':
            return 1   # zoned out
        elif row['response'] == 'hit':
            return 1 if row['rt'] > slow_threshold else 0
        else:
            return np.nan  # false alarm on target – exclude

    target['zoned_out'] = target.apply(label_row, axis=1)
    return target


# ──────────────────────────────────────────────
#  MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────

def train(data_dir, model_out_path, plot=True):
    print(f'\n[LOAD] {data_dir}')
    epochs_path  = os.path.join(data_dir, 'eeg_epochs.npy')
    meta_path    = os.path.join(data_dir, 'metadata.csv')

    epochs = np.load(epochs_path, allow_pickle=True)
    meta   = pd.read_csv(meta_path)

    print(f'  Epochs shape   : {epochs.shape}')
    print(f'  Metadata rows  : {len(meta)}')

    # ── Label target trials ───────────────────
    target_df = assign_labels(meta)
    target_df = target_df.dropna(subset=['zoned_out'])

    print(f'\n  Labelled target trials : {len(target_df)}')
    print(f'  Focused  (0) : {(target_df["zoned_out"]==0).sum()}')
    print(f'  Zoned-out(1) : {(target_df["zoned_out"]==1).sum()}')

    # ── Extract features for target epochs only ───
    target_indices = target_df['index'].values
    target_epochs  = epochs[target_indices]
    labels         = target_df['zoned_out'].values.astype(int)

    # Drop any NaN epochs (insufficient EEG data at session edges)
    valid_mask    = ~np.any(np.isnan(target_epochs.reshape(len(target_epochs), -1)), axis=1)
    target_epochs = target_epochs[valid_mask]
    labels        = labels[valid_mask]
    print(f'  Valid epochs after NaN drop: {len(target_epochs)}')

    print('\n[FEATURES] Extracting…')
    X = extract_features(target_epochs)
    y = labels

    # Drop any feature rows that contain NaN
    feat_valid = ~np.any(np.isnan(X), axis=1)
    X, y = X[feat_valid], y[feat_valid]
    print(f'  Feature matrix : {X.shape}')

    if len(np.unique(y)) < 2:
        print('[WARN] Only one class in labels – cannot train a classifier.')
        print('       Collect more data or adjust the slow-RT threshold.')
        return

    # ── Cross-validation comparison ───────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'RandomForest' : Pipeline([('scaler', StandardScaler()),
                                    ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))]),
        'LDA'          : Pipeline([('scaler', StandardScaler()),
                                    ('clf', LinearDiscriminantAnalysis())]),
        'SVM-RBF'      : Pipeline([('scaler', StandardScaler()),
                                    ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True))]),
        'GradBoost'    : Pipeline([('scaler', StandardScaler()),
                                    ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))]),
    }

    print('\n[CV] 5-fold stratified cross-validation')
    best_score = 0
    best_name  = None
    cv_results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='balanced_accuracy')
        cv_results[name] = scores
        mean_s = scores.mean()
        print(f'  {name:15s}  balanced_acc = {mean_s:.3f} ± {scores.std():.3f}')
        if mean_s > best_score:
            best_score = mean_s
            best_name  = name

    print(f'\n  Best model: {best_name}  ({best_score:.3f})')

    # ── Fit best model on all data & save ─────
    best_pipe = models[best_name]
    best_pipe.fit(X, y)

    os.makedirs(os.path.dirname(model_out_path) if os.path.dirname(model_out_path) else '.', exist_ok=True)
    with open(model_out_path, 'wb') as f:
        pickle.dump({'model': best_pipe, 'model_name': best_name,
                     'feature_names': _feature_names(),
                     'cv_results': cv_results}, f)
    print(f'\n[SAVED] Model → {model_out_path}')

    # ── Detailed report on full fit ───────────
    y_pred = best_pipe.predict(X)
    print('\n[REPORT] (on full training set – for sanity check)')
    print(classification_report(y, y_pred, target_names=['Focused', 'Zoned-out']))
    print('Confusion matrix:')
    print(confusion_matrix(y, y_pred))

    # ── Feature importance plot (RF only) ─────
    if plot:
        _plot_results(cv_results, best_pipe, best_name, data_dir)


def _feature_names():
    ch = [f'ch{i}' for i in range(8)]
    return (['theta_mean','alpha_mean','beta_mean','p300_amp',
             'erp_peak_lat','FAA','epoch_var','pre_alpha_mean']
            + [f'alpha_{c}' for c in ch]
            + [f'pre_alpha_{c}' for c in ch])


def _plot_results(cv_results, best_pipe, best_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # CV bar chart
    ax = axes[0]
    names  = list(cv_results.keys())
    means  = [cv_results[n].mean() for n in names]
    stds   = [cv_results[n].std()  for n in names]
    bars = ax.bar(names, means, yerr=stds, capsize=4, color='steelblue', alpha=0.8)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Chance')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('5-Fold CV Model Comparison')
    ax.set_ylim(0, 1)
    ax.legend()

    # Feature importance (RF only)
    ax2 = axes[1]
    if 'RandomForest' in best_name:
        rf  = best_pipe.named_steps['clf']
        imp = rf.feature_importances_
        fnames = _feature_names()[:len(imp)]
        idx = np.argsort(imp)[::-1][:15]
        ax2.barh([fnames[i] for i in idx[::-1]], imp[idx[::-1]], color='teal', alpha=0.8)
        ax2.set_xlabel('Importance')
        ax2.set_title(f'Top-15 Feature Importances ({best_name})')
    else:
        ax2.text(0.5, 0.5, f'Feature importance\nnot available for {best_name}',
                 ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'classifier_results.png')
    plt.savefig(plot_path, dpi=120)
    print(f'[PLOT] Saved → {plot_path}')
    plt.close()


# ──────────────────────────────────────────────
#  CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RSVP attention classifier')
    parser.add_argument('--data_dir',  default='data/rsvp/sub-01/ses-01',
                        help='Directory containing eeg_epochs.npy and metadata.csv')
    parser.add_argument('--out_model', default='cache/attention_model.pkl',
                        help='Where to save the trained model')
    parser.add_argument('--no_plot',   action='store_true',
                        help='Skip saving the results plot')
    args = parser.parse_args()

    train(args.data_dir, args.out_model, plot=not args.no_plot)

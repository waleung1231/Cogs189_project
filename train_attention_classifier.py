#!/usr/bin/env python3
"""
RSVP Attention Classifier - Training and Report Generator
=========================================================

Trains and compares four models for focused (0) vs zoned-out (1) EEG state:
- RandomForest
- LDA
- SVM (RBF)
- GradientBoosting

Outputs report artifacts for write-ups:
- model_metrics.csv
- class_recalls.csv
- top_features.csv
- channel_importance.csv
- training_report.json
- model_comparison.png
- feature_importance.png
- best_model_confusion.png

Examples:
    uv run python train_attention_classifier.py --data_root data/rsvp
    uv run python train_attention_classifier.py --data_root data/rsvp --save_model_by performance
"""

import argparse
import glob
import json
import os
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as spsig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    ImbPipeline = Pipeline

warnings.filterwarnings("ignore")

# EEG constants
SAMPLING_RATE = 250
EPOCH_PRE_S = 0.200
EPOCH_PRE_SAMP = round(EPOCH_PRE_S * SAMPLING_RATE)

# Must match run_rsvp.py channel order
CHANNEL_NAMES = ["O1", "O2", "T5", "P3", "Pz", "P4", "T6", "REF"]

# Metadata schema
REQUIRED_META_COLUMNS = {"index", "is_target", "response", "rt"}
RECOMMENDED_META_COLUMNS = {
    "trial_start_wall",
    "flip_on_wall",
    "flip_off_wall",
    "isi_s",
    "keypress_count",
    "keypress_times",
    "keypress_phase",
    "block_idx",
    "quality_flag",
}

# Model complexity factor for utilization ranking
MODEL_COMPLEXITY_FACTOR = {
    "LDA": 1.00,
    "SVM": 0.85,
    "RandomForest": 0.75,
    "GradientBoosting": 0.65,
}


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------
def bandpower(epoch_1ch, sfreq, band):
    f, psd = spsig.welch(epoch_1ch, fs=sfreq, nperseg=min(256, len(epoch_1ch)))
    idx = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx):
        return 0.0
    return float(np.trapezoid(psd[idx], f[idx]))


def _compute_coherence(ch1, ch2, sfreq, band):
    f, coh = spsig.coherence(ch1, ch2, fs=sfreq, nperseg=min(128, len(ch1)))
    idx = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx):
        return 0.0
    return float(np.mean(coh[idx]))


def _safe_skew(arr):
    if len(arr) < 3:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3))


def _safe_kurtosis(arr):
    if len(arr) < 4:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3)


def _get_feature_count(n_ch):
    return 21 + 4 * n_ch


def extract_features(epochs, sfreq=SAMPLING_RATE, pre_samp=EPOCH_PRE_SAMP):
    n_trials, n_ch, _ = epochs.shape
    features = []

    for ep in epochs:
        if np.any(np.isnan(ep)):
            features.append([np.nan] * _get_feature_count(n_ch))
            continue

        post_epoch = ep[:, pre_samp:]
        n_post = post_epoch.shape[1]

        theta_pows = [bandpower(post_epoch[ch], sfreq, (4, 8)) for ch in range(n_ch)]
        alpha_pows = [bandpower(post_epoch[ch], sfreq, (8, 13)) for ch in range(n_ch)]
        beta_pows = [bandpower(post_epoch[ch], sfreq, (13, 30)) for ch in range(n_ch)]
        gamma_pows = [bandpower(post_epoch[ch], sfreq, (30, 45)) for ch in range(n_ch)]

        mean_theta = float(np.mean(theta_pows))
        mean_alpha = float(np.mean(alpha_pows))
        mean_beta = float(np.mean(beta_pows))
        mean_gamma = float(np.mean(gamma_pows))

        n200_start = round(0.150 * sfreq)
        n200_end = round(0.250 * sfreq)
        n200_amp = (
            float(np.mean(post_epoch[:, n200_start:n200_end])) if n200_end <= n_post else 0.0
        )

        p300_start = round(0.300 * sfreq)
        p300_end = round(0.500 * sfreq)
        p300_amp = (
            float(np.mean(post_epoch[:, p300_start:p300_end])) if p300_end <= n_post else 0.0
        )

        lpp_start = round(0.400 * sfreq)
        lpp_end = round(0.800 * sfreq)
        lpp_amp = (
            float(np.mean(post_epoch[:, lpp_start:lpp_end])) if lpp_end <= n_post else 0.0
        )

        lat_start = round(0.200 * sfreq)
        lat_end = round(0.600 * sfreq)
        if lat_end <= n_post:
            mean_erp = np.mean(post_epoch[:, lat_start:lat_end], axis=0)
            peak_lat = float(np.argmax(np.abs(mean_erp)) / sfreq)
        else:
            peak_lat = 0.5

        if n_ch >= 6:
            # Posterior alpha asymmetry proxy from left-right parietal channels.
            # P3 is now at index 3, P4 is now at index 5
            paa = float(np.log(alpha_pows[3] + 1e-10) - np.log(alpha_pows[5] + 1e-10))
        elif n_ch >= 2:
            # Use O1 (index 0) and O2 (index 1) for asymmetry when fewer channels
            paa = float(np.log(alpha_pows[0] + 1e-10) - np.log(alpha_pows[1] + 1e-10))
        else:
            paa = 0.0

        epoch_var = float(np.mean(np.var(post_epoch, axis=1)))
        epoch_std = float(np.mean(np.std(post_epoch, axis=1)))
        epoch_skew = float(np.mean([_safe_skew(post_epoch[ch]) for ch in range(n_ch)]))
        epoch_kurt = float(np.mean([_safe_kurtosis(post_epoch[ch]) for ch in range(n_ch)]))

        mean_amp = float(np.mean(post_epoch))
        max_amp = float(np.max(np.abs(post_epoch)))
        zero_crossings = float(
            np.mean([np.sum(np.diff(np.sign(post_epoch[ch])) != 0) for ch in range(n_ch)])
        )

        if n_ch >= 2:
            alpha_coh = _compute_coherence(post_epoch[0], post_epoch[1], sfreq, (8, 13))
            theta_coh = _compute_coherence(post_epoch[0], post_epoch[1], sfreq, (4, 8))
        else:
            alpha_coh = 0.0
            theta_coh = 0.0

        pre_epoch = ep[:, :pre_samp]
        pre_alpha = [bandpower(pre_epoch[ch], sfreq, (8, 13)) for ch in range(n_ch)]
        mean_pre_alpha = float(np.mean(pre_alpha))
        alpha_change = mean_alpha - mean_pre_alpha

        first_half = post_epoch[:, : n_post // 2]
        second_half = post_epoch[:, n_post // 2 :]
        erp_slope = float(np.mean(second_half) - np.mean(first_half))

        feat_vec = [
            mean_theta,
            mean_alpha,
            mean_beta,
            mean_gamma,
            n200_amp,
            p300_amp,
            lpp_amp,
            peak_lat,
            paa,
            epoch_var,
            epoch_std,
            epoch_skew,
            epoch_kurt,
            mean_amp,
            max_amp,
            zero_crossings,
            alpha_coh,
            theta_coh,
            mean_pre_alpha,
            alpha_change,
            erp_slope,
        ] + alpha_pows + theta_pows + beta_pows + pre_alpha

        features.append(feat_vec)

    return np.array(features, dtype=np.float32)


def _feature_names(n_ch=8):
    ch = CHANNEL_NAMES[:n_ch]
    return [
        "theta_mean",
        "alpha_mean",
        "beta_mean",
        "gamma_mean",
        "n200_amp",
        "p300_amp",
        "lpp_amp",
        "erp_peak_lat",
        "PAA",
        "epoch_var",
        "epoch_std",
        "epoch_skew",
        "epoch_kurt",
        "mean_amp",
        "max_amp",
        "zero_crossings",
        "alpha_coh",
        "theta_coh",
        "pre_alpha_mean",
        "alpha_change",
        "erp_slope",
    ] + [f"alpha_{c}" for c in ch] + [f"theta_{c}" for c in ch] + [f"beta_{c}" for c in ch] + [f"pre_alpha_{c}" for c in ch]


# ------------------------------------------------------------
# Labels and data loading
# ------------------------------------------------------------
def validate_metadata_schema(meta_df):
    missing_required = REQUIRED_META_COLUMNS - set(meta_df.columns)
    if missing_required:
        raise ValueError(f"Missing required metadata columns: {sorted(missing_required)}")

    missing_recommended = RECOMMENDED_META_COLUMNS - set(meta_df.columns)
    if missing_recommended:
        print(f"[WARN] Recommended metadata columns missing: {sorted(missing_recommended)}")


def assign_labels(metadata_df):
    target = metadata_df[metadata_df["is_target"] == 1].copy()

    hit_rts = target.loc[target["response"] == "hit", "rt"].dropna()
    if len(hit_rts) == 0:
        target["zoned_out"] = np.where(target["response"] == "miss", 1, np.nan)
        return target

    rt_mean = hit_rts.mean()
    rt_std = hit_rts.std() if len(hit_rts) > 1 else 0.0
    slow_threshold = rt_mean + 1.5 * rt_std

    def label_row(row):
        if row["response"] == "miss":
            return 1
        if row["response"] == "hit":
            return 1 if row["rt"] > slow_threshold else 0
        return np.nan

    target["zoned_out"] = target.apply(label_row, axis=1)
    return target


def _extract_session_id(data_dir):
    parts = os.path.normpath(data_dir).split(os.sep)
    sub = next((p for p in parts if p.startswith("sub-")), "sub-unknown")
    ses = next((p for p in parts if p.startswith("ses-")), "ses-unknown")
    return f"{sub}_{ses}"


def _load_one_session(data_dir):
    epochs_path = os.path.join(data_dir, "eeg_epochs.npy")
    meta_path = os.path.join(data_dir, "metadata.csv")

    if not os.path.exists(epochs_path):
        raise FileNotFoundError(f"Missing file: {epochs_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing file: {meta_path}")

    epochs = np.load(epochs_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)
    validate_metadata_schema(meta)
    
    # Keep all channels in the current 8-channel format.
    # NOTE: legacy 7-channel datasets (O2, T5, P3, Pz, P4, T6, Fz) are loaded as-is.

    target_df = assign_labels(meta).dropna(subset=["zoned_out"])
    target_indices = target_df["index"].astype(int).values

    target_epochs = epochs[target_indices]
    labels = target_df["zoned_out"].values.astype(int)

    valid_mask = ~np.any(np.isnan(target_epochs.reshape(len(target_epochs), -1)), axis=1)
    target_epochs = target_epochs[valid_mask]
    labels = labels[valid_mask]

    X = extract_features(target_epochs)
    feat_valid = ~np.any(np.isnan(X), axis=1)
    X = X[feat_valid]
    y = labels[feat_valid]

    return X, y


def _discover_session_dirs(data_root):
    pattern = os.path.join(data_root, "sub-*", "ses-*")
    return sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])


def _load_dataset(data_dir=None, data_root=None):
    if data_root:
        session_dirs = _discover_session_dirs(data_root)
        if not session_dirs:
            raise FileNotFoundError(f"No session directories found under {data_root}")

        all_x, all_y, groups = [], [], []
        for session_dir in session_dirs:
            print(f"[LOAD] {session_dir}")
            x_s, y_s = _load_one_session(session_dir)
            if len(y_s) == 0:
                print("  [WARN] No valid labeled target trials, skipping session")
                continue

            session_id = _extract_session_id(session_dir)
            all_x.append(x_s)
            all_y.append(y_s)
            groups.append(np.array([session_id] * len(y_s), dtype=object))
            cls = dict(zip(*np.unique(y_s, return_counts=True)))
            print(f"  [OK] X={x_s.shape}, classes={cls}")

        if not all_x:
            raise ValueError("No usable labeled data found across sessions")

        return np.vstack(all_x), np.concatenate(all_y), np.concatenate(groups), session_dirs

    if not data_dir:
        raise ValueError("Either data_dir or data_root must be provided")

    print(f"[LOAD] {data_dir}")
    x_s, y_s = _load_one_session(data_dir)
    cls = dict(zip(*np.unique(y_s, return_counts=True)))
    print(f"  [OK] X={x_s.shape}, classes={cls}")
    return x_s, y_s, None, [data_dir]


# ------------------------------------------------------------
# CV and models
# ------------------------------------------------------------
def _get_stratified_n_splits(y, max_splits=5):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 0
    return int(min(max_splits, counts.min()))


def _build_cv(y, groups=None):
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            return "LeaveOneGroupOut", LeaveOneGroupOut(), groups

    n_splits = _get_stratified_n_splits(y)
    if n_splits < 2:
        return None, None, None

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return f"StratifiedKFold({n_splits})", cv, None


def _build_models(use_smote=True):
    PipelineClass = ImbPipeline if (HAS_IMBLEARN and use_smote) else Pipeline
    pre_steps = [("scaler", StandardScaler())]
    smote_step = (
        [("smote", SMOTE(random_state=42))] if (HAS_IMBLEARN and use_smote) else []
    )
    base_steps = pre_steps + smote_step + [("selector", SelectKBest(mutual_info_classif, k="all"))]

    return {
        "RandomForest": PipelineClass(
            base_steps
            + [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LDA": PipelineClass(
            base_steps + [("clf", LinearDiscriminantAnalysis())]
        ),
        "SVM": PipelineClass(
            base_steps
            + [
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "GradientBoosting": PipelineClass(
            base_steps + [("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))]
        ),
    }


def _get_param_distributions():
    return {
        "RandomForest": {
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
            "clf__max_features": ["sqrt", "log2", None],
            "selector__k": [10, 20, 30, "all"],
        },
        "LDA": {
            "selector__k": [10, 20, 30, "all"],
        },
        "SVM": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.01, 0.1],
            "selector__k": [10, 20, 30, "all"],
        },
        "GradientBoosting": {
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__max_depth": [2, 3, 5],
            "clf__subsample": [0.8, 1.0],
            "selector__k": [10, 20, 30, "all"],
        },
    }


def _count_param_combinations(param_dict):
    count = 1
    for values in param_dict.values():
        count *= len(values) if isinstance(values, list) else 1
    return count


def _tune_model(pipe, param_dist, X, y, cv, cv_groups):
    n_iter = min(24, _count_param_combinations(param_dist))
    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="balanced_accuracy",
        random_state=42,
        n_jobs=-1,
        error_score="raise",
    )
    if cv_groups is not None:
        search.fit(X, y, groups=cv_groups)
    else:
        search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def _evaluate_model_cv(pipe, X, y, cv, cv_groups):
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }
    cv_stats = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        groups=cv_groups,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    y_pred = cross_val_predict(
        pipe,
        X,
        y,
        cv=cv,
        groups=cv_groups,
        method="predict",
        n_jobs=-1,
    )

    conf = confusion_matrix(y, y_pred, labels=[0, 1])
    rec_focus = conf[0, 0] / conf[0].sum() if conf[0].sum() > 0 else 0.0
    rec_zoned = conf[1, 1] / conf[1].sum() if conf[1].sum() > 0 else 0.0

    metrics = {
        "accuracy_mean": float(np.mean(cv_stats["test_accuracy"])),
        "accuracy_std": float(np.std(cv_stats["test_accuracy"])),
        "balanced_accuracy_mean": float(np.mean(cv_stats["test_balanced_accuracy"])),
        "balanced_accuracy_std": float(np.std(cv_stats["test_balanced_accuracy"])),
        "fit_time_mean": float(np.mean(cv_stats["fit_time"])),
        "fit_time_std": float(np.std(cv_stats["fit_time"])),
        "focused_recall": float(rec_focus),
        "zoned_out_recall": float(rec_zoned),
        "oof_accuracy": float(accuracy_score(y, y_pred)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
    }

    report_dict = classification_report(
        y,
        y_pred,
        target_names=["Focused", "Zoned-out"],
        output_dict=True,
        zero_division=0,
    )

    return metrics, conf, y_pred, report_dict


def _compute_speed_scores(results_df):
    fastest = results_df["fit_time_mean"].min()
    return fastest / results_df["fit_time_mean"].clip(lower=1e-8)


def _compute_utilization_scores(results_df):
    speed_scores = _compute_speed_scores(results_df)
    util_scores = []
    for idx, row in results_df.iterrows():
        complexity = MODEL_COMPLEXITY_FACTOR.get(idx, 0.7)
        util = (
            0.60 * row["balanced_accuracy_mean"]
            + 0.25 * row["zoned_out_recall"]
            + 0.15 * speed_scores.loc[idx]
        ) * complexity
        util_scores.append(util)
    return pd.Series(util_scores, index=results_df.index), speed_scores


# ------------------------------------------------------------
# Importance and plotting
# ------------------------------------------------------------
def _compute_permutation_importance(best_pipe, X, y, feature_names):
    perm = permutation_importance(
        best_pipe,
        X,
        y,
        scoring="balanced_accuracy",
        n_repeats=40,
        random_state=42,
        n_jobs=-1,
    )
    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return imp_df


def _compute_channel_importance(feature_importance_df):
    rows = []
    for channel in CHANNEL_NAMES:
        mask = feature_importance_df["feature"].str.endswith(f"_{channel}")
        channel_sum = feature_importance_df.loc[mask, "importance_mean"].sum()
        rows.append({"channel": channel, "importance_sum": float(channel_sum)})
    return pd.DataFrame(rows).sort_values("importance_sum", ascending=False)


def _save_model_comparison_plot(results_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    names = results_df.index.tolist()

    ax0 = axes[0]
    ax0.bar(
        names,
        results_df["balanced_accuracy_mean"],
        yerr=results_df["balanced_accuracy_std"],
        capsize=4,
        color="#3A7CA5",
    )
    ax0.axhline(0.5, color="#AA3333", linestyle="--", linewidth=1)
    ax0.set_ylabel("Balanced Accuracy")
    ax0.set_ylim(0, 1)
    ax0.set_title("Cross-Validated Performance")
    ax0.tick_params(axis="x", rotation=20)

    ax1 = axes[1]
    x = np.arange(len(names))
    w = 0.38
    ax1.bar(x - w / 2, results_df["focused_recall"], width=w, label="Focused Recall", color="#00A878")
    ax1.bar(x + w / 2, results_df["zoned_out_recall"], width=w, label="Zoned-out Recall", color="#F26419")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Recall")
    ax1.set_title("Class-wise Recall (OOF)")
    ax1.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_confusion_plot(conf, out_path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(conf, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["Focused", "Zoned-out"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_importance_plot(feature_importance_df, channel_importance_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    top_features = feature_importance_df.head(15).iloc[::-1]
    ax0 = axes[0]
    ax0.barh(top_features["feature"], top_features["importance_mean"], color="#2A9D8F")
    ax0.set_xlabel("Permutation Importance")
    ax0.set_title("Top 15 Features")

    top_channels = channel_importance_df.head(8).iloc[::-1]
    ax1 = axes[1]
    ax1.barh(top_channels["channel"], top_channels["importance_sum"], color="#E76F51")
    ax1.set_xlabel("Aggregated Importance")
    ax1.set_title("Channel Contribution")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------
# Training orchestration
# ------------------------------------------------------------
def train(
    data_dir,
    model_out_path=None,
    report_dir=None,
    plot=True,
    data_root=None,
    tune_hyperparams=True,
    save_model_by="utilization",
):
    active_root = data_root if (data_root and os.path.isdir(data_root)) else None
    if data_root and active_root is None:
        print(f"[WARN] data_root not found: {data_root}, falling back to data_dir")

    X, y, groups, used_sessions = _load_dataset(data_dir=data_dir, data_root=active_root)

    if len(np.unique(y)) < 2:
        raise ValueError("Only one class found. Need both focused and zoned-out labels.")

    cv_name, cv, cv_groups = _build_cv(y, groups=groups)
    if cv is None:
        raise ValueError("Not enough samples per class for cross-validation.")

    print(f"\n[DATA] Feature matrix: {X.shape}")
    print(f"[CV] Strategy: {cv_name}")
    if groups is not None:
        print(f"[CV] Sessions: {len(np.unique(groups))}")
    print(f"[SMOTE] {'Enabled' if HAS_IMBLEARN else 'Disabled (imbalanced-learn not installed)'}")

    models = _build_models()
    param_dists = _get_param_distributions()

    fitted_models = {}
    confusions = {}
    class_reports = {}
    results_rows = {}

    for name, pipe in models.items():
        print(f"\n[MODEL] {name}")
        tuned_pipe = pipe
        best_params = {}
        tuning_score = None

        if tune_hyperparams and len(y) >= 50:
            try:
                tuned_pipe, best_params, tuning_score = _tune_model(
                    pipe,
                    param_dists[name],
                    X,
                    y,
                    cv,
                    cv_groups,
                )
                print(f"  [TUNING] Best score: {tuning_score:.3f}")
                print(f"  [TUNING] Best params: {best_params}")
            except Exception as exc:
                print(f"  [TUNING] Failed ({exc}), using default params")

        metrics, conf, _, report_dict = _evaluate_model_cv(tuned_pipe, X, y, cv, cv_groups)
        print(
            "  [CV] balanced_acc={:.3f} +- {:.3f}, focus_recall={:.3f}, zoned_recall={:.3f}".format(
                metrics["balanced_accuracy_mean"],
                metrics["balanced_accuracy_std"],
                metrics["focused_recall"],
                metrics["zoned_out_recall"],
            )
        )

        tuned_pipe.fit(X, y)
        fitted_models[name] = tuned_pipe
        confusions[name] = conf
        class_reports[name] = report_dict

        row = dict(metrics)
        row["tuning_score"] = tuning_score if tuning_score is not None else np.nan
        row["best_params"] = json.dumps(best_params, sort_keys=True)
        results_rows[name] = row

    results_df = pd.DataFrame.from_dict(results_rows, orient="index")
    util_scores, speed_scores = _compute_utilization_scores(results_df)
    results_df["speed_score"] = speed_scores
    results_df["complexity_factor"] = [MODEL_COMPLEXITY_FACTOR.get(n, 0.7) for n in results_df.index]
    results_df["utilization_score"] = util_scores

    best_perf_name = results_df["balanced_accuracy_mean"].idxmax()
    best_util_name = results_df["utilization_score"].idxmax()
    selected_name = best_util_name if save_model_by == "utilization" else best_perf_name

    selected_model = fitted_models[selected_name]
    n_ch_estimate = max(1, (X.shape[1] - 21) // 4)
    feature_names = _feature_names(n_ch=n_ch_estimate)
    if len(feature_names) != X.shape[1]:
        feature_names = [f"f_{i}" for i in range(X.shape[1])]

    feature_imp_df = _compute_permutation_importance(selected_model, X, y, feature_names)
    channel_imp_df = _compute_channel_importance(feature_imp_df)

    focus_rec = float(results_df.loc[selected_name, "focused_recall"])
    zoned_rec = float(results_df.loc[selected_name, "zoned_out_recall"])
    better_class = "Focused" if focus_rec >= zoned_rec else "Zoned-out"

    print("\n[SUMMARY]")
    print(results_df[["balanced_accuracy_mean", "accuracy_mean", "focused_recall", "zoned_out_recall", "fit_time_mean", "utilization_score"]].round(4).to_string())
    print(f"\n[WINNER-PERFORMANCE] {best_perf_name}")
    print(f"[WINNER-UTILIZATION] {best_util_name}")
    print(f"[SELECTED-SAVED-MODEL] {selected_name} (criterion={save_model_by})")
    print(
        f"[CLASS-BEHAVIOR] Selected model recall -> Focused={focus_rec:.3f}, Zoned-out={zoned_rec:.3f}. Better at: {better_class}"
    )

    print("\n[TOP FEATURES]")
    print(feature_imp_df.head(10).round(6).to_string(index=False))
    print("\n[TOP CHANNELS]")
    print(channel_imp_df.head(5).round(6).to_string(index=False))

    if model_out_path is None and report_dir is None:
        print("[INFO] Caching is disabled; skipping saved model and report files.")
    else:
        if model_out_path is not None:
            os.makedirs(os.path.dirname(model_out_path) or ".", exist_ok=True)
            with open(model_out_path, "wb") as f:
                pickle.dump(
                    {
                        "model": selected_model,
                        "model_name": selected_name,
                        "selected_by": save_model_by,
                        "best_performance_model": best_perf_name,
                        "best_utilization_model": best_util_name,
                        "feature_names": feature_names,
                        "cv_strategy": cv_name,
                        "sessions": used_sessions,
                        "model_metrics": results_df.to_dict(orient="index"),
                    },
                    f,
                )
            print(f"\n[SAVED] Model -> {model_out_path}")

        if report_dir is not None:
            os.makedirs(report_dir, exist_ok=True)
            metrics_csv = os.path.join(report_dir, "model_metrics.csv")
            recalls_csv = os.path.join(report_dir, "class_recalls.csv")
            top_feat_csv = os.path.join(report_dir, "top_features.csv")
            channel_csv = os.path.join(report_dir, "channel_importance.csv")
            report_json_path = os.path.join(report_dir, "training_report.json")

            comparison_plot = os.path.join(report_dir, "model_comparison.png")
            importance_plot = os.path.join(report_dir, "feature_importance.png")
            confusion_plot = os.path.join(report_dir, "best_model_confusion.png")

            results_df.to_csv(metrics_csv, index_label="model")
            results_df[["focused_recall", "zoned_out_recall"]].to_csv(recalls_csv, index_label="model")
            feature_imp_df.to_csv(top_feat_csv, index=False)
            channel_imp_df.to_csv(channel_csv, index=False)

            report_payload = {
                "selected_model": selected_name,
                "save_model_by": save_model_by,
                "best_performance_model": best_perf_name,
                "best_utilization_model": best_util_name,
                "cv_strategy": cv_name,
                "sessions": used_sessions,
                "feature_matrix_shape": [int(X.shape[0]), int(X.shape[1])],
                "label_distribution": {
                    "focused": int(np.sum(y == 0)),
                    "zoned_out": int(np.sum(y == 1)),
                },
                "selected_model_class_behavior": {
                    "focused_recall": focus_rec,
                    "zoned_out_recall": zoned_rec,
                    "better_at": better_class,
                },
                "top_features": feature_imp_df.head(20).to_dict(orient="records"),
                "top_channels": channel_imp_df.head(8).to_dict(orient="records"),
                "model_metrics": json.loads(results_df.to_json(orient="index")),
                "best_model_confusion_matrix": confusions[selected_name].tolist(),
                "selected_model_classification_report": class_reports[selected_name],
            }

            with open(report_json_path, "w", encoding="utf-8") as f:
                json.dump(report_payload, f, indent=2)

            if plot:
                _save_model_comparison_plot(results_df, comparison_plot)
                _save_importance_plot(feature_imp_df, channel_imp_df, importance_plot)
                _save_confusion_plot(
                    confusions[selected_name],
                    confusion_plot,
                    f"Confusion Matrix ({selected_name})",
                )
                print(f"[PLOT] {comparison_plot}")
                print(f"[PLOT] {importance_plot}")
                print(f"[PLOT] {confusion_plot}")

            print(f"[REPORT] {metrics_csv}")
            print(f"[REPORT] {recalls_csv}")
            print(f"[REPORT] {top_feat_csv}")
            print(f"[REPORT] {channel_csv}")
            print(f"[REPORT] {report_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RSVP attention classifier and generate report")
    parser.add_argument(
        "--data_dir",
        default="data/rsvp/sub-01/ses-01",
        help="One session directory containing eeg_epochs.npy and metadata.csv",
    )
    parser.add_argument(
        "--data_root",
        default="data/rsvp",
        help="Dataset root dir with sub-*/ses-* (preferred when available)",
    )
    parser.add_argument(
        "--out_model",
        default="cache/attention_model.pkl",
        help="Output path for selected trained model",
    )
    parser.add_argument(
        "--report_dir",
        default="cache",
        help="Directory for CSV/JSON/plot outputs",
    )
    parser.add_argument("--cache", action="store_true", help="Persist model/report artifacts")
    parser.add_argument("--no_plot", action="store_true", help="Skip saving plots")
    parser.add_argument("--no_tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument(
        "--save_model_by",
        choices=["utilization", "performance"],
        default="utilization",
        help="Select which winner to persist",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_out_path=args.out_model if args.cache else None,
        report_dir=args.report_dir if args.cache else None,
        plot=not args.no_plot,
        data_root=args.data_root,
        tune_hyperparams=not args.no_tune,
        save_model_by=args.save_model_by,
    )

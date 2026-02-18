# ============================================================
# RSVP Attention Classifier - Training Script
# ============================================================

import argparse
import glob
import os
import pickle
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as spsig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
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

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

SAMPLING_RATE = 250
EPOCH_PRE_S = 0.200
EPOCH_PRE_SAMP = round(EPOCH_PRE_S * SAMPLING_RATE)

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


def bandpower(epoch_1ch, sfreq, band):
    f, psd = spsig.welch(epoch_1ch, fs=sfreq, nperseg=min(256, len(epoch_1ch)))
    idx = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx):
        return 0.0
    return np.trapz(psd[idx], f[idx])


def _compute_coherence(ch1, ch2, sfreq, band):
    f, coh = spsig.coherence(ch1, ch2, fs=sfreq, nperseg=min(128, len(ch1)))
    idx = np.logical_and(f >= band[0], f <= band[1])
    if not np.any(idx):
        return 0.0
    return float(np.mean(coh[idx]))


def extract_features(epochs, sfreq=SAMPLING_RATE, pre_samp=EPOCH_PRE_SAMP):
    n_trials, n_ch, _ = epochs.shape
    features = []
    post_start = pre_samp

    for ep in epochs:
        if np.any(np.isnan(ep)):
            features.append([np.nan] * _get_feature_count(n_ch))
            continue

        post_epoch = ep[:, post_start:]
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
            float(np.mean(post_epoch[:, n200_start:n200_end]))
            if n200_end <= n_post
            else 0.0
        )

        p300_start = round(0.300 * sfreq)
        p300_end = round(0.500 * sfreq)
        p300_amp = (
            float(np.mean(post_epoch[:, p300_start:p300_end]))
            if p300_end <= n_post
            else 0.0
        )

        lpp_start = round(0.400 * sfreq)
        lpp_end = round(0.800 * sfreq)
        lpp_amp = (
            float(np.mean(post_epoch[:, lpp_start:lpp_end]))
            if lpp_end <= n_post
            else 0.0
        )

        lat_start = round(0.200 * sfreq)
        lat_end = round(0.600 * sfreq)
        if lat_end <= n_post:
            mean_erp = np.mean(post_epoch[:, lat_start:lat_end], axis=0)
            peak_lat = float(np.argmax(np.abs(mean_erp)) / sfreq)
        else:
            peak_lat = 0.5

        if n_ch >= 5:
            paa = float(np.log(alpha_pows[1] + 1e-10) - np.log(alpha_pows[3] + 1e-10))
        elif n_ch >= 2:
            paa = float(np.log(alpha_pows[0] + 1e-10) - np.log(alpha_pows[1] + 1e-10))
        else:
            paa = 0.0
        epoch_var = float(np.mean(np.var(post_epoch, axis=1)))
        epoch_std = float(np.mean(np.std(post_epoch, axis=1)))
        epoch_skew = float(np.mean([_safe_skew(post_epoch[ch]) for ch in range(n_ch)]))
        epoch_kurt = float(np.mean([_safe_kurtosis(post_epoch[ch]) for ch in range(n_ch)]))

        mean_amp = float(np.mean(post_epoch))
        max_amp = float(np.max(np.abs(post_epoch)))
        zero_crossings = float(np.mean([np.sum(np.diff(np.sign(post_epoch[ch])) != 0) for ch in range(n_ch)]))

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

        first_half = post_epoch[:, :n_post // 2]
        second_half = post_epoch[:, n_post // 2:]
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

    epochs = np.load(epochs_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)
    validate_metadata_schema(meta)

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
            print(f"  [OK] X={x_s.shape}, classes={dict(zip(*np.unique(y_s, return_counts=True)))}")

        if not all_x:
            raise ValueError("No usable labeled data found across sessions")

        return np.vstack(all_x), np.concatenate(all_y), np.concatenate(groups), session_dirs

    if not data_dir:
        raise ValueError("Either data_dir or data_root must be provided")

    print(f"[LOAD] {data_dir}")
    x_s, y_s = _load_one_session(data_dir)
    print(f"  [OK] X={x_s.shape}, classes={dict(zip(*np.unique(y_s, return_counts=True)))}")
    return x_s, y_s, None, [data_dir]


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
    smote_step = [("smote", SMOTE(random_state=42))] if (HAS_IMBLEARN and use_smote) else []

    models = {
        "RandomForest": PipelineClass(
            smote_step + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(mutual_info_classif, k="all")),
                ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
            ]
        ),
        "LDA": PipelineClass(
            smote_step + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(mutual_info_classif, k="all")),
                ("clf", LinearDiscriminantAnalysis()),
            ]
        ),
        "SVM-RBF": PipelineClass(
            smote_step + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(mutual_info_classif, k="all")),
                ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True)),
            ]
        ),
        "GradBoost": PipelineClass(
            smote_step + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(mutual_info_classif, k="all")),
                ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ]
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = PipelineClass(
            smote_step + [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(mutual_info_classif, k="all")),
                ("clf", XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                )),
            ]
        )

    return models


def _get_param_distributions():
    return {
        "RandomForest": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
            "clf__max_features": ["sqrt", "log2", None],
            "selector__k": [10, 20, "all"],
        },
        "SVM-RBF": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.01, 0.1],
            "selector__k": [10, 20, "all"],
        },
        "GradBoost": {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth": [3, 5, 7],
            "clf__subsample": [0.8, 1.0],
            "selector__k": [10, 20, "all"],
        },
        "XGBoost": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "selector__k": [10, 20, "all"],
        },
        "LDA": {
            "selector__k": [10, 20, "all"],
        },
    }


def _count_param_combinations(param_dict):
    count = 1
    for v in param_dict.values():
        count *= len(v) if isinstance(v, list) else 1
    return count


def train(data_dir, model_out_path, plot=True, data_root=None, tune_hyperparams=True):
    X, y, groups, used_sessions = _load_dataset(data_dir=data_dir, data_root=data_root)

    if len(np.unique(y)) < 2:
        print("[WARN] Only one class in labels. Need more balanced data.")
        return

    cv_name, cv, cv_groups = _build_cv(y, groups=groups)
    if cv is None:
        print("[WARN] Not enough samples per class for cross-validation.")
        print("       Collect more sessions or reduce label sparsity.")
        return

    print(f"\n[DATA] Feature matrix: {X.shape}")
    print(f"[CV] Using {cv_name}")
    if groups is not None:
        print(f"[CV] Groups: {len(np.unique(groups))} session(s)")

    if HAS_IMBLEARN:
        print("[SMOTE] Enabled for class imbalance handling")
    else:
        print("[SMOTE] Not available (install imbalanced-learn for SMOTE)")

    models = _build_models()
    param_dists = _get_param_distributions()

    best_score = -np.inf
    best_name = None
    best_pipe = None
    cv_results = {}

    for name, pipe in models.items():
        print(f"\n[MODEL] {name}")

        if tune_hyperparams and name in param_dists and len(y) >= 50:
            print(f"  [TUNING] Running RandomizedSearchCV...")
            n_iter = min(20, _count_param_combinations(param_dists[name]))
            search = RandomizedSearchCV(
                pipe,
                param_dists[name],
                n_iter=n_iter,
                cv=cv,
                scoring="balanced_accuracy",
                random_state=42,
                n_jobs=-1,
                error_score="raise",
            )
            try:
                search.fit(X, y)
                scores = np.array([search.best_score_])
                tuned_pipe = search.best_estimator_
                print(f"  [TUNING] Best params: {search.best_params_}")
            except Exception as e:
                print(f"  [TUNING] Failed: {e}, falling back to defaults")
                scores = cross_val_score(pipe, X, y, cv=cv, groups=cv_groups, scoring="balanced_accuracy")
                tuned_pipe = pipe
        else:
            scores = cross_val_score(pipe, X, y, cv=cv, groups=cv_groups, scoring="balanced_accuracy")
            tuned_pipe = pipe

        cv_results[name] = scores
        mean_s = float(scores.mean())
        print(f"  {name:15s} balanced_acc = {mean_s:.3f} +- {scores.std():.3f}")

        if mean_s > best_score:
            best_score = mean_s
            best_name = name
            best_pipe = tuned_pipe

    print(f"\n[SELECT] Best model: {best_name} ({best_score:.3f})")

    if not hasattr(best_pipe, "classes_"):
        best_pipe.fit(X, y)

    os.makedirs(os.path.dirname(model_out_path) if os.path.dirname(model_out_path) else ".", exist_ok=True)
    with open(model_out_path, "wb") as f:
        pickle.dump(
            {
                "model": best_pipe,
                "model_name": best_name,
                "feature_names": _feature_names(),
                "cv_results": cv_results,
                "cv_strategy": cv_name,
                "sessions": used_sessions,
            },
            f,
        )

    print(f"[SAVED] Model -> {model_out_path}")

    y_pred = best_pipe.predict(X)
    print("\n[REPORT] Training-set sanity report")
    print(classification_report(y, y_pred, target_names=["Focused", "Zoned-out"]))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    if plot:
        out_dir = data_root if data_root else data_dir
        _plot_results(cv_results, best_pipe, best_name, out_dir)


CHANNEL_NAMES = ["T5", "P3", "Pz", "P4", "T6", "O1", "O2", "REF"]


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


def _plot_results(cv_results, best_pipe, best_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    names = list(cv_results.keys())
    means = [cv_results[n].mean() for n in names]
    stds = [cv_results[n].std() for n in names]
    ax.bar(names, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("CV Model Comparison")
    ax.set_ylim(0, 1)
    ax.legend()

    ax2 = axes[1]
    has_importance = False
    if "RandomForest" in best_name or "GradBoost" in best_name or "XGBoost" in best_name:
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            fnames = _feature_names()[: len(imp)]
            idx = np.argsort(imp)[::-1][:15]
            ax2.barh([fnames[i] for i in idx[::-1]], imp[idx[::-1]], color="teal", alpha=0.8)
            ax2.set_xlabel("Importance")
            ax2.set_title(f"Top-15 Feature Importances ({best_name})")
            has_importance = True

    if not has_importance:
        ax2.text(
            0.5,
            0.5,
            f"Feature importance\nnot available for {best_name}",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, "classifier_results.png")
    plt.savefig(plot_path, dpi=120)
    print(f"[PLOT] Saved -> {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RSVP attention classifier")
    parser.add_argument(
        "--data_dir",
        default="data/rsvp/sub-01/ses-01",
        help="One session directory containing eeg_epochs.npy and metadata.csv",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Root dataset dir to aggregate all sub-*/ses-* sessions (enables group CV)",
    )
    parser.add_argument("--out_model", default="cache/attention_model.pkl", help="Model output path")
    parser.add_argument("--no_plot", action="store_true", help="Skip saving the results plot")
    parser.add_argument("--no_tune", action="store_true", help="Skip hyperparameter tuning")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_out_path=args.out_model,
        plot=not args.no_plot,
        data_root=args.data_root,
        tune_hyperparams=not args.no_tune,
    )

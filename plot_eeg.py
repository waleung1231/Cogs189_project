"""
Quick EEG sanity-check plots.

Usage:
    python plot_eeg.py                          # defaults to sub-01/ses-01
    python plot_eeg.py --subject 2 --session 1
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CH_LABELS = ['O1', 'O2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'Fz']
SAMPLING_RATE = 250
EPOCH_PRE_S   = 0.200
EPOCH_POST_S  = 0.800

def load_data(subject, session):
    d = f'data/rsvp/sub-{subject:02d}/ses-{session:02d}'
    epochs   = np.load(os.path.join(d, 'eeg_epochs.npy'))
    metadata = pd.read_csv(os.path.join(d, 'metadata.csv'))
    raw_path = os.path.join(d, 'eeg_raw.npy')
    raw_eeg  = np.load(raw_path) if os.path.exists(raw_path) else None
    return epochs, metadata, raw_eeg


def plot_raw_snippet(raw_eeg, seconds=5):
    """Plot first `seconds` of continuous raw EEG."""
    if raw_eeg is None:
        print('No raw EEG file found (demo mode?) — skipping raw plot.')
        return
    n_samp = min(int(seconds * SAMPLING_RATE), raw_eeg.shape[1])
    t = np.arange(n_samp) / SAMPLING_RATE

    fig, axes = plt.subplots(len(CH_LABELS), 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Raw EEG — first {seconds}s', fontsize=14)
    for i, (ax, label) in enumerate(zip(axes, CH_LABELS)):
        ax.plot(t, raw_eeg[i, :n_samp], linewidth=0.5)
        ax.set_ylabel(label)
        ax.set_xlim(t[0], t[-1])
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_erps(epochs, metadata):
    """Plot average ERP for target vs non-target trials."""
    n_samples = epochs.shape[2]
    t = np.linspace(-EPOCH_PRE_S, EPOCH_POST_S, n_samples) * 1000  # ms

    target_mask = metadata['is_target'] == 1
    target_epochs     = epochs[target_mask.values]
    nontarget_epochs  = epochs[~target_mask.values]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle('Average ERP: Target vs Non-Target', fontsize=14)
    for i, (ax, label) in enumerate(zip(axes.flat, CH_LABELS)):
        if target_epochs.shape[0] > 0:
            ax.plot(t, target_epochs[:, i, :].mean(axis=0), color='red', label='Target')
        if nontarget_epochs.shape[0] > 0:
            ax.plot(t, nontarget_epochs[:, i, :].mean(axis=0), color='blue', alpha=0.7, label='Non-target')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel('Time (ms)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel('Amplitude (µV)')
    axes[1, 0].set_ylabel('Amplitude (µV)')
    plt.tight_layout()
    plt.show()


def plot_single_trials(epochs, metadata, n_trials=10):
    """Plot a few individual target epochs (stacked) per channel to check signal."""
    target_idx = metadata.index[metadata['is_target'] == 1].tolist()[:n_trials]
    if not target_idx:
        print('No target trials found — skipping single-trial plot.')
        return

    n_samples = epochs.shape[2]
    t = np.linspace(-EPOCH_PRE_S, EPOCH_POST_S, n_samples) * 1000

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    fig.suptitle(f'Single Target Trials (first {len(target_idx)})', fontsize=14)
    for i, (ax, label) in enumerate(zip(axes.flat, CH_LABELS)):
        for j, idx in enumerate(target_idx):
            ax.plot(t, epochs[idx, i, :] + j * 20, linewidth=0.6, alpha=0.8)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude (µV, stacked)')
    axes[1, 0].set_ylabel('Amplitude (µV, stacked)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot EEG sanity checks')
    parser.add_argument('--subject', type=int, default=1)
    parser.add_argument('--session', type=int, default=1)
    args = parser.parse_args()

    epochs, metadata, raw_eeg = load_data(args.subject, args.session)
    print(f'Loaded epochs: {epochs.shape}  ({metadata.shape[0]} trials)')

    plot_raw_snippet(raw_eeg)
    plot_erps(epochs, metadata)
    plot_single_trials(epochs, metadata)

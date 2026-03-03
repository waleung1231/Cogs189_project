#!/usr/bin/env python3
"""
EEG Data Viewer for RSVP Attention Experiment
==============================================
Interactive visualization tool for viewing EEG recordings.

Usage:
    python view_eeg.py                           # Default: sub-01/ses-01
    python view_eeg.py --subject 1 --session 1   # Specify subject/session
    python view_eeg.py --data_dir path/to/data   # Direct path to session folder
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, Slider, CheckButtons
from scipy import signal as spsig

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SAMPLING_RATE = 250  # Hz
CH_LABELS = ['O1', 'O2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'Fz']
EPOCH_PRE_S = 0.200
EPOCH_POST_S = 0.800

# Color scheme for channels
CHANNEL_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
]


class EEGViewer:
    """Interactive EEG data viewer with multiple visualization modes."""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_data()
        self.setup_figure()
        
    def load_data(self):
        """Load EEG data and metadata from the session directory."""
        print(f"Loading data from: {self.data_dir}")
        
        # Load epoched EEG
        epochs_path = os.path.join(self.data_dir, 'eeg_epochs.npy')
        if os.path.exists(epochs_path):
            self.epochs = np.load(epochs_path, allow_pickle=True)
            print(f"  Epochs: {self.epochs.shape} (trials × channels × samples)")
        else:
            self.epochs = None
            print("  [WARN] No eeg_epochs.npy found")
        
        # Load raw EEG
        raw_path = os.path.join(self.data_dir, 'eeg_raw.npy')
        if os.path.exists(raw_path):
            self.raw_eeg = np.load(raw_path, allow_pickle=True)
            print(f"  Raw EEG: {self.raw_eeg.shape} (channels × samples)")
            self.duration_s = self.raw_eeg.shape[1] / SAMPLING_RATE
            print(f"  Duration: {self.duration_s:.1f} seconds")
        else:
            self.raw_eeg = None
            print("  [WARN] No eeg_raw.npy found")
        
        # Load metadata
        meta_path = os.path.join(self.data_dir, 'metadata.csv')
        if os.path.exists(meta_path):
            self.metadata = pd.read_csv(meta_path)
            print(f"  Metadata: {len(self.metadata)} trials")
            n_targets = self.metadata['is_target'].sum()
            n_hits = (self.metadata['response'] == 'hit').sum()
            print(f"  Targets: {n_targets}, Hits: {n_hits}")
        else:
            self.metadata = None
            print("  [WARN] No metadata.csv found")
        
        # Load events
        events_path = os.path.join(self.data_dir, 'events.npy')
        if os.path.exists(events_path):
            self.events = np.load(events_path, allow_pickle=True)
            print(f"  Events: {len(self.events)} onset markers")
        else:
            self.events = None
        
        # Validate we have something to show
        if self.epochs is None and self.raw_eeg is None:
            raise FileNotFoundError(f"No EEG data found in {self.data_dir}")
        
        # Current view state
        self.current_epoch = 0
        self.current_time = 0.0
        self.window_size = 5.0  # seconds for raw view
        self.view_mode = 'epochs'  # 'epochs' or 'raw'
        self.show_filtered = True
        self.selected_channels = list(range(len(CH_LABELS)))
        
    def setup_figure(self):
        """Create the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle('EEG Data Viewer', fontsize=14, fontweight='bold')
        
        # Main EEG plot area
        self.ax_eeg = self.fig.add_axes([0.08, 0.35, 0.85, 0.55])
        
        # Info text area
        self.ax_info = self.fig.add_axes([0.08, 0.92, 0.85, 0.06])
        self.ax_info.axis('off')
        
        # Navigation buttons
        btn_width = 0.08
        btn_height = 0.04
        btn_y = 0.25
        
        self.ax_prev = self.fig.add_axes([0.08, btn_y, btn_width, btn_height])
        self.btn_prev = Button(self.ax_prev, '◀ Prev')
        self.btn_prev.on_clicked(self.prev_view)
        
        self.ax_next = self.fig.add_axes([0.17, btn_y, btn_width, btn_height])
        self.btn_next = Button(self.ax_next, 'Next ▶')
        self.btn_next.on_clicked(self.next_view)
        
        self.ax_mode = self.fig.add_axes([0.30, btn_y, 0.12, btn_height])
        self.btn_mode = Button(self.ax_mode, 'Switch to Raw')
        self.btn_mode.on_clicked(self.toggle_mode)
        
        self.ax_filter = self.fig.add_axes([0.43, btn_y, 0.10, btn_height])
        self.btn_filter = Button(self.ax_filter, 'Filter: ON')
        self.btn_filter.on_clicked(self.toggle_filter)
        
        # Epoch/time slider
        self.ax_slider = self.fig.add_axes([0.08, 0.15, 0.75, 0.04])
        if self.epochs is not None:
            self.slider = Slider(
                self.ax_slider, 'Epoch', 0, len(self.epochs) - 1,
                valinit=0, valstep=1
            )
        else:
            self.slider = Slider(
                self.ax_slider, 'Time (s)', 0, self.duration_s - self.window_size,
                valinit=0
            )
        self.slider.on_changed(self.on_slider_change)
        
        # Window size slider (for raw view)
        self.ax_window = self.fig.add_axes([0.08, 0.08, 0.75, 0.04])
        self.slider_window = Slider(
            self.ax_window, 'Window (s)', 1, 30,
            valinit=self.window_size
        )
        self.slider_window.on_changed(self.on_window_change)
        
        # Channel checkboxes
        self.ax_channels = self.fig.add_axes([0.86, 0.08, 0.12, 0.20])
        self.check_channels = CheckButtons(
            self.ax_channels, CH_LABELS,
            [True] * len(CH_LABELS)
        )
        self.check_channels.on_clicked(self.on_channel_toggle)
        
        # Keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial plot
        self.update_plot()
        
    def bandpass_filter(self, data, low=1.0, high=40.0):
        """Apply bandpass filter to EEG data."""
        nyq = SAMPLING_RATE / 2
        low_norm = low / nyq
        high_norm = high / nyq
        b, a = spsig.butter(4, [low_norm, high_norm], btype='band')
        
        if data.ndim == 1:
            return spsig.filtfilt(b, a, data)
        else:
            return np.array([spsig.filtfilt(b, a, ch) for ch in data])
    
    def update_plot(self):
        """Update the EEG plot based on current view state."""
        self.ax_eeg.clear()
        
        if self.view_mode == 'epochs' and self.epochs is not None:
            self.plot_epoch()
        elif self.raw_eeg is not None:
            self.plot_raw()
        else:
            self.ax_eeg.text(0.5, 0.5, 'No data available for this view',
                            ha='center', va='center', transform=self.ax_eeg.transAxes)
        
        self.fig.canvas.draw_idle()
    
    def plot_epoch(self):
        """Plot a single epoch with channel stacking."""
        epoch = self.epochs[self.current_epoch]
        n_ch, n_samp = epoch.shape
        
        # Time axis
        t = np.linspace(-EPOCH_PRE_S, EPOCH_POST_S, n_samp)
        
        # Apply filter if enabled
        if self.show_filtered:
            epoch = self.bandpass_filter(epoch)
        
        # Calculate spacing between channels
        spacing = np.nanmax(np.abs(epoch)) * 2.5
        if spacing == 0 or np.isnan(spacing):
            spacing = 50  # default µV spacing
        
        # Plot each selected channel
        for i, ch_idx in enumerate(self.selected_channels):
            if ch_idx < n_ch:
                offset = (len(self.selected_channels) - 1 - i) * spacing
                color = CHANNEL_COLORS[ch_idx % len(CHANNEL_COLORS)]
                self.ax_eeg.plot(t, epoch[ch_idx] + offset, color=color, 
                                linewidth=0.8, label=CH_LABELS[ch_idx])
        
        # Mark stimulus onset
        self.ax_eeg.axvline(0, color='red', linestyle='--', linewidth=1.5, 
                           label='Stimulus onset')
        
        # Shade pre-stimulus period
        self.ax_eeg.axvspan(-EPOCH_PRE_S, 0, alpha=0.1, color='gray')
        
        # Labels and formatting
        self.ax_eeg.set_xlabel('Time (s)', fontsize=11)
        self.ax_eeg.set_ylabel('Amplitude (µV)', fontsize=11)
        self.ax_eeg.set_xlim(-EPOCH_PRE_S, EPOCH_POST_S)
        self.ax_eeg.grid(True, alpha=0.3)
        
        # Add channel labels on y-axis
        yticks = [(len(self.selected_channels) - 1 - i) * spacing 
                  for i in range(len(self.selected_channels))]
        yticklabels = [CH_LABELS[ch] for ch in self.selected_channels]
        self.ax_eeg.set_yticks(yticks)
        self.ax_eeg.set_yticklabels(yticklabels)
        
        # Update info text
        self.update_info_text()
    
    def plot_raw(self):
        """Plot continuous raw EEG data."""
        start_samp = int(self.current_time * SAMPLING_RATE)
        end_samp = int((self.current_time + self.window_size) * SAMPLING_RATE)
        end_samp = min(end_samp, self.raw_eeg.shape[1])
        
        data = self.raw_eeg[:, start_samp:end_samp]
        n_ch, n_samp = data.shape
        
        # Time axis
        t = np.linspace(self.current_time, self.current_time + n_samp / SAMPLING_RATE, n_samp)
        
        # Apply filter if enabled
        if self.show_filtered and n_samp > 50:
            data = self.bandpass_filter(data)
        
        # Calculate spacing
        spacing = np.nanmax(np.abs(data)) * 2.5
        if spacing == 0 or np.isnan(spacing):
            spacing = 50
        
        # Plot each selected channel
        for i, ch_idx in enumerate(self.selected_channels):
            if ch_idx < n_ch:
                offset = (len(self.selected_channels) - 1 - i) * spacing
                color = CHANNEL_COLORS[ch_idx % len(CHANNEL_COLORS)]
                self.ax_eeg.plot(t, data[ch_idx] + offset, color=color, 
                                linewidth=0.5, label=CH_LABELS[ch_idx])
        
        # Mark event onsets if available
        if self.events is not None and self.metadata is not None:
            for idx, onset_samp in enumerate(self.events):
                if start_samp <= onset_samp < end_samp:
                    onset_time = onset_samp / SAMPLING_RATE
                    is_target = self.metadata.iloc[idx]['is_target'] if idx < len(self.metadata) else 0
                    color = 'red' if is_target else 'blue'
                    alpha = 0.8 if is_target else 0.3
                    self.ax_eeg.axvline(onset_time, color=color, linestyle='--', 
                                       linewidth=1, alpha=alpha)
        
        # Labels and formatting
        self.ax_eeg.set_xlabel('Time (s)', fontsize=11)
        self.ax_eeg.set_ylabel('Amplitude (µV)', fontsize=11)
        self.ax_eeg.set_xlim(self.current_time, self.current_time + self.window_size)
        self.ax_eeg.grid(True, alpha=0.3)
        
        # Channel labels
        yticks = [(len(self.selected_channels) - 1 - i) * spacing 
                  for i in range(len(self.selected_channels))]
        yticklabels = [CH_LABELS[ch] for ch in self.selected_channels]
        self.ax_eeg.set_yticks(yticks)
        self.ax_eeg.set_yticklabels(yticklabels)
        
        # Update info
        self.update_info_text()
    
    def update_info_text(self):
        """Update the information text display."""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        if self.view_mode == 'epochs' and self.metadata is not None:
            row = self.metadata.iloc[self.current_epoch]
            letter = row['letter']
            is_target = '🎯 TARGET' if row['is_target'] else ''
            response = row['response']
            rt = row['rt']
            rt_str = f"{rt*1000:.0f} ms" if not np.isnan(rt) else "N/A"
            
            info = (f"Epoch {self.current_epoch + 1}/{len(self.epochs)}  |  "
                   f"Letter: {letter} {is_target}  |  "
                   f"Response: {response}  |  RT: {rt_str}")
        elif self.view_mode == 'raw':
            info = (f"Raw EEG  |  Time: {self.current_time:.1f} - "
                   f"{self.current_time + self.window_size:.1f} s  |  "
                   f"Total: {self.duration_s:.1f} s")
        else:
            info = "No metadata available"
        
        filter_status = "Filtered (1-40 Hz)" if self.show_filtered else "Unfiltered"
        info += f"  |  {filter_status}"
        
        self.ax_info.text(0.5, 0.5, info, ha='center', va='center',
                         fontsize=11, transform=self.ax_info.transAxes)
    
    def prev_view(self, event=None):
        """Go to previous epoch/time window."""
        if self.view_mode == 'epochs':
            self.current_epoch = max(0, self.current_epoch - 1)
            self.slider.set_val(self.current_epoch)
        else:
            self.current_time = max(0, self.current_time - self.window_size / 2)
            self.slider.set_val(self.current_time)
        self.update_plot()
    
    def next_view(self, event=None):
        """Go to next epoch/time window."""
        if self.view_mode == 'epochs':
            self.current_epoch = min(len(self.epochs) - 1, self.current_epoch + 1)
            self.slider.set_val(self.current_epoch)
        else:
            max_time = self.duration_s - self.window_size
            self.current_time = min(max_time, self.current_time + self.window_size / 2)
            self.slider.set_val(self.current_time)
        self.update_plot()
    
    def toggle_mode(self, event=None):
        """Switch between epoch and raw view modes."""
        if self.view_mode == 'epochs' and self.raw_eeg is not None:
            self.view_mode = 'raw'
            self.btn_mode.label.set_text('Switch to Epochs')
            # Update slider for raw mode
            self.slider.valmin = 0
            self.slider.valmax = max(0.1, self.duration_s - self.window_size)
            self.slider.set_val(0)
            self.slider.label.set_text('Time (s)')
        elif self.view_mode == 'raw' and self.epochs is not None:
            self.view_mode = 'epochs'
            self.btn_mode.label.set_text('Switch to Raw')
            # Update slider for epoch mode
            self.slider.valmin = 0
            self.slider.valmax = len(self.epochs) - 1
            self.slider.set_val(self.current_epoch)
            self.slider.label.set_text('Epoch')
        self.update_plot()
    
    def toggle_filter(self, event=None):
        """Toggle bandpass filter on/off."""
        self.show_filtered = not self.show_filtered
        self.btn_filter.label.set_text(f"Filter: {'ON' if self.show_filtered else 'OFF'}")
        self.update_plot()
    
    def on_slider_change(self, val):
        """Handle slider value change."""
        if self.view_mode == 'epochs':
            self.current_epoch = int(val)
        else:
            self.current_time = val
        self.update_plot()
    
    def on_window_change(self, val):
        """Handle window size slider change."""
        self.window_size = val
        if self.view_mode == 'raw':
            self.slider.valmax = max(0.1, self.duration_s - self.window_size)
            self.update_plot()
    
    def on_channel_toggle(self, label):
        """Handle channel checkbox toggle."""
        idx = CH_LABELS.index(label)
        if idx in self.selected_channels:
            if len(self.selected_channels) > 1:  # Keep at least one channel
                self.selected_channels.remove(idx)
        else:
            self.selected_channels.append(idx)
            self.selected_channels.sort()
        self.update_plot()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'left':
            self.prev_view()
        elif event.key == 'right':
            self.next_view()
        elif event.key == 'f':
            self.toggle_filter()
        elif event.key == 'm':
            self.toggle_mode()
        elif event.key == 'q':
            plt.close(self.fig)
    
    def show(self):
        """Display the viewer."""
        print("\nKeyboard shortcuts:")
        print("  ← / →  : Previous / Next")
        print("  f      : Toggle filter")
        print("  m      : Toggle view mode (epochs/raw)")
        print("  q      : Quit")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='EEG Data Viewer for RSVP Experiment')
    parser.add_argument('--subject', '-s', type=int, default=1,
                       help='Subject number (default: 1)')
    parser.add_argument('--session', '-e', type=int, default=1,
                       help='Session number (default: 1)')
    parser.add_argument('--data_dir', '-d', type=str, default=None,
                       help='Direct path to session data directory')
    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Look for data in standard locations
        base_paths = [
            f'data/rsvp/sub-{args.subject:02d}/ses-{args.session:02d}',
            f'data/sub-{args.subject:02d}/ses-{args.session:02d}',
        ]
        data_dir = None
        for path in base_paths:
            if os.path.exists(path):
                data_dir = path
                break
        
        if data_dir is None:
            print(f"Error: Could not find data directory for subject {args.subject}, session {args.session}")
            print("Tried paths:")
            for path in base_paths:
                print(f"  {path}")
            sys.exit(1)
    
    # Create and show viewer
    viewer = EEGViewer(data_dir)
    viewer.show()


if __name__ == '__main__':
    main()

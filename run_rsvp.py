# ============================================================
#  RSVP Attention / Mind-Wandering EEG Experiment
#  Hardware: OpenBCI Cyton 8-channel + PsychoPy
# ============================================================
#
#  TASK OVERVIEW
#  -------------
#  A stream of letters (A–Z) flashes one at a time in the
#  centre of the screen at 5 Hz.  ~20% of letters are the
#  pre-assigned target.  Participant presses SPACEBAR for the
#  target.  EEG is epoched around every letter onset
#  (-200 ms … +800 ms) and saved alongside behavioural data
#  so that a classifier can later distinguish focused from
#  zoned-out trials.
#
#  TIMING MODEL (per letter)
#  -------------------------
#  flash_duration  = 300 ms   (letter visible)
#  ISI             = 100–150 ms + ±30 ms jitter (blank screen)
#  Total per stim  ≈ 200–280 ms  →  effective rate ≈ 5 Hz
#  Session         ≈ 3 min  →  ~900 letters total
#  Target rate     ≈ 20%    →  ~180 target appearances
#  Repetitions per character in practice: 10-15
#
#  DATA SAVED (per letter event)
#  ------------------------------
#  - EEG epoch   : (8 ch × epoch_samples) numpy array
#  - letter      : which letter was shown
#  - is_target   : bool
#  - response    : 'hit' | 'miss' | 'false_alarm' | 'cr'
#  - rt          : reaction time in seconds (NaN if no press)
#  - block_time  : time since session start at letter onset
#
#  FILE LAYOUT
#  -----------
#  data/rsvp/sub-<XX>/ses-<YY>/
#      eeg_raw.npy          – continuous raw EEG  (8 × N)
#      eeg_epochs.npy       – epoched EEG         (trials × 8 × epoch_samples)
#      metadata.csv         – per-letter behavioural data
#      events.npy           – sample indices of each letter onset in raw EEG
#
# ============================================================

import os, sys, glob, time, random, pickle
import numpy as np
from psychopy import visual, core, event
from psychopy.hardware import keyboard as kb

# ──────────────────────────────────────────────
#  CONFIG  (edit these before each session)
# ──────────────────────────────────────────────
SUBJECT         = 1
SESSION         = 1
TARGET_LETTER   = 'X'          # same for whole session + practice
CYTON_IN        = False        # False = demo mode (no EEG hardware)

WIDTH, HEIGHT   = 1536, 960
REFRESH_RATE    = 60.0          # Hz – adjust to your monitor

FLASH_DURATION_S    = 0.300     # 300 ms
ISI_BASE_S          = 0.125     # midpoint of 100–150 ms range
ISI_JITTER_S        = 0.030     # ±30 ms uniform jitter
SESSION_DURATION_S  = 180       # 3 minutes
TARGET_PROBABILITY  = 0.20      # ~1 in 5 letters is target

EPOCH_PRE_S     = 0.200         # 200 ms before letter onset
EPOCH_POST_S    = 0.800         # 800 ms after letter onset
SAMPLING_RATE   = 250           # Cyton default

RESPONSE_WINDOW_S = 0.600       # max RT counted as a hit (after onset)

# Channel montage (OpenBCI Cyton 8-ch)
# REF (SRB) at Cz, GND/BIAS near Fz
#   Ch1=O1, Ch2=O2, Ch3=T5, Ch4=P3, Ch5=Pz, Ch6=P4, Ch7=T6, Ch8=Fz
N_EEG_CHANNELS  = 8
CH_LABELS       = ['O1', 'O2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'Fz']

SAVE_DIR = f'data/rsvp/sub-{SUBJECT:02d}/ses-{SESSION:02d}/'

# Practice settings
PRACTICE_N_TARGETS  = 12        # ~10-15 target appearances in practice
PRACTICE_PROPORTION = 0.20

# ──────────────────────────────────────────────
#  DERIVED CONSTANTS
# ──────────────────────────────────────────────
FLASH_FRAMES    = round(FLASH_DURATION_S * REFRESH_RATE)   # frames letter is ON
EPOCH_PRE_SAMP  = round(EPOCH_PRE_S  * SAMPLING_RATE)
EPOCH_POST_SAMP = round(EPOCH_POST_S * SAMPLING_RATE)
EPOCH_SAMPLES   = EPOCH_PRE_SAMP + EPOCH_POST_SAMP

LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
NON_TARGETS = [l for l in LETTERS if l != TARGET_LETTER]

# Random fonts and colors for each flashing letter
FONT_POOL = [
    'Arial', 'Courier New', 'Georgia', 'Times New Roman',
    'Verdana', 'Trebuchet MS', 'Palatino', 'Futura',
    'Helvetica', 'Comic Sans MS', 'Impact', 'Lucida Console',
]
COLOR_POOL = [
    'white', 'red', 'cyan', 'yellow', 'lime',
    'orange', 'magenta', 'dodgerblue', 'springgreen',
    'coral', 'violet', 'gold',
]

# ──────────────────────────────────────────────
#  HELPER: build a letter sequence
# ──────────────────────────────────────────────
def make_sequence(n_total, target=TARGET_LETTER, p_target=TARGET_PROBABILITY, seed=None):
    """Return a list of letters with ~p_target proportion being the target.
    Ensures no two consecutive identical letters."""
    rng = np.random.default_rng(seed)
    n_targets = round(n_total * p_target)
    n_non     = n_total - n_targets
    seq = [target] * n_targets + random.sample(NON_TARGETS * (n_non // len(NON_TARGETS) + 1), n_non)
    rng.shuffle(seq)
    # break up consecutive duplicates with a simple swap
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            for j in range(i+1, len(seq)):
                if seq[j] != seq[i-1]:
                    seq[i], seq[j] = seq[j], seq[i]
                    break
    return seq

def make_practice_sequence(n_targets=PRACTICE_N_TARGETS, p_target=PRACTICE_PROPORTION, seed=42):
    n_total = round(n_targets / p_target)
    return make_sequence(n_total, seed=seed)

# ──────────────────────────────────────────────
#  EEG / CYTON SETUP
# ──────────────────────────────────────────────
if CYTON_IN:
    import serial
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from threading import Thread, Event
    from queue import Queue

    CYTON_BOARD_ID  = 0
    BAUD_RATE       = 115200
    ANALOGUE_MODE   = '/2'

    def find_openbci_port():
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i+1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Unsupported OS for port detection')
        for port in ports:
            try:
                s = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    while '$$$' not in line:
                        line += s.read().decode('utf-8', errors='replace')
                    if 'OpenBCI' in line:
                        s.close()
                        return port
                s.close()
            except (OSError, serial.SerialException):
                pass
        raise OSError('Cannot find OpenBCI Cyton port.')

    params = BrainFlowInputParams()
    params.serial_port = find_openbci_port()
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    board.config_board('/0')
    board.config_board('//')
    board.config_board(ANALOGUE_MODE)
    board.start_stream(45000)

    # Background thread fills a queue with chunks of EEG
    stop_event  = Event()
    eeg_queue   = Queue()

    def _collect(queue):
        while not stop_event.is_set():
            chunk = board.get_board_data()
            ts  = chunk[BoardShim.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg = chunk[BoardShim.get_eeg_channels(CYTON_BOARD_ID)]  # shape (8, N)
            if ts.size > 0:
                queue.put((eeg, ts))
            time.sleep(0.02)   # poll at ~50 Hz

    eeg_thread = Thread(target=_collect, args=(eeg_queue,), daemon=True)
    eeg_thread.start()

    # Accumulators for continuous raw data
    raw_eeg       = np.zeros((N_EEG_CHANNELS, 0), dtype=np.float32)
    raw_ts        = np.zeros((0,),   dtype=np.float64)

    def flush_queue():
        """Drain the EEG queue into the accumulators."""
        global raw_eeg, raw_ts
        while not eeg_queue.empty():
            eeg_chunk, ts_chunk = eeg_queue.get()
            raw_eeg = np.concatenate([raw_eeg, eeg_chunk], axis=1)
            raw_ts  = np.concatenate([raw_ts,  ts_chunk])

    def current_sample_index():
        """Returns the number of EEG samples collected so far."""
        flush_queue()
        return raw_eeg.shape[1]

else:
    # ── DEMO / no-hardware mode ──────────────────
    raw_eeg = np.zeros((N_EEG_CHANNELS, 0), dtype=np.float32)
    raw_ts  = np.zeros((0,),   dtype=np.float64)

    def flush_queue():
        pass

    def current_sample_index():
        # Simulate samples based on wall time
        return int(core.getTime() * SAMPLING_RATE)


# ──────────────────────────────────────────────
#  PSYCHOPY WINDOW & STIMULI
# ──────────────────────────────────────────────
win = visual.Window(
    size=[WIDTH, HEIGHT],
    fullscr=True,
    allowGUI=False,
    checkTiming=True,
    useRetina=False,
    color='black',
    units='norm',
)

letter_stim = visual.TextStim(
    win=win,
    text='',
    height=0.25,
    color='white',
    font='Helvetica',
    bold=True,
)

fixation = visual.TextStim(
    win=win,
    text='+',
    height=0.10,
    color='gray',
    font='Helvetica',
)

instruction_stim = visual.TextStim(
    win=win,
    text='',
    height=0.06,
    color='white',
    wrapWidth=1.8,
    alignText='center',
)

feedback_stim = visual.TextStim(
    win=win,
    text='',
    height=0.07,
    color='lime',
    pos=(0, -0.4),
)

# Small photosensor dot (bottom-right) – turns white at letter onset
# Use this with a physical photodiode for precise EEG sync if available
photosensor = visual.Rect(
    win=win, units='norm',
    width=0.07, height=0.07 * (WIDTH/HEIGHT),
    fillColor='black', lineWidth=0,
    pos=(1 - 0.04, -1 + 0.04),
)

keyboard = kb.Keyboard()


# ──────────────────────────────────────────────
#  INSTRUCTION SCREENS
# ──────────────────────────────────────────────
def show_instructions(text, wait_key='space'):
    instruction_stim.text = text
    instruction_stim.draw()
    win.flip()
    keys = event.waitKeys(keyList=[wait_key, 'escape'])
    if keys and 'escape' in keys:
        _abort()


# ──────────────────────────────────────────────
#  CORE TRIAL RUNNER
#  Returns a list of dicts, one per letter shown
# ──────────────────────────────────────────────
def run_stream(sequence, is_practice=False):
    """
    Flash each letter in `sequence`.
    Returns list of event dicts with EEG onset sample indices.
    """
    events = []          # one dict per letter
    session_clock = core.Clock()

    # Pre-generate ISIs for each letter (100–150 ms ± 30 ms jitter, clipped >0)
    isi_list = np.clip(
        np.random.uniform(ISI_BASE_S - 0.025, ISI_BASE_S + 0.025, len(sequence))
        + np.random.uniform(-ISI_JITTER_S, ISI_JITTER_S, len(sequence)),
        0.050, 0.250
    )

    keyboard.clock.reset()
    keyboard.clearEvents()

    for i_letter, letter in enumerate(sequence):
        # ── ISI: blank + fixation cross ──────────────
        isi = isi_list[i_letter]
        fixation.draw()
        photosensor.fillColor = 'black'
        photosensor.draw()
        win.flip()
        core.wait(isi, hogCPUperiod=isi)   # tight wait for accurate timing

        # ── Flush pending keypresses from ISI ────────
        # (will categorise as false alarms in analysis)
        isi_keys = keyboard.getKeys(keyList=['space'], clear=True)

        # ── Letter onset ─────────────────────────────
        is_target  = (letter == TARGET_LETTER)
        onset_wall = session_clock.getTime()
        onset_samp = current_sample_index()   # EEG sample at onset

        letter_stim.text  = letter
        letter_stim.font  = random.choice(FONT_POOL)
        letter_stim.color = random.choice(COLOR_POOL)
        letter_stim.draw()
        photosensor.fillColor = 'white'
        photosensor.draw()
        win.flip()

        # Flash duration: show letter for FLASH_FRAMES frames
        # then blank.  Also collect keypresses within response window.
        response_rt     = np.nan
        response_given  = False
        response_key    = None

        flash_clock = core.Clock()
        while flash_clock.getTime() < FLASH_DURATION_S:
            keys = keyboard.getKeys(keyList=['space', 'escape'], clear=False)
            for k in keys:
                if k.name == 'escape':
                    _abort()
                if k.name == 'space' and not response_given:
                    response_given = True
                    response_rt    = k.rt   # time from keyboard.clock reset (session start)
                    response_key   = k.name

        # ── Letter OFF ────────────────────────────────
        photosensor.fillColor = 'black'
        photosensor.draw()
        win.flip()

        # Continue collecting keypresses during the remainder of the response window
        remaining = RESPONSE_WINDOW_S - FLASH_DURATION_S
        wait_clock = core.Clock()
        while wait_clock.getTime() < remaining:
            keys = keyboard.getKeys(keyList=['space', 'escape'], clear=False)
            for k in keys:
                if k.name == 'escape':
                    _abort()
                if k.name == 'space' and not response_given:
                    response_given = True
                    response_rt    = k.rt
                    response_key   = k.name

        # ── Classify response ─────────────────────────
        if is_target and response_given:
            response_class = 'hit'
            rt = response_rt - onset_wall   # RT relative to onset
        elif is_target and not response_given:
            response_class = 'miss'
            rt = np.nan
        elif not is_target and response_given:
            response_class = 'false_alarm'
            rt = response_rt - onset_wall
        else:
            response_class = 'correct_rejection'
            rt = np.nan

        # Practice live feedback
        if is_practice:
            if response_class == 'hit':
                feedback_stim.color = 'lime'
                feedback_stim.text  = '✓'
            elif response_class == 'miss':
                feedback_stim.color = 'red'
                feedback_stim.text  = 'MISSED!'
            elif response_class == 'false_alarm':
                feedback_stim.color = 'orange'
                feedback_stim.text  = 'Wrong!'
            else:
                feedback_stim.text = ''
            if response_class != 'correct_rejection':
                feedback_stim.draw()
                win.flip()
                core.wait(0.25)

        ev = dict(
            index        = i_letter,
            letter       = letter,
            is_target    = int(is_target),
            response     = response_class,
            rt           = rt,
            onset_wall   = onset_wall,
            onset_samp   = onset_samp,    # index into raw_eeg for epoching
        )
        events.append(ev)

        # Abort if user pressed escape
        if event.getKeys(['escape']):
            _abort()

        # Check for session time limit (real runs only)
        if not is_practice and onset_wall >= SESSION_DURATION_S:
            break

    return events


def _abort():
    """Clean shutdown on escape."""
    if CYTON_IN:
        stop_event.set()
        board.stop_stream()
        board.release_session()
    win.close()
    core.quit()


# ──────────────────────────────────────────────
#  EPOCH EEG AROUND EVENTS
# ──────────────────────────────────────────────
def epoch_eeg(events):
    """
    Cut epochs from raw_eeg using onset_samp stored in each event.
    Returns array of shape (n_events, N_EEG_CHANNELS, EPOCH_SAMPLES).
    Events where there isn't enough data are filled with NaN.
    """
    import mne
    if not CYTON_IN:
        # Return random noise in demo mode
        return np.random.randn(len(events), N_EEG_CHANNELS, EPOCH_SAMPLES).astype(np.float32)

    flush_queue()
    # Bandpass filter the whole continuous recording once
    filtered = mne.filter.filter_data(
        raw_eeg.astype(np.float64),
        sfreq=SAMPLING_RATE, l_freq=1.0, h_freq=40.0, verbose=False
    )

    epochs = np.full((len(events), N_EEG_CHANNELS, EPOCH_SAMPLES), np.nan, dtype=np.float32)
    for i, ev in enumerate(events):
        start = ev['onset_samp'] - EPOCH_PRE_SAMP
        end   = start + EPOCH_SAMPLES
        if start >= 0 and end <= filtered.shape[1]:
            epoch = filtered[:, start:end]
            # Baseline correction: subtract mean of pre-stimulus window
            baseline_mean = epoch[:, :EPOCH_PRE_SAMP].mean(axis=1, keepdims=True)
            epoch = epoch - baseline_mean
            epochs[i] = epoch
    return epochs


# ──────────────────────────────────────────────
#  SAVE DATA
# ──────────────────────────────────────────────
def save_data(events, epochs):
    import csv
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Raw EEG
    if CYTON_IN:
        flush_queue()
        np.save(os.path.join(SAVE_DIR, 'eeg_raw.npy'), raw_eeg)

    # Epochs
    np.save(os.path.join(SAVE_DIR, 'eeg_epochs.npy'), epochs)

    # Event sample indices (for re-epoching later)
    onset_samps = np.array([e['onset_samp'] for e in events])
    np.save(os.path.join(SAVE_DIR, 'events.npy'), onset_samps)

    # Behavioural metadata as CSV
    csv_path = os.path.join(SAVE_DIR, 'metadata.csv')
    fieldnames = ['index','letter','is_target','response','rt','onset_wall','onset_samp']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            writer.writerow({k: ev[k] for k in fieldnames})

    print(f'\n[SAVED] {len(events)} events → {SAVE_DIR}')
    print(f'  eeg_raw.npy      : {raw_eeg.shape}')
    print(f'  eeg_epochs.npy   : {epochs.shape}')
    print(f'  metadata.csv     : {csv_path}')


# ──────────────────────────────────────────────
#  PERFORMANCE SUMMARY
# ──────────────────────────────────────────────
def summarise(events, label='Session'):
    evs = [e for e in events if e['is_target']]
    hits  = sum(1 for e in evs if e['response'] == 'hit')
    miss  = sum(1 for e in evs if e['response'] == 'miss')
    fa    = sum(1 for e in events if e['response'] == 'false_alarm')
    rts   = [e['rt'] for e in evs if e['response'] == 'hit' and not np.isnan(e['rt'])]
    hit_rate = hits / len(evs) if evs else 0
    mean_rt  = np.mean(rts) * 1000 if rts else np.nan
    print(f'\n── {label} Summary ──────────────────')
    print(f'  Targets shown : {len(evs)}')
    print(f'  Hits          : {hits}  ({hit_rate*100:.1f}%)')
    print(f'  Misses        : {miss}')
    print(f'  False alarms  : {fa}')
    print(f'  Mean RT       : {mean_rt:.0f} ms')
    return hit_rate, mean_rt


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

# ── Welcome screen ───────────────────────────
show_instructions(
    f"RSVP Attention Task\n\n"
    f"Letters will flash rapidly in the centre of the screen.\n\n"
    f"Your target letter is:  {TARGET_LETTER}\n\n"
    f"Press SPACEBAR as quickly as possible every time you see  {TARGET_LETTER}.\n\n"
    f"Do NOT press spacebar for any other letter.\n\n"
    f"Press SPACE to begin the practice.",
    wait_key='space'
)

# ── PRACTICE ─────────────────────────────────
show_instructions(
    f"PRACTICE\n\n"
    f"This is a short practice run. You will see live feedback.\n"
    f"Green ✓ = correct,  Red MISSED = you missed the target,  Orange Wrong! = false alarm\n\n"
    f"Target:  {TARGET_LETTER}\n\n"
    f"Press SPACE to start practice.",
    wait_key='space'
)

practice_seq   = make_practice_sequence(seed=99)
practice_events = run_stream(practice_seq, is_practice=True)
p_hit_rate, p_rt = summarise(practice_events, label='Practice')

show_instructions(
    f"Practice complete!\n\n"
    f"Hit rate : {p_hit_rate*100:.0f}%\n"
    f"Mean RT  : {p_rt:.0f} ms\n\n"
    f"The real session is about to begin ({SESSION_DURATION_S//60} minutes).\n"
    f"There will be no feedback during the real task.\n\n"
    f"Stay as still as possible to keep the EEG signal clean.\n\n"
    f"Press SPACE when you're ready.",
    wait_key='space'
)

# ── Countdown ────────────────────────────────
for n in [3, 2, 1]:
    instruction_stim.text = str(n)
    instruction_stim.draw()
    win.flip()
    core.wait(1.0)

# ── REAL SESSION ─────────────────────────────
# Estimate total letters needed: SESSION_DURATION_S / avg_letter_period
avg_letter_period = FLASH_DURATION_S + ISI_BASE_S   # ~225 ms
n_letters_needed  = int(SESSION_DURATION_S / avg_letter_period) + 100  # buffer
session_seq    = make_sequence(n_letters_needed, seed=SESSION)
session_events = run_stream(session_seq, is_practice=False)

# ── Epoch and save ────────────────────────────
instruction_stim.text = 'Saving data…'
instruction_stim.draw()
win.flip()

epochs = epoch_eeg(session_events)
save_data(session_events, epochs)
summarise(session_events, label='Session')

# ── Goodbye ───────────────────────────────────
hit_rate, mean_rt = summarise(session_events)
show_instructions(
    f"Session complete — thank you!\n\n"
    f"Hit rate : {hit_rate*100:.0f}%\n"
    f"Mean RT  : {mean_rt:.0f} ms\n\n"
    f"Press SPACE to exit.",
    wait_key='space'
)

# ── Teardown ──────────────────────────────────
if CYTON_IN:
    stop_event.set()
    board.stop_stream()
    board.release_session()

win.close()
core.quit()

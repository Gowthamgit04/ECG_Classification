#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import wfdb
from scipy import signal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# =====================================
# CONFIGURATION
# =====================================
CFG = {
    "db_name": "cpsc2021",
    "data_dir": "CPSC2021",
    "out_npz": "CPSC2021_3class_80_20.npz",
    "bp_lo": 0.5,
    "bp_hi": 50.0,
    "preferred_lead": "MLII",
    "fs": 500,
    "notch_hz": 50.0,
    "notch_q": 30.0,
    "classes": ["AF", "PAF", "Non-AF"],  # 3-class setup
    "beat_winL": 150,  # samples before R peak
    "beat_winR": 150,  # samples after R peak
}

# =====================================
# SIGNAL PROCESSING
# =====================================
def preprocess_signal(sig, fs, bp_lo, bp_hi, notch_hz, notch_q):
    sig = signal.detrend(sig)
    b, a = signal.butter(4, [bp_lo/(fs/2), bp_hi/(fs/2)], btype="band")
    sig = signal.filtfilt(b, a, sig)
    if notch_hz:
        b_notch, a_notch = signal.iirnotch(notch_hz/(fs/2), notch_q)
        sig = signal.filtfilt(b_notch, a_notch, sig)
    sig = (sig - np.mean(sig)) / np.std(sig)
    return sig

def choose_channel_idx(sig_name, preferred):
    if preferred in sig_name:
        return sig_name.index(preferred)
    return 0

# =====================================
# SAFE RECORD LOADING
# =====================================
def safe_load_record(data_dir, rec_id):
    try:
        rec_path = os.path.join(data_dir, rec_id)
        rec = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, 'atr')
        return rec, ann
    except Exception as e:
        print(f"Skipping record {rec_id}: {e}")
        return None, None

def list_record_ids(data_dir):
    return sorted([f.replace(".dat","") for f in os.listdir(data_dir) if f.endswith(".dat")])

# =====================================
# EXTRACT BEATS BASED ON R-PEAKS
# =====================================
def records_to_Xy(data_dir, records, cfg):
    X, y, recmap = [], [], []
    for rec_id in records:
        rec, ann = safe_load_record(data_dir, rec_id)
        if rec is None:
            continue

        ch_idx = choose_channel_idx(rec.sig_name, cfg["preferred_lead"])
        sig = rec.p_signal[:, ch_idx]
        sig = preprocess_signal(sig, cfg["fs"], cfg["bp_lo"], cfg["bp_hi"], cfg["notch_hz"], cfg["notch_q"])

        # Determine class from filename
        if "AF" in rec_id.upper() and "PAF" not in rec_id.upper():
            cls = "AF"
        elif "PAF" in rec_id.upper():
            cls = "PAF"
        else:
            cls = "Non-AF"

        for r in ann.sample:
            start = r - cfg["beat_winL"]
            end = r + cfg["beat_winR"]
            if start < 0 or end >= len(sig):
                continue
            beat = sig[start:end+1]
            if len(beat) == cfg["beat_winL"] + cfg["beat_winR"] + 1:
                X.append(beat)
                y.append(cls)
                recmap.append(rec_id)

        print(f"Processed record {rec_id}, total beats so far: {len(y)}")

    if len(X) == 0:
        raise ValueError("No beats extracted! Check dataset and beat extraction parameters.")

    return np.array(X), np.array(y), np.array(recmap)

# =====================================
# VISUALIZATION
# =====================================
def plot_random_beats(X, y, classes, n=10):
    plt.figure(figsize=(12,6))
    for i in range(n):
        idx = random.randint(0, len(X)-1)
        plt.plot(X[idx], alpha=0.7, label=f"{y[idx]}" if i==0 else "")
    plt.title(f"Random {n} Beats")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

def plot_class_distribution(y, classes):
    counts = [np.sum(np.array(y) == cls) for cls in classes]
    plt.figure(figsize=(6,4))
    plt.bar(classes, counts)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

# =====================================
# MAIN
# =====================================
def main():
    cfg = CFG
    if not os.path.exists(cfg["data_dir"]):
        os.makedirs(cfg["data_dir"])
        print(f"Please download CPSC2021 dataset manually into {cfg['data_dir']}")
        return

    all_local = list_record_ids(cfg["data_dir"])
    if len(all_local) == 0:
        print(f"No records found in {cfg['data_dir']}")
        return
    print(f"Total local records found: {len(all_local)}")

    X_all, y_all, recmap_all = records_to_Xy(cfg["data_dir"], all_local, cfg)

    cls2idx = {c:i for i,c in enumerate(cfg["classes"])}
    y_all_i = np.array([cls2idx[c] for c in y_all], dtype=np.int64)

    # Train/Test split
    Xtr, Xte, ytr_i, yte_i, tr_recmap, te_recmap = train_test_split(
        X_all, y_all_i, recmap_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all_i
    )

    np.savez_compressed(
        cfg["out_npz"],
        X_train=Xtr, y_train=ytr_i, rec_train=np.array(tr_recmap),
        X_test=Xte, y_test=yte_i, rec_test=np.array(te_recmap),
        classes=np.array(cfg["classes"]),
        fs=np.array(cfg["fs"])
    )
    print(f"Saved dataset: {cfg['out_npz']}")

    # Visualization
    plot_random_beats(Xtr, [cfg["classes"][i] for i in ytr_i], cfg["classes"], n=20)
    plot_class_distribution([cfg["classes"][i] for i in ytr_i], cfg["classes"])

if __name__ == "__main__":
    main()


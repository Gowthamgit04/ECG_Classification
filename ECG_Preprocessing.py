#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[4]:


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
    "db_name": "mitdb",
    "data_dir": "data",
    "out_npz": "mitdb_beats_80_20.npz",
    "fs": 360,
    "beat_winL": 99,
    "beat_winR": 100,
    "bp_lo": 0.5,
    "bp_hi": 40.0,
    "notch_hz": 50.0,
    "notch_q": 30.0,
    "preferred_lead": "MLII",
    "keep_aami": ["N", "S", "V", "F", "Q"],
}

# =====================================
# SIGNAL PROCESSING HELPERS
# =====================================
def dl_if_needed(db_name, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    wfdb.dl_database(db_name, dl_dir=data_dir)

def list_record_ids(data_dir):
    return sorted([f.replace(".dat", "") for f in os.listdir(data_dir) if f.endswith(".dat")])

def load_record(data_dir, rec_id):
    rec_path = os.path.join(data_dir, rec_id)
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")
    return rec, ann

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

def records_to_Xy(data_dir, records, cfg):
    X, y, recmap = [], [], []
    for rec_id in records:
        rec, ann = load_record(data_dir, rec_id)
        ch_idx = choose_channel_idx(rec.sig_name, cfg["preferred_lead"])
        sig = rec.p_signal[:, ch_idx]
        sig = preprocess_signal(sig, cfg["fs"], cfg["bp_lo"], cfg["bp_hi"], cfg["notch_hz"], cfg["notch_q"])

        for r, sym in zip(ann.sample, ann.symbol):
            if sym not in cfg["keep_aami"]:
                continue
            start = r - cfg["beat_winL"]
            end = r + cfg["beat_winR"]
            if start < 0 or end >= len(sig):
                continue
            beat = sig[start:end+1]
            if len(beat) == cfg["beat_winL"] + cfg["beat_winR"] + 1:
                X.append(beat)
                y.append(sym)
                recmap.append(rec_id)
        print(f"Record {rec_id}: {len(y)} beats so far")
    return np.array(X), np.array(y), np.array(recmap)

# =====================================
# VISUALIZATION HELPERS
# =====================================
def plot_raw_signal(rec, ann, cfg, n_samples=2000):
    sig = rec.p_signal[:, choose_channel_idx(rec.sig_name, cfg["preferred_lead"])]
    sig = preprocess_signal(sig, cfg["fs"], cfg["bp_lo"], cfg["bp_hi"], cfg["notch_hz"], cfg["notch_q"])

    plt.figure(figsize=(12, 4))
    plt.plot(sig[:n_samples], label="ECG")
    for r, sym in zip(ann.sample, ann.symbol):
        if r < n_samples:
            plt.axvline(r, color="r", linestyle="--", alpha=0.5)
            plt.text(r, sig[r], sym, color="r", fontsize=8, rotation=90, va="bottom")
    plt.title("Raw ECG with Beat Annotations")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (normalized)")
    plt.legend()
    plt.show()

def plot_random_beats(X, y, classes, n=10):
    plt.figure(figsize=(12, 6))
    for i in range(n):
        idx = random.randint(0, len(X) - 1)
        plt.plot(X[idx], alpha=0.7, label=f"{classes[y[idx]]}" if i == 0 else "")
    plt.title(f"Random {n} Beats (windowed around R-peak)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

def plot_class_distribution(y, classes):
    counts = np.bincount(y)
    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts)
    plt.title("Class Distribution of Beats")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

# =====================================
# MAIN
# =====================================
def main():
    cfg = CFG
    dl_if_needed(cfg["db_name"], cfg["data_dir"])
    all_local = list_record_ids(cfg["data_dir"])
    print(f"Total available records: {len(all_local)}")

    print("\n--- Extracting ALL RECORDS ---")
    X_all, y_all, recmap_all = records_to_Xy(cfg["data_dir"], all_local, cfg)

    classes = sorted(list(set(cfg["keep_aami"]).intersection(set(np.unique(y_all)))))
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_all_i = np.array([cls2idx[c] for c in y_all], dtype=np.int64)

    print(f"\nClasses kept: {classes}")
    print(f"Total beats: {X_all.shape[0]}, Beat length: {X_all.shape[1]}")

    # 80/20 split
    Xtr, Xte, ytr_i, yte_i, tr_recmap, te_recmap = train_test_split(
        X_all, y_all_i, recmap_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all_i
    )
    print(f"Train beats: {Xtr.shape[0]}, Test beats: {Xte.shape[0]}")

    np.savez_compressed(
        cfg["out_npz"],
        X_train=Xtr, y_train=ytr_i, rec_train=np.array(tr_recmap),
        X_test=Xte, y_test=yte_i, rec_test=np.array(te_recmap),
        classes=np.array(classes), fs=np.array(cfg["fs"])
    )
    print(f"Saved dataset: {cfg['out_npz']}")

    # ==============================
    # OPTIONAL VISUALIZATION
    # ==============================
    # Plot random beats
    plot_random_beats(Xtr, ytr_i, classes, n=20)

    # Plot class distribution
    plot_class_distribution(ytr_i, classes)

    # Plot raw signal for one record
    rec, ann = load_record(cfg["data_dir"], all_local[0])
    plot_raw_signal(rec, ann, cfg, n_samples=2000)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- settings --------
NPZ_PATH = "mitbih_beats_80_20.npz"  # change if your file has a different name/path
SPLIT = "train"  # "train" or "test"
PICK_STRATEGY = "prototype"  # "first", "random", "prototype"
OUT_PNG = Path("one_beat_per_class1.png")

# -------- load --------
d = np.load(NPZ_PATH, allow_pickle=True)
X = d["X_train"] if SPLIT == "train" else d["X_test"]
y = d["y_train"] if SPLIT == "train" else d["y_test"]
classes = d["classes"].astype(str)            # e.g., ['F','N','Q','S','V'] (order aligns with label integers)
fs = int(d["fs"])
n_win = X.shape[1]
t = np.arange(n_win) / fs  # seconds

print(f"Loaded {SPLIT}: X={X.shape}, y={y.shape}, fs={fs} Hz, classes={list(classes)}")

# -------- pick 1 beat per class --------
rng = np.random.default_rng(0)

def pick_index_for_class(ci, strategy="first"):
    idxs = np.where(y == ci)[0]
    if len(idxs) == 0:
        return None
    if strategy == "first":
        return idxs[0]
    elif strategy == "random":
        return rng.choice(idxs)
    elif strategy == "prototype":
        # pick beat closest (L2) to the class mean -> a representative/typical example
        Xm = X[idxs]
        mean_wave = Xm.mean(axis=0, dtype=np.float64)
        dists = np.sum((Xm - mean_wave)**2, axis=1)
        return idxs[np.argmin(dists)]
    else:
        raise ValueError("strategy must be one of {'first','random','prototype'}")

picked = []
for ci, cls in enumerate(classes):
    idx = pick_index_for_class(ci, PICK_STRATEGY)
    picked.append((ci, cls, idx))

# -------- plot --------
fig, axes = plt.subplots(len(classes), 1, figsize=(9, 1.8*len(classes)), sharex=True)
if len(classes) == 1:
    axes = [axes]

for ax, (ci, cls, idx) in zip(axes, picked):
    if idx is None:
        ax.text(0.5, 0.5, f"No {cls} beats in {SPLIT}", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
        continue
    ax.plot(t, X[idx], lw=1.2)
    ax.set_ylabel(cls)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"One representative beat per class ({SPLIT}, strategy={PICK_STRATEGY})", y=0.98)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")


# In[ ]:





# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- settings --------
NPZ_PATH = "mitbih_beats_80_20.npz"  # change if your file has a different name/path
SPLIT = "train"  # "train" or "test"
PICK_STRATEGY = "prototype"  # "first", "random", "prototype"
OUT_PNG = Path("one_beat_per_class1.png")

# -------- load --------
d = np.load(NPZ_PATH, allow_pickle=True)
X = d["X_train"] if SPLIT == "train" else d["X_test"]
y = d["y_train"] if SPLIT == "train" else d["y_test"]
classes = d["classes"].astype(str)            # e.g., ['F','N','Q','S','V']
fs = int(d["fs"])
n_win = X.shape[1]
t = np.arange(n_win) / fs  # seconds

print(f"Loaded {SPLIT}: X={X.shape}, y={y.shape}, fs={fs} Hz, classes={list(classes)}")

# -------- counts per class --------
counts = {cls: np.sum(y == ci) for ci, cls in enumerate(classes)}

print("\nBeat counts per class:")
for cls, cnt in counts.items():
    print(f"  {cls}: {cnt}")

# -------- pick 1 beat per class --------
rng = np.random.default_rng(0)

def pick_index_for_class(ci, strategy="first"):
    idxs = np.where(y == ci)[0]
    if len(idxs) == 0:
        return None
    if strategy == "first":
        return idxs[0]
    elif strategy == "random":
        return rng.choice(idxs)
    elif strategy == "prototype":
        # pick beat closest (L2) to the class mean -> a representative/typical example
        Xm = X[idxs]
        mean_wave = Xm.mean(axis=0, dtype=np.float64)
        dists = np.sum((Xm - mean_wave)**2, axis=1)
        return idxs[np.argmin(dists)]
    else:
        raise ValueError("strategy must be one of {'first','random','prototype'}")

picked = []
for ci, cls in enumerate(classes):
    idx = pick_index_for_class(ci, PICK_STRATEGY)
    picked.append((ci, cls, idx))

# -------- plot --------
fig, axes = plt.subplots(len(classes), 1, figsize=(9, 1.8*len(classes)), sharex=True)
if len(classes) == 1:
    axes = [axes]

for ax, (ci, cls, idx) in zip(axes, picked):
    if idx is None:
        ax.text(0.5, 0.5, f"No {cls} beats in {SPLIT}", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
        continue
    ax.plot(t, X[idx], lw=1.2)
    ax.set_ylabel(f"{cls}\n(n={counts[cls]})")  # add counts per class
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"One representative beat per class ({SPLIT}, strategy={PICK_STRATEGY})", y=0.98)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





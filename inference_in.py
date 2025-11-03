import numpy as np
import wfdb
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------- Beat Mapping (AAMI Standard) ----------------
def map_to_aami_classes(symbols):
    mapping = {
       'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
        'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
        'V': 'V', 'E': 'V',
        'F': 'F',
        '/': 'Q', 'f': 'Q', 'Q': 'Q', '?': 'Q', 'P': 'Q', '|': 'Q' 
    }
    return np.array([mapping.get(sym, 'Q') for sym in symbols])  

# ---------------- Load CPSC ECG Beats ----------------
def load_cpsc_dataset(path):
    all_beats, all_labels = [], []
    record_names = sorted([f[:-4] for f in os.listdir(path) if f.endswith('.hea')])

    for rec in record_names:
        record_path = os.path.join(path, rec)
        try:
            signal, _ = wfdb.rdsamp(record_path)
            ann = wfdb.rdann(record_path, 'atr')
        except Exception as e:
            print(f"?? Skipping {rec} (error: {e})")
            continue

        symbols = map_to_aami_classes(np.array(ann.symbol))
        r_peaks = np.array(ann.sample)

        for i, r in enumerate(r_peaks):
            if r - 90 < 0 or r + 90 > len(signal):
                continue
            beat = signal[r - 90:r + 90, 0]
            all_beats.append(beat)
            all_labels.append(symbols[i])

    return np.array(all_beats), np.array(all_labels)


# ---------------- Preprocess ----------------
def preprocess_signals(X, y, le=None):
    scaler = StandardScaler()
    X_scaled = np.array([scaler.fit_transform(x.reshape(-1, 1)).flatten() for x in X])

    if le is None:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
    else:
        y_enc = le.transform([lbl if lbl in le.classes_ else 'Q' for lbl in y])

    return X_scaled[..., np.newaxis], y_enc, le


# ---------------- Main Inference ----------------
if __name__ == "__main__":
    # Path to your CPSC data folder (contains .dat, .hea, .atr)
    data_path = "cp"

    # Load your trained model (.keras format)
    model_path = "best_ecg_model_higher.keras"
    model = load_model(model_path, compile=False)
    print(f"? Loaded trained model from {model_path}")

    # Load dataset
    X_test, y_test_symbols = load_cpsc_dataset(data_path)
    print(f"\nTotal beats loaded: {len(X_test)}")

    # Encode labels using same AAMI classes
    X_test, y_test, le = preprocess_signals(X_test, y_test_symbols)

    # Predict
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n? Test Accuracy on CPSC dataset: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - CPSC 2021 ECG Classification")
    plt.show()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Disable GPU

import numpy as np
import wfdb
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Bidirectional,
                                     Dense, Dropout, Flatten, Multiply, Activation,
                                     RepeatVector, Permute)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


# ---------------- Beat Mapping (AAMI Standard) ----------------
def map_to_aami_classes(symbols):
    mapping = {
        # N - Normal
        'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
        # S - Supraventricular
        'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
        # V - Ventricular
        'V': 'V', 'E': 'V',
        # F - Fusion
        'F': 'F',
        # Q - Unknown / paced / others
        '/': 'Q', 'f': 'Q', 'Q': 'Q', '?': 'Q', 'P': 'Q', '|': 'Q'
    }
    return np.array([mapping.get(sym, 'Q') for sym in symbols])  # default to Q if unknown


# ---------------- Load MIT-BIH Data ----------------
def load_mitbih_dataset(path, records):
    all_beats, all_labels = [], []
    for record in records:
        record_path = os.path.join(path, str(record))
        signal, _ = wfdb.rdsamp(record_path)
        ann = wfdb.rdann(record_path, 'atr')

        symbols = map_to_aami_classes(np.array(ann.symbol))
        r_peaks = np.array(ann.sample)

        for i, r in enumerate(r_peaks):
            if r - 90 < 0 or r + 90 > len(signal):
                continue
            beat = signal[r - 90:r + 90, 0]
            all_beats.append(beat)
            all_labels.append(symbols[i])
    return np.array(all_beats), np.array(all_labels)


# ---------------- Preprocessing ----------------
def preprocess_signals(X, y, le=None):
    scaler = StandardScaler()
    X_scaled = np.array([scaler.fit_transform(x.reshape(-1, 1)).flatten() for x in X])
    if le is None:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
    else:
        y_enc = le.transform([lbl if lbl in le.classes_ else 'Q' for lbl in y])
    y_cat = to_categorical(y_enc, num_classes=len(le.classes_))
    return X_scaled, y_cat, le


# ---------------- Attention Block ----------------
def attention_block(inputs):
    e = Dense(1, activation='relu', name='att_dense')(inputs)
    e = Flatten(name='att_flat')(e)
    alpha = Activation('softmax', name='attention_weights')(e)
    alpha_rep = RepeatVector(inputs.shape[-1], name='att_repeat')(alpha)
    alpha_rep = Permute([2, 1], name='att_permute')(alpha_rep)
    attended = Multiply(name='attention_mul')([inputs, alpha_rep])
    return attended


# ---------------- Model Architecture ----------------
def build_model(n_timesteps, n_channels=1, n_classes=5):
    inp = Input(shape=(n_timesteps, n_channels), name='input')
    x = Conv1D(128, kernel_size=2, activation='relu', name='conv1')(inp)
    x = MaxPooling1D(2, name='pool1')(x)
    x = Conv1D(256, kernel_size=3, activation='relu', name='conv2')(x)
    x = MaxPooling1D(2, name='pool2')(x)
    x = Bidirectional(LSTM(512, return_sequences=True), name='bilstm')(x)
    x = attention_block(x)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='dense1')(x)
    x = Dropout(0.3, name='dropout')(x)
    out = Dense(n_classes, activation='softmax', name='predictions')(x)
    return Model(inputs=inp, outputs=out, name='cnn_bilstm_attention')


# ---------------- Main Script ----------------
if __name__ == "__main__":
    data_path = "mit/"  # Change this to your dataset path

    patient_records = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                       111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
                       122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
                       209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
                       222, 223, 228, 230, 231, 232, 233, 234]

    train_patients, test_patients = train_test_split(patient_records, test_size=0.25, random_state=None)
    print(f"Training patients: {train_patients}")
    print(f"Testing patients: {test_patients}")

    X_train, y_train = load_mitbih_dataset(data_path, train_patients)
    X_test, y_test = load_mitbih_dataset(data_path, test_patients)

    X_train, y_train, le = preprocess_signals(X_train, y_train)
    X_test, y_test, _ = preprocess_signals(X_test, y_test, le)

    n_timesteps = X_train.shape[1]
    n_classes = y_train.shape[1]
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model = build_model(n_timesteps, 1, n_classes)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # ---------------- Training Callbacks ----------------
    checkpoint = ModelCheckpoint(
        "best_ecg_model_higher.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    # ---------------- Train Model ----------------
    history = model.fit(
        X_train, y_train,
        epochs=120,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )

    # ---------------- Evaluation ----------------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n? Test Accuracy on Unseen Patients: {test_acc:.4f}")

    print("\nClass Mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - Inter-Patient ECG Classification")
    plt.show()

    print("\n? Best model automatically saved as 'best_ecg_model.h5'")

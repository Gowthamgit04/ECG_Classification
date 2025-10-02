
"""
CNN + BiLSTM + Attention + Explainability
- Learning curves (train/val)
- t-SNE distribution plots (train, test, combined) on penultimate features
- SHAP/IG plotted as a ribbon *within* the signal
- Robust to feature-length mismatches (e.g., SHAP=252 vs signal=1260)

Save as: explainability_signal_ribbon_tsne.py
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense,
                                     Dropout, Flatten, Bidirectional, Activation,
                                     Permute, Multiply, RepeatVector)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ---------------- Settings ----------------
NPZ_PATH = "mitbih_beats_80_20.npz"
MODEL_PATH = Path("cnn_bilstm_attention_explain.h5")
EXPLAIN_DIR = Path("explain_plots")
EXPLAIN_DIR.mkdir(exist_ok=True)

# ---------------- Load dataset ----------------
d = np.load(NPZ_PATH, allow_pickle=True)
X = d["X_train"]          # (n_samples, n_timesteps)
y = d["y_train"]
classes = d["classes"].astype(str)
fs = int(d["fs"])
n_win = X.shape[1]
X = X.reshape((-1, n_win, 1))

# ---------------- Train-test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- One-hot encode labels ----------------
num_classes = len(classes)
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat  = to_categorical(y_test,  num_classes=num_classes)

# ---------------- Attention block ----------------
def attention_block(inputs):
    e = Dense(1, activation='relu', name='att_dense')(inputs)
    e = Flatten(name='att_flat')(e)
    alpha = Activation('softmax', name='attention_weights')(e)
    alpha_rep = RepeatVector(inputs.shape[-1], name='att_repeat')(alpha)
    alpha_rep = Permute([2, 1], name='att_permute')(alpha_rep)
    attended = Multiply(name='attention_mul')([inputs, alpha_rep])
    return attended

# ---------------- Build Model ----------------
def build_model(n_timesteps, n_channels=1, n_classes=2):
    inp = Input(shape=(n_timesteps, n_channels), name='input')
    x = Conv1D(64, kernel_size=5, activation='relu', name='conv1')(inp)
    x = MaxPooling1D(2, name='pool1')(x)
    x = Conv1D(128, kernel_size=3, activation='relu', name='conv2')(x)
    x = MaxPooling1D(2, name='pool2')(x)
    x = Bidirectional(LSTM(128, return_sequences=True), name='bilstm')(x)
    x = attention_block(x)
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='dense1')(x)
    x = Dropout(0.3, name='dropout')(x)
    out = Dense(n_classes, activation='softmax', name='predictions')(x)
    return Model(inputs=inp, outputs=out, name='cnn_bilstm_attention')

model = build_model(n_win, 1, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------- Train ----------------
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=50, batch_size=64,
    callbacks=[es],
    verbose=2
)

# ---------------- Learning curves (Train vs Val) ----------------
def plot_learning_curves(hist, out_path):
    acc = hist.history.get("accuracy", [])
    val_acc = hist.history.get("val_accuracy", [])
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])

    epochs = np.arange(1, len(loss) + 1)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label="Train")
    plt.plot(epochs, val_acc, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label="Train")
    plt.plot(epochs, val_loss, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

plot_learning_curves(history, EXPLAIN_DIR / "learning_curves.png")

# ---------------- Evaluate ----------------
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")
model.save(MODEL_PATH)
print(f"Saved model to: {MODEL_PATH}")

# ---------------- Confusion Matrix ----------------
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(EXPLAIN_DIR / "confusion_matrix.png", dpi=150); plt.close()
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=classes))

# ---------------- Prob. distribution (max class prob) ----------------
def plot_prob_distribution(probs, labels, title, out_path):
    maxp = probs.max(axis=1)
    plt.figure(figsize=(6,4))
    sns.histplot(maxp, bins=30, kde=True)
    plt.xlabel("Max predicted probability"); plt.ylabel("Count")
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

plot_prob_distribution(model.predict(X_train, verbose=0), y_train,
                       "Train: Max predicted probability", EXPLAIN_DIR / "prob_dist_train.png")
plot_prob_distribution(y_pred_prob, y_true,
                       "Test: Max predicted probability", EXPLAIN_DIR / "prob_dist_test.png")

# ============================================================
# ---------------- t-SNE on penultimate features -------------
# ============================================================
def get_penultimate_extractor(m):
    # Use the dense layer before dropout/softmax for embedding
    return Model(inputs=m.input, outputs=m.get_layer("dense1").output)

def balanced_subsample(Xa, ya, per_class_max=500, random_state=42):
    rng = np.random.default_rng(random_state)
    Xs, ys = [], []
    for c in np.unique(ya):
        idx = np.where(ya == c)[0]
        if len(idx) > per_class_max:
            idx = rng.choice(idx, size=per_class_max, replace=False)
        Xs.append(Xa[idx])
        ys.append(ya[idx])
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

def tsne_embed(features, random_state=42):
    n = len(features)
    # Pick a safe perplexity
    perplexity = max(5, min(30, (n - 1) // 3))
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca",
                perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(features)

def plot_tsne(Z, labels, title, out_path):
    plt.figure(figsize=(6.8,5.4))
    sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=[classes[i] for i in labels],
                    s=12, alpha=0.8, edgecolor=None, palette="tab10")
    plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

extractor = get_penultimate_extractor(model)

# Subsample for speed/balance (adjust per_class_max if you want more/less)
Xtr_sub, ytr_sub = balanced_subsample(X_train, y_train, per_class_max=600)
Xte_sub, yte_sub = balanced_subsample(X_test,  y_test,  per_class_max=600)

Ftr = extractor.predict(Xtr_sub, verbose=0)
Fte = extractor.predict(Xte_sub, verbose=0)

# Combined embedding so train/test share the same 2D space
F_all = np.vstack([Ftr, Fte])
Z_all = tsne_embed(F_all, random_state=42)
Z_tr, Z_te = Z_all[:len(Ftr)], Z_all[len(Ftr):]

plot_tsne(Z_tr, ytr_sub, "t-SNE (Train, penultimate features)", EXPLAIN_DIR / "tsne_train.png")
plot_tsne(Z_te, yte_sub, "t-SNE (Test, penultimate features)",  EXPLAIN_DIR / "tsne_test.png")

# Combined with domain label (Train/Test)
plt.figure(figsize=(6.8,5.4))
sns.scatterplot(x=Z_tr[:,0], y=Z_tr[:,1], s=12, alpha=0.6, label="Train")
sns.scatterplot(x=Z_te[:,0], y=Z_te[:,1], s=12, alpha=0.6, label="Test")
plt.title("t-SNE (Train vs Test, penultimate features)")
plt.legend(); plt.tight_layout(); plt.savefig(EXPLAIN_DIR / "tsne_train_test_combined.png", dpi=150); plt.close()

# ============================================================
# ---------------- Explainability Utilities ------------------
# ============================================================


def smooth(arr, window=15):
    if window is None or window < 2:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")

def resize_to(arr, target_len):
    """Linear interpolate 1D array to target_len."""
    arr = np.asarray(arr).astype(float).reshape(-1)
    src_len = len(arr)
    if src_len == target_len:
        return arr
    x_old = np.linspace(0.0, 1.0, src_len)
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, arr)

def attribution_ribbon(signal, attribution, scale_frac=0.35,
                       smooth_window=15, signal_mask_frac=0.02):
    sig = signal.squeeze().astype(float)
    att = attribution.squeeze().astype(float)

    # match lengths if needed
    if len(att) != len(sig):
        att = resize_to(att, len(sig))

    # smooth + mask
    att = smooth(att, window=smooth_window)
    max_sig = np.max(np.abs(sig)) + 1e-8
    mask = np.abs(sig) > (signal_mask_frac * max_sig)
    att = att * mask

    # normalize + scale to a fraction of signal amplitude
    max_att = np.max(np.abs(att)) + 1e-8
    att_scaled = (att / max_att) * (scale_frac * max_sig)

    y1 = sig
    y2 = sig + att_scaled
    return y1, y2

def plot_within_signal(signal, attribution, title, filename,
                       label_signal="Signal", label_attr="Attribution",
                       color_attr="orange"):
    y1, y2 = attribution_ribbon(signal, attribution)
    x = np.arange(len(y1))
    plt.figure(figsize=(12, 4))
    plt.plot(x, y1, label=label_signal, linewidth=1)
    plt.fill_between(x, y1, y2, alpha=0.25, label=label_attr, color=color_attr, linewidth=0)
    plt.plot(x, y2, linewidth=0.8, color=color_attr, alpha=0.9)
    plt.title(title); plt.xlabel("Time step"); plt.legend()
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()

# --- Integrated Gradients ---
def integrated_gradients(model, x, class_index, baseline=None, steps=100):
    if baseline is None:
        baseline = np.zeros_like(x)
    x_tensor = tf.convert_to_tensor(x[np.newaxis, ...], dtype=tf.float32)
    baseline_tensor = tf.convert_to_tensor(baseline[np.newaxis, ...], dtype=tf.float32)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    grads = []
    for alpha in alphas:
        inp = baseline_tensor + alpha * (x_tensor - baseline_tensor)
        with tf.GradientTape() as tape:
            tape.watch(inp)
            preds = model(inp)
            loss = preds[:, class_index]
        grad = tape.gradient(loss, inp)
        grads.append(grad.numpy()[0])
    avg_grads = np.mean(grads[:-1], axis=0)
    return (x - baseline) * avg_grads  # (T, 1)

# --- SHAP KernelExplainer ---
try:
    import shap
    shap_available = True
except Exception as e:
    print("SHAP not available:", e)
    shap_available = False

shap_explainer = None
if shap_available:
    try:
        # background width determines SHAP feature count
        bg_size = min(50, max(10, len(X_train)//20))
        idx = np.random.choice(len(X_train), bg_size, replace=False)
        background = X_train[idx]
        background_flat = background.reshape((len(background), -1))  # (B, n_features)

        def shap_predict(flat_samples):
            reshaped = flat_samples.reshape((-1, n_win, 1))
            return model.predict(reshaped, verbose=0)

        print(f"SHAP background features: {background_flat.shape[1]}  (signal length: {n_win})")
        shap_explainer = shap.KernelExplainer(shap_predict, background_flat)
        print("SHAP KernelExplainer ready.")
    except Exception as e:
        print("SHAP KernelExplainer initialization failed:", e)
        shap_explainer = None
        shap_available = False

# ============================================================
# ---------------- Generate Explanations ---------------------
# ============================================================

N_SAMPLES = min(5, len(X_test))
for i in range(N_SAMPLES):
    sample = X_test[i]
    true_label = y_test[i]
    pred_prob = model.predict(sample[np.newaxis, ...], verbose=0)[0]
    pred_label = np.argmax(pred_prob)

    print(f"\nSample {i} - True: {classes[true_label]}  Pred: {classes[pred_label]}  Prob: {pred_prob[pred_label]:.3f}")

    # ----- IG (ribbon inside signal) -----
    ig_zero = integrated_gradients(model, sample, pred_label, baseline=np.zeros_like(sample), steps=100)
    ig_mean = integrated_gradients(model, sample, pred_label, baseline=np.mean(X_train, axis=0), steps=100)
    ig_avg = (ig_zero + ig_mean) / 2.0
    plot_within_signal(sample, ig_avg,
                       title=f"Sample {i} Integrated Gradients",
                       filename=EXPLAIN_DIR / f"sample_{i}_ig.png",
                       label_attr="Integrated Gradients")

    # ----- SHAP (ribbon inside signal + length matching) -----
    if shap_available and shap_explainer is not None:
        try:
            sample_flat = sample.reshape(1, -1)  # (1, n_win)
            shap_vals_all = shap_explainer.shap_values(sample_flat, nsamples=100)

            # For multiclass, SHAP may return a list; pick the predicted class
            shap_for_pred = np.array(shap_vals_all[pred_label])[0] if isinstance(shap_vals_all, list) else np.array(shap_vals_all)[0]
            shap_1d = shap_for_pred.reshape(-1)
            np.savetxt(EXPLAIN_DIR / f"sample_{i}_shap_raw.txt", shap_1d, fmt="%.6f")

            plot_within_signal(sample, shap_1d,
                               title=f"Sample {i} SHAP (pred={classes[pred_label]})",
                               filename=EXPLAIN_DIR / f"sample_{i}_shap.png",
                               label_attr="SHAP")
            print(f"Saved SHAP plot for sample {i}  [len(signal)={len(sample)}, len(SHAP_raw)={len(shap_1d)}]")
        except Exception as e:
            print("SHAP failed for sample", i, ":", e)

print(f"\nAll plots saved to: {EXPLAIN_DIR.resolve()}")

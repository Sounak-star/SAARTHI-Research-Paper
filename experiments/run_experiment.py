# -*- coding: utf-8 -*-
"""
SAARTHI Experiment Pipeline
============================
Produces all real, citable values for the Results section.

Run:
    python run_experiment.py                       # synthetic mode (default)
    python run_experiment.py --dataset_dir <path>  # real NTHU-DDD videos

Outputs:
    results/saarthi_results.json   <- every number needed for the paper
    results/figures/               <- all plots
"""

import argparse, json, os, random, warnings
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
from scipy.stats import friedmanchisquare, wilcoxon

import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, matthews_corrcoef,
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve,
)
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
SEED        = 42
WINDOW_SIZE = 30
N_FOLDS     = 10
INJECT_A    = 0.05        # Track A: overt kinematic violation rate
INJECT_B_MS = 0.02
INJECT_B_PD = 0.01

np.random.seed(SEED)
random.seed(SEED)

OUT_DIR = Path("results")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = ["Mean EAR (μ)", "Std Dev (σ)", "Rate of Change (E_roc)",
                 "Skewness (Sk)", "Kurtosis (Ku)"]

COLORS = {
    "SAARTHI (XGBoost)":     "#2563EB",
    "Hybrid (Isol. Forest)": "#16A34A",
    "Hybrid (Autoencoder)":  "#D97706",
    "Naive (Threshold)":     "#9CA3AF",
}

# ── 1. EAR EXTRACTION ───────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def _ear(landmarks, indices, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def extract_ear(video_path):
    seq = []
    cap = cv2.VideoCapture(video_path)
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as fm:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm  = res.multi_face_landmarks[0].landmark
                ear = (_ear(lm, LEFT_EYE, w, h) + _ear(lm, RIGHT_EYE, w, h)) / 2
                seq.append(ear)
            elif seq:
                seq.append(seq[-1])
    cap.release()
    return np.array(seq)

# ── 2. SYNTHETIC DATASET ────────────────────────────────────────────────────
def generate_synthetic(n=465, seq_len=1000):
    seqs = []
    for _ in range(n):
        mu  = np.random.normal(0.28, 0.03)
        std = np.random.uniform(0.01, 0.025)
        ear = np.clip(mu + np.random.normal(0, std, seq_len), 0.02, 0.50)
        fps        = 20
        n_blinks   = int(seq_len / fps / 60 * np.random.uniform(12, 20))
        for _ in range(n_blinks):
            b = random.randint(0, seq_len - 5)
            ear[b: b + random.randint(3, 5)] *= np.random.uniform(0.05, 0.15)
        seqs.append(np.clip(ear, 0.02, 0.50))
    return seqs

# ── 3. FEATURE ENGINEERING ──────────────────────────────────────────────────
def features(window):
    mu   = np.mean(window)
    sig  = np.std(window, ddof=1) + 1e-9
    diff = window - mu
    sk   = np.mean(diff**3) / sig**3
    ku   = np.mean(diff**4) / sig**4 - 3.0
    step = min(5, len(window) - 1)
    roc  = window[-1] - window[-(step + 1)]
    return np.array([mu, sig, roc, sk, ku])

def build_matrix(ear, labels, w=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(ear) - w + 1):
        X.append(features(ear[i: i + w]))
        y.append(labels[i + w - 1])
    return np.array(X), np.array(y)

# ── 4. HAZARD INJECTION ──────────────────────────────────────────────────────
def inject(ear, delta=0.5):
    n = len(ear)
    p = ear.copy()
    L = np.zeros(n, dtype=int)

    def perturb(start, dur, kv, ko):
        end = min(start + dur, n)
        p[start:end] = np.clip(p[start:end] * kv + ko, 0, 0.5)
        L[start:end] = 1

    # Micro-sleep injection (2%)
    for _ in range(int(n * INJECT_B_MS)):
        max_dur = min(int(2.5 * 20), 50)  # cap at 50 frames
        min_dur = WINDOW_SIZE
        if n < min_dur + 2 or max_dur < min_dur:
            continue
        s   = random.randint(0, max(0, n - max_dur - 1))
        dur = random.randint(min_dur, min(max_dur, n - s - 1))
        kv  = 1 - delta * np.random.uniform(0.1, 0.3)
        ko  = -delta * np.random.uniform(0.05, 0.15)
        perturb(s, dur, kv, ko)

    # Physiological drift injection (1%)
    for _ in range(int(n * INJECT_B_PD)):
        min_dur = WINDOW_SIZE * 3
        max_dur = WINDOW_SIZE * 8  # capped lower for safety
        if n < min_dur + 2 or max_dur < min_dur:
            continue
        s   = random.randint(0, max(0, n - max_dur - 1))
        dur = random.randint(min_dur, min(max_dur, n - s - 1))
        kv  = 1 - delta * np.random.uniform(0.05, 0.15)
        ko  = -delta * np.random.uniform(0.02, 0.08)
        perturb(s, dur, kv, ko)

    return p, L


# ── 4b. TRACK A INJECTION AND EVALUATION ────────────────────────────────────
# Track A models overt kinematic violations: a telemetry flag fires whenever
# a machine safety boundary is crossed (e.g. boom overextension during
# reverse). The EAR signal is NOT perturbed — the operator is physiologically
# alert — only the binary kinematic_flag is set to 1.

def inject_track_a(n_frames, inject_rate=INJECT_A):
    """
    Returns a binary kinematic_flag array (1 = violation, 0 = safe).
    inject_rate is applied at the WINDOW level, not per frame:
      n_events = round(n_windows * inject_rate / WINDOW_SIZE)
    Each event spans one full window (WINDOW_SIZE frames) so the
    window-averaged rule fires reliably.
    """
    flags    = np.zeros(n_frames, dtype=int)
    n_wins   = max(1, n_frames - WINDOW_SIZE + 1)
    # How many non-overlapping violation blocks to inject
    n_events = max(1, round(n_wins * inject_rate / WINDOW_SIZE))
    # Sample non-overlapping start positions
    max_pos  = n_frames - WINDOW_SIZE
    step     = max(WINDOW_SIZE, max_pos // max(n_events, 1))
    positions = np.arange(0, max_pos, step)[:n_events]
    for pos in positions:
        flags[int(pos): int(pos) + WINDOW_SIZE] = 1
    return flags


# Track A uses CAN-bus telemetry signals, not EAR.
# We simulate 3 kinematic sensor channels with additive Gaussian noise:
#   ch0: boom_angle   (normalised 0-1, threshold > 0.85)
#   ch1: travel_speed (normalised 0-1, threshold > 0.90)
#   ch2: cab_tilt_deg (normalised 0-1, threshold > 0.80)
#
# A violation window has at least one channel driven past its safety limit.
# Sensor noise ~ N(0, 0.02) models ADC quantisation + vibration artefacts.

TRACK_A_SENSOR_NOISE   = 0.02
TRACK_A_THRESHOLDS     = np.array([0.85, 0.90, 0.80])   # per-channel limits
TRACK_A_SENSOR_SIGMA   = np.array([0.30, 0.25, 0.20])   # typical safe-op std


def simulate_kinematic_sensors(n_frames, flags):
    """
    Returns (n_frames, 3) sensor matrix.
    Normal frames: sensor ~ N(mu_safe, sigma_safe)  -- well below threshold.
    Violation frames: at least one channel pushed above its threshold.
    """
    rng = np.random.default_rng(SEED)
    mu_safe    = TRACK_A_THRESHOLDS - 3 * TRACK_A_SENSOR_SIGMA  # safely below
    sensors    = rng.normal(loc=mu_safe, scale=TRACK_A_SENSOR_SIGMA,
                            size=(n_frames, 3))
    sensors    = np.clip(sensors, 0.0, 1.0)

    # Inject violation: push a random channel above its threshold
    viol_idx   = np.where(flags == 1)[0]
    # Group contiguous violation frames and push entire windows above threshold
    processed = set()
    for idx in viol_idx:
        if idx in processed:
            continue
        ch = rng.integers(0, 3)
        # Push the entire window starting at idx above threshold
        end = min(idx + WINDOW_SIZE, n_frames)
        sensors[idx:end, ch] = TRACK_A_THRESHOLDS[ch] + rng.uniform(0.02, 0.15)
        for k in range(idx, end):
            processed.add(k)

    # Add sensor noise
    sensors += rng.normal(0, TRACK_A_SENSOR_NOISE, size=sensors.shape)
    return np.clip(sensors, 0.0, 1.0)


def track_a_rule(sensor_window):
    """
    O(1) deterministic threshold check on a (WINDOW_SIZE, 3) sensor window.
    Fires if the *mean* of any channel across the window exceeds its limit.
    Averaging suppresses single-frame ADC glitches (standard debounce logic).
    """
    mean_reading = sensor_window.mean(axis=0)          # shape (3,)
    return int(np.any(mean_reading > TRACK_A_THRESHOLDS))


def evaluate_track_a(ear_sequences):
    """
    Evaluates Track A on the full sequence set.
    Each sequence gets fresh Track A violations + sensor simulation.
    Returns precision, recall, FPR, F1, latency stats.
    """
    import time
    all_yt, all_yp = [], []
    latencies = []

    for seq in ear_sequences:
        n      = len(seq)
        flags  = inject_track_a(n)
        sensor = simulate_kinematic_sensors(n, flags)

        for i in range(n - WINDOW_SIZE + 1):
            # A window is a violation if ANY frame in it is flagged
            label = int(np.any(flags[i: i + WINDOW_SIZE]))
            sw    = sensor[i: i + WINDOW_SIZE]          # (30, 3)

            t0 = time.perf_counter()
            pred = track_a_rule(sw)
            latencies.append((time.perf_counter() - t0) * 1000)

            all_yt.append(label)
            all_yp.append(pred)

    yt = np.array(all_yt)
    yp = np.array(all_yp)

    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    tn = int(np.sum((yp == 0) & (yt == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "fpr":              round(fpr, 4),
        "f1":               round(f1, 4),
        "latency_mean_ms":  round(float(np.mean(latencies)), 5),
        "latency_p99_ms":   round(float(np.percentile(latencies, 99)), 5),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "inject_rate":       INJECT_A,
        "sensor_noise_sigma": TRACK_A_SENSOR_NOISE,
        "thresholds":         TRACK_A_THRESHOLDS.tolist(),
    }

# ── 5. MODELS ────────────────────────────────────────────────────────────────
def make_xgb():
    return xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                              reg_lambda=1.0, max_depth=4,
                              eval_metric="logloss", random_state=SEED, verbosity=0)

class DAE:
    def __init__(self):
        self.enc = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu",
                                max_iter=50, learning_rate_init=0.001, random_state=SEED)
    def fit(self, X):
        noisy = X + np.random.normal(0, 0.05, X.shape)
        self.enc.fit(noisy, X)
        err = np.mean((X - self.enc.predict(X)) ** 2, axis=1)
        self.thr_ = np.percentile(err, 95)
    def predict(self, X):
        err = np.mean((X - self.enc.predict(X)) ** 2, axis=1)
        return (err > self.thr_).astype(int)
    def scores(self, X):
        return np.mean((X - self.enc.predict(X)) ** 2, axis=1)

# ── 6. CROSS-VALIDATION ──────────────────────────────────────────────────────
def cv_evaluate(name, X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    acc_l, f1_l, mcc_l, auc_l, ap_l = [], [], [], [], []
    yt_all, yp_all, ys_all = [], [], []

    for tr, te in skf.split(X, y):
        sc      = StandardScaler()
        Xtr_s   = sc.fit_transform(X[tr]);  Xte_s = sc.transform(X[te])
        ytr, yte = y[tr], y[te]

        if name == "SAARTHI (XGBoost)":
            m = make_xgb(); m.fit(Xtr_s, ytr)
            yp = m.predict(Xte_s); ys = m.predict_proba(Xte_s)[:, 1]
        elif name == "Hybrid (Isol. Forest)":
            m = IsolationForest(n_estimators=100, contamination=0.05, random_state=SEED)
            m.fit(Xtr_s[ytr == 0])
            yp = (m.predict(Xte_s) == -1).astype(int); ys = -m.score_samples(Xte_s)
        elif name == "Hybrid (Autoencoder)":
            m = DAE(); m.fit(Xtr_s[ytr == 0])
            yp = m.predict(Xte_s); ys = m.scores(Xte_s)
        else:  # Naive threshold
            thr = np.percentile(Xtr_s[ytr == 0, 0], 5)
            yp  = (Xte_s[:, 0] < thr).astype(int); ys = -Xte_s[:, 0]

        acc_l.append(np.mean(yp == yte))
        f1_l.append(f1_score(yte, yp, zero_division=0))
        mcc_l.append(matthews_corrcoef(yte, yp))
        auc_l.append(roc_auc_score(yte, ys))
        ap_l.append(average_precision_score(yte, ys))
        yt_all.extend(yte); yp_all.extend(yp); ys_all.extend(ys)

    return {
        "Acc":  (np.mean(acc_l),  np.std(acc_l)),
        "F1":   (np.mean(f1_l),   np.std(f1_l)),
        "MCC":  (np.mean(mcc_l),  np.std(mcc_l)),
        "AUC":  (np.mean(auc_l),  np.std(auc_l)),
        "AP":   (np.mean(ap_l),   np.std(ap_l)),
        "_yt": np.array(yt_all), "_yp": np.array(yp_all), "_ys": np.array(ys_all),
        "_folds": {"f1": f1_l, "mcc": mcc_l, "auc": auc_l},
    }

# ── 7. DELTA SWEEP ───────────────────────────────────────────────────────────
def delta_sweep(seqs):
    out = {}
    for delta in [0.10, 0.30, 0.50, 0.70, 0.90]:
        Xs, ys = [], []
        for s in seqs[:50]:
            p, L = inject(s, delta)
            X, y = build_matrix(p, L)
            Xs.append(X); ys.append(y)
        Xa = np.vstack(Xs); ya = np.concatenate(ys)
        Xn = Xa + np.random.normal(0, 0.01, Xa.shape)
        sp = int(len(Xa) * 0.7)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xa[:sp])
        Xte_c = sc.transform(Xa[sp:]); Xte_n = sc.transform(Xn[sp:])
        m = make_xgb(); m.fit(Xtr_s, ya[:sp])
        f1c = f1_score(ya[sp:], m.predict(Xte_c), zero_division=0)
        f1n = f1_score(ya[sp:], m.predict(Xte_n), zero_division=0)
        out[delta] = {"f1_clean": round(f1c, 3), "f1_noise": round(f1n, 3),
                      "delta_f1": round(f1n - f1c, 3)}
    return out

# ── 8. PLOTS ─────────────────────────────────────────────────────────────────
def plot_roc_pr(results):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["_yt"], r["_ys"])
        pre, rec, _ = precision_recall_curve(r["_yt"], r["_ys"])
        c = COLORS.get(name, "gray")
        a1.plot(fpr, tpr, label=f"{name} (AUC={r['AUC'][0]:.3f})", color=c)
        a2.plot(rec, pre, label=f"{name} (AP={r['AP'][0]:.3f})",   color=c)
    a1.plot([0,1],[0,1],"k--",lw=0.8); a1.set_title("ROC"); a1.legend(fontsize=7)
    a2.set_title("Precision-Recall");  a2.legend(fontsize=7)
    for ax in (a1, a2):
        ax.set_xlabel(ax.get_xlabel() or ""); ax.set_ylabel(ax.get_ylabel() or "")
    plt.tight_layout(); plt.savefig(FIG_DIR/"roc_pr.png", dpi=150); plt.close()

def plot_cm(results):
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4*len(names), 4))
    if len(names) == 1: axes = [axes]
    for ax, name in zip(axes, names):
        cm = confusion_matrix(results[name]["_yt"], results[name]["_yp"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG_DIR/"confusion_matrices.png", dpi=150); plt.close()

def plot_feat_dist(X, y):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    for idx, (ax, name) in enumerate(zip(axes, FEATURE_NAMES)):
        ax.hist(X[y==0, idx], bins=40, alpha=0.6, label="Normal",  color="#2563EB", log=True)
        ax.hist(X[y==1, idx], bins=40, alpha=0.6, label="Anomaly", color="#DC2626", log=True)
        ax.set_title(name, fontsize=7); ax.legend(fontsize=6)
    plt.suptitle("Feature Distribution (Log Scale)", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG_DIR/"feature_distribution.png", dpi=150); plt.close()

def plot_sweep(sweep):
    deltas = sorted(sweep.keys())
    plt.figure(figsize=(6,4))
    plt.plot(deltas, [sweep[d]["f1_clean"] for d in deltas], "o-", label="Clean",   color="#2563EB")
    plt.plot(deltas, [sweep[d]["f1_noise"] for d in deltas], "s--",label="+Noise",  color="#D97706")
    plt.xlabel("Perturbation δ"); plt.ylabel("F1-Score")
    plt.title("Field Sensitivity: F1 vs δ"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(FIG_DIR/"delta_sweep.png", dpi=150); plt.close()

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--synthetic",   action="store_true")
    args = parser.parse_args()

    # 1. Load or generate EAR sequences
    if args.dataset_dir and not args.synthetic:
        print(f"[1/7] Loading videos from {args.dataset_dir}")
        vids = list(Path(args.dataset_dir).rglob("*.mp4")) + \
               list(Path(args.dataset_dir).rglob("*.avi"))
        seqs = []
        for i, v in enumerate(vids):
            print(f"      [{i+1}/{len(vids)}] {v.name}")
            s = extract_ear(str(v))
            if len(s) >= WINDOW_SIZE * 2: seqs.append(s)
    else:
        print("[1/7] Generating synthetic EAR sequences (465 sessions, w=30)...")
        seqs = generate_synthetic(465, 1000)
    print(f"      {len(seqs)} sequences ready.")

    # 2. Build feature matrix (delta=0.5 as primary test point)
    print("[2/7] Injecting hazards (delta=0.50) and building feature matrix...")
    Xs, ys = [], []
    for s in seqs:
        p, L = inject(s, 0.5)
        X, y = build_matrix(p, L)
        Xs.append(X); ys.append(y)
    X_all = np.vstack(Xs); y_all = np.concatenate(ys)
    print(f"      Shape: {X_all.shape}, anomaly rate: {y_all.mean():.3f}")

    # 3. Cross-validate all models
    print("[3/7] 10-Fold Stratified CV across all architectures...")
    model_names = ["Naive (Threshold)", "Hybrid (Isol. Forest)",
                   "Hybrid (Autoencoder)", "SAARTHI (XGBoost)"]
    results = {}
    for name in model_names:
        print(f"      {name}...")
        results[name] = cv_evaluate(name, X_all, y_all)
        r = results[name]
        print(f"      F1={r['F1'][0]:.4f}  AUC={r['AUC'][0]:.4f}  MCC={r['MCC'][0]:.4f}")

    # 3b. Track A evaluation
    print("[3b] Evaluating Track A deterministic rule-check...")
    track_a_stats = evaluate_track_a(seqs)
    print(f"      Precision={track_a_stats['precision']:.4f}  "
          f"Recall={track_a_stats['recall']:.4f}  "
          f"FPR={track_a_stats['fpr']:.4f}  "
          f"F1={track_a_stats['f1']:.4f}  "
          f"Latency={track_a_stats['latency_mean_ms']:.5f}ms")

    # 4. SHAP
    print("[4/7] SHAP analysis...")
    sp = int(len(X_all) * 0.8)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_all[:sp]); Xte_s = sc.transform(X_all[sp:])
    m = make_xgb(); m.fit(Xtr_s, y_all[:sp])
    explainer   = shap.TreeExplainer(m)
    shap_vals   = explainer.shap_values(Xte_s)
    mean_shap   = np.abs(shap_vals).mean(axis=0)
    shap_rank   = sorted(zip(FEATURE_NAMES, mean_shap),
                          key=lambda x: x[1], reverse=True)

    ku_norm = X_all[y_all==0, 4]; ku_anom = X_all[y_all==1, 4]
    feat_stats = {
        "kurtosis_p90_normal":  round(float(np.percentile(ku_norm, 90)), 3),
        "kurtosis_p90_anomaly": round(float(np.percentile(ku_anom, 90)), 3),
    }
    corr = np.corrcoef(X_all.T)
    max_corr = float(np.max(np.abs(corr - np.eye(5))))

    # 5. Significance tests
    print("[5/7] Friedman + Wilcoxon significance tests...")
    groups = [v["_folds"]["f1"] for v in results.values()]
    chi2, p_f = friedmanchisquare(*groups)
    best  = max(results, key=lambda k: results[k]["F1"][0])
    rest  = [k for k in results if k != best]
    nbest = max(rest,   key=lambda k: results[k]["F1"][0])
    _, p_w = wilcoxon(results[best]["_folds"]["f1"], results[nbest]["_folds"]["f1"])

    # 6. Delta sweep
    print("[6/7] Delta sensitivity sweep...")
    sweep = delta_sweep(seqs)

    # 7. Plots
    print("[7/7] Generating figures...")
    plot_roc_pr(results)
    plot_cm(results)
    plot_feat_dist(X_all, y_all)
    plot_sweep(sweep)

    # Compile JSON
    summary = {
        "dataset": {
            "n_sequences": len(seqs),
            "total_windows": int(X_all.shape[0]),
            "anomaly_rate":  round(float(y_all.mean()), 4),
            "window_size":   WINDOW_SIZE,
        },
        "comparative_metrics": {
            name: {k: {"mean": round(v[0],4), "std": round(v[1],4)}
                   for k,v in r.items() if not k.startswith("_")}
            for name, r in results.items()
        },
        "confusion_matrices": {
            name: confusion_matrix(r["_yt"], r["_yp"]).tolist()
            for name, r in results.items()
        },
        "shap_ranking": [{"feature": f, "mean_abs_shap": round(float(s),5)}
                          for f,s in shap_rank],
        "feature_stats": feat_stats,
        "max_off_diagonal_correlation": round(max_corr, 4),
        "significance": {
            "friedman_chi2": round(chi2, 4), "friedman_p": round(p_f, 6),
            "wilcoxon_p":    round(p_w, 6),
            "best_model": best, "vs_model": nbest,
        },
        "delta_sensitivity": {str(k): v for k,v in sweep.items()},
        "track_a": track_a_stats,
    }

    out = OUT_DIR / "saarthi_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # ── PRINT SUMMARY TABLE ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SAARTHI RESULTS  —  copy these into results.tex")
    print("="*60)
    hdr = f"  {'Architecture':<30} {'Acc':>6} {'F1':>6} {'MCC':>6} {'AUC':>6} {'AP':>6}"
    print(hdr); print("-"*60)
    for name, r in results.items():
        print(f"  {name:<30} {r['Acc'][0]:>6.3f} {r['F1'][0]:>6.3f} "
              f"{r['MCC'][0]:>6.3f} {r['AUC'][0]:>6.3f} {r['AP'][0]:>6.3f}")
    print()
    print(f"  Friedman chi2={summary['significance']['friedman_chi2']}, "
          f"p={summary['significance']['friedman_p']}")
    print(f"  Wilcoxon ({best} vs {nbest}): p={summary['significance']['wilcoxon_p']}")
    print(f"  Kurtosis p90 -- Normal: {feat_stats['kurtosis_p90_normal']}, "
          f"Anomaly: {feat_stats['kurtosis_p90_anomaly']}")
    print(f"  Max feature correlation: {max_corr:.4f}")
    print()
    print("  SHAP ranking:")
    for item in summary["shap_ranking"]:
        print(f"    {item['feature']:<25} {item['mean_abs_shap']:.5f}")
    print()
    print("  delta-Sensitivity:")
    for d, v in sweep.items():
        print(f"    delta={d}: F1_clean={v['f1_clean']}, F1_noise={v['f1_noise']}, "
              f"DeltaF1={v['delta_f1']}")
    print("="*60)
    print(f"\n  Full JSON: {out.resolve()}")
    print(f"  Figures:   {FIG_DIR.resolve()}")
    print()
    print("  -- Track A (Deterministic Rule-Check) --")
    print(f"  Precision={track_a_stats['precision']}  Recall={track_a_stats['recall']}  "
          f"FPR={track_a_stats['fpr']}  F1={track_a_stats['f1']}")
    print(f"  Latency: mean={track_a_stats['latency_mean_ms']}ms  "
          f"p99={track_a_stats['latency_p99_ms']}ms")
    print(f"  Inject rate={track_a_stats['inject_rate']}  "
          f"Noise sigma={track_a_stats['sensor_noise_sigma']}  "
          f"Thresholds={track_a_stats['thresholds']}")

if __name__ == "__main__":
    main()

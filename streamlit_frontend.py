"""
Streamlit Frontend for the Multi-label Psychological Traits MLP

This script was originally a Streamlit app. Some environments (like sandboxed CI or this runtime)
may not have `streamlit` installed which previously caused the crash
`ModuleNotFoundError: No module named 'streamlit'`.

To make the code robust, this file now supports two modes:
  1. Streamlit UI mode (if `streamlit` is available)
  2. CLI fallback mode (interactive / argparse) when `streamlit` is not present

It also supports two model backends:
  - Keras/TensorFlow (if available)
  - sklearn fallback (MultiOutputClassifier with RandomForest) otherwise

The goal: the script will not crash on missing Streamlit. If the environment lacks
TensorFlow, the script will still be usable with the sklearn fallback.

How to use:
  - If streamlit is installed: `streamlit run streamlit_frontend.py`
  - If streamlit is NOT installed: run CLI mode:
        python streamlit_frontend.py --mode cli --cmd batch_predict --csv path/to/file.csv

Commands (CLI mode):
  - single_predict  : prompt or accept feature values to get prediction
  - batch_predict   : provide a CSV path with features (and optionally labels)
  - train           : provide labeled CSV to train a model
  - gen_synthetic   : generate a synthetic dataset for quick testing (1200 samples by default)
  - run_tests       : run a few basic internal tests (no external dependencies)

Notes:
  - This file intentionally keeps feature/label names stable to match your dataset.
  - Model files: scaler saved to `scaler.joblib`. Models saved as `model.h5` (Keras) or
    `model.joblib` (sklearn fallback). Loading logic will try both.

"""

from __future__ import annotations
import os
import sys
import json
import argparse
import traceback
from typing import Tuple, Optional

# Try to import streamlit; fall back to CLI if unavailable
try:
    import streamlit as st
    USE_STREAMLIT = True
except Exception:
    st = None
    USE_STREAMLIT = False

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, jaccard_score, hamming_loss, roc_auc_score

# sklearn fallback models
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Try TensorFlow/Keras; if unavailable we will use sklearn fallback for training/prediction
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAVE_TF = True
except Exception:
    tf = None
    HAVE_TF = False

# ---------- Configuration: feature and label names (must match your dataset) ----------
FEATURE_NAMES = [
    "Alpha","Beta","Gamma","Pattern_Recognition_Score","Logical_Reasoning_Score",
    "Task_Completion_Speed","Stress_Response","Adaptability_Score","Heart_Rate_Variability",
    "Memory_Score","Emotional_Quotient","Attention_Span","Motivation_Level",
    "Decision_Making_Speed","Collaboration_Score","Creativity_Score","Focus_Level"
]

LABEL_NAMES = [
    "Creative_Thinker","Analytical_Thinker","Fast_Learner","Deep_Thinker",
    "Hardworking_and_Persistent","Adaptive_Learner","Business_and_Leadership",
    "Memorization_Expert","Innovative_Thinker","Strategic_Planner",
    "Detail_Oriented","Intuitive_Learner","Resilient_Thinker"
]

MODEL_PATH_KERAS = "model.h5"
MODEL_PATH_SKLEARN = "model.joblib"
SCALER_PATH = "scaler.joblib"

# ---------- Helper functions ----------

def build_keras_model(input_dim: int, output_dim: int, lr: float = 0.001):
    """Build the Keras MLP model. Requires TensorFlow to be available."""
    if not HAVE_TF:
        raise RuntimeError("TensorFlow is not available in this environment.")
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_sklearn_model() -> MultiOutputClassifier:
    """Build a sklearn MultiOutputClassifier using RandomForest as a fallback."""
    base = RandomForestClassifier(n_estimators=100, random_state=42)
    return MultiOutputClassifier(base)


def save_model_any(model, path_keras=MODEL_PATH_KERAS, path_sklearn=MODEL_PATH_SKLEARN):
    """Save model either as Keras or sklearn. Keras models are saved to model.h5. Others via joblib."""
    # Keras detection
    try:
        if HAVE_TF and hasattr(model, 'save'):
            model.save(path_keras)
            print(f"Saved Keras model to {path_keras}")
            return
    except Exception as e:
        print("Failed to save as Keras model, falling back to joblib:", e)

    # fallback: joblib
    joblib.dump(model, path_sklearn)
    print(f"Saved sklearn model to {path_sklearn}")


def load_model_if_exists() -> Optional[object]:
    """Try loading Keras model or sklearn model if present. Returns model or None."""
    # Try keras first
    if os.path.exists(MODEL_PATH_KERAS) and HAVE_TF:
        try:
            model = keras_load_model(MODEL_PATH_KERAS)
            print(f"Loaded Keras model from {MODEL_PATH_KERAS}")
            return model
        except Exception as e:
            print("Found Keras file but failed to load it (maybe corrupted or TF incompatible):", e)

    # Try sklearn joblib
    if os.path.exists(MODEL_PATH_SKLEARN):
        try:
            model = joblib.load(MODEL_PATH_SKLEARN)
            print(f"Loaded sklearn model from {MODEL_PATH_SKLEARN}")
            return model
        except Exception as e:
            print("Found sklearn model file but failed to load:", e)

    return None


def load_scaler_if_exists(path=SCALER_PATH):
    if os.path.exists(path):
        try:
            s = joblib.load(path)
            print(f"Loaded scaler from {path}")
            return s
        except Exception as e:
            print(f"Scaler exists but failed to load: {e}")
            return None
    return None


def preprocess_X(df: pd.DataFrame, scaler: Optional[StandardScaler] = None, fit_scaler: bool = False) -> Tuple[np.ndarray, StandardScaler]:
    """Return scaled X and the scaler used. If scaler is None and fit_scaler True it fits a new scaler."""
    if not all(f in df.columns for f in FEATURE_NAMES):
        missing = [f for f in FEATURE_NAMES if f not in df.columns]
        raise ValueError(f"Input dataframe missing feature columns: {missing}")
    X = df[FEATURE_NAMES].astype(float).values
    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        return Xs, scaler
    elif scaler is not None:
        return scaler.transform(X), scaler
    else:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        return Xs, scaler


def predict_and_format_any(model: object, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Predict probabilities and binarize with threshold. Supports Keras and sklearn MultiOutputClassifier.

    Returns (probs, labels_bin)
    - probs: shape (n_samples, n_labels) with probabilities in [0,1]
    - labels_bin: binary array same shape with 0/1
    """
    # Keras model detection: has predict and (HAVE_TF True)
    if HAVE_TF and hasattr(model, 'predict') and not isinstance(model, MultiOutputClassifier):
        probs = model.predict(X)
    else:
        # sklearn MultiOutputClassifier: predict_proba returns list of arrays (n_estimators) per label
        if hasattr(model, 'predict_proba'):
            proba_list = model.predict_proba(X)
            # predict_proba for MultiOutputClassifier returns list-like where each element is
            # an array of shape (n_samples, n_classes). For binary classification, we take column 1.
            probs_cols = []
            for arr in proba_list:
                # arr shape can be (n_samples, 2) or (n_samples, n_classes)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    probs_cols.append(arr[:, 1])
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    probs_cols.append(arr[:, 0])
                else:
                    # if multiclass in a label, take max prob column as heuristic
                    probs_cols.append(np.max(arr, axis=1))
            probs = np.vstack(probs_cols).T
        else:
            # Last resort: use predict() (hard predictions) to build probs as 0/1
            preds = model.predict(X)
            probs = preds.astype(float)

    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = probs.reshape(-1, len(LABEL_NAMES))
    # if number of columns mismatches labels, try to reshape if possible
    if probs.shape[1] != len(LABEL_NAMES):
        # try to infer and warn
        print(f"Warning: predicted probabilities shape {probs.shape} does not match label count {len(LABEL_NAMES)}")
        # try to reshape when possible
        if probs.size == X.shape[0] * len(LABEL_NAMES):
            probs = probs.reshape(X.shape[0], len(LABEL_NAMES))
        else:
            # pad or trim
            n = min(probs.shape[1], len(LABEL_NAMES))
            probs = probs[:, :n]
            if probs.shape[1] < len(LABEL_NAMES):
                # pad with zeros
                pad = np.zeros((X.shape[0], len(LABEL_NAMES) - probs.shape[1]))
                probs = np.hstack([probs, pad])

    labels_bin = (probs >= threshold).astype(int)
    return probs, labels_bin

# ---------- Simple utilities ----------

def ensure_columns(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def generate_synthetic_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    X = rng.normal(loc=50, scale=15, size=(n_samples, len(FEATURE_NAMES)))
    # Synthetic rule-based labels: threshold on some features + noise
    y = np.zeros((n_samples, len(LABEL_NAMES)), dtype=int)
    # Map some features to labels heuristically
    for i in range(n_samples):
        vals = X[i]
        # creative if Creativity_Score + (Alpha) high
        if vals[FEATURE_NAMES.index('Creativity_Score')] + vals[FEATURE_NAMES.index('Alpha')] > 100:
            y[i, LABEL_NAMES.index('Creative_Thinker')] = 1
        # analytical if Logical_Reasoning_Score high
        if vals[FEATURE_NAMES.index('Logical_Reasoning_Score')] > 60:
            y[i, LABEL_NAMES.index('Analytical_Thinker')] = 1
        # fast learner: Memory + Focus
        if vals[FEATURE_NAMES.index('Memory_Score')] + vals[FEATURE_NAMES.index('Focus_Level')] > 100:
            y[i, LABEL_NAMES.index('Fast_Learner')] = 1
        # random small chance for other labels
        noise = rng.rand(len(LABEL_NAMES))
        y[i] = np.where(noise < 0.05, 1, y[i])

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    for j, lname in enumerate(LABEL_NAMES):
        df[lname] = y[:, j]
    return df

# ---------- CLI Implementation ----------

def cli_single_predict(args):
    # interactive prompt for 17 features or read JSON from file
    if args.features_json:
        with open(args.features_json, 'r') as f:
            d = json.load(f)
        df_input = pd.DataFrame([d])
    else:
        print("Enter 17 feature values (press Enter to use default 50):")
        vals = {}
        for feat in FEATURE_NAMES:
            inp = input(f"{feat} [50]: ")
            if inp.strip() == "":
                vals[feat] = 50.0
            else:
                vals[feat] = float(inp)
        df_input = pd.DataFrame([vals])

    scaler = load_scaler_if_exists()
    Xs, scaler = preprocess_X(df_input, scaler=scaler, fit_scaler=(scaler is None))

    model = load_model_if_exists()
    if model is None:
        print("No saved model found. Train a model first (use train command) or provide a model file.")
        return

    probs, labels_bin = predict_and_format_any(model, Xs, threshold=args.threshold)
    print("Probabilities:")
    for name, p in zip(LABEL_NAMES, probs[0]):
        print(f"  {name}: {p:.4f}")
    print("Predicted labels:")
    for i,v in enumerate(labels_bin[0]):
        if v == 1:
            print(" ", LABEL_NAMES[i])


def cli_batch_predict(args):
    if not os.path.exists(args.csv):
        print("CSV file not found:", args.csv)
        return
    df = pd.read_csv(args.csv)
    if not ensure_columns(df, FEATURE_NAMES):
        missing = [f for f in FEATURE_NAMES if f not in df.columns]
        print("Missing features in CSV:", missing)
        return

    scaler = load_scaler_if_exists()
    Xs, scaler = preprocess_X(df, scaler=scaler, fit_scaler=(scaler is None))

    model = load_model_if_exists()
    if model is None:
        print("No saved model available. Train a model first or provide a model file.")
        return

    probs, labels_bin = predict_and_format_any(model, Xs, threshold=args.threshold)
    out_df = df.copy()
    for i,label in enumerate(LABEL_NAMES):
        out_df[label + '_prob'] = probs[:, i]
        out_df[label + '_pred'] = labels_bin[:, i]
    out_path = args.output or "predictions_output.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    if ensure_columns(df, LABEL_NAMES):
        y_true = df[LABEL_NAMES].astype(int).values
        y_pred = labels_bin
        exact_acc = np.mean(np.all(y_true==y_pred, axis=1))
        hl = hamming_loss(y_true, y_pred)
        try:
            jacc = jaccard_score(y_true, y_pred, average='samples')
        except Exception:
            jacc = None
        print(f"Exact-match accuracy: {exact_acc:.4f}")
        print(f"Hamming loss: {hl:.4f}")
        if jacc is not None:
            print(f"Jaccard (samples avg): {jacc:.4f}")


def cli_train(args):
    if not os.path.exists(args.csv):
        print("CSV file not found:", args.csv)
        return
    df = pd.read_csv(args.csv)
    if not ensure_columns(df, FEATURE_NAMES + LABEL_NAMES):
        missing_feats = [f for f in FEATURE_NAMES if f not in df.columns]
        missing_labels = [l for l in LABEL_NAMES if l not in df.columns]
        print("Missing features:", missing_feats)
        print("Missing labels:", missing_labels)
        return

    X = df[FEATURE_NAMES].astype(float).values
    y = df[LABEL_NAMES].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    scaler_new = StandardScaler()
    X_train_scaled = scaler_new.fit_transform(X_train)
    X_test_scaled = scaler_new.transform(X_test)

    if args.backend == 'keras' and HAVE_TF:
        print("Training Keras model...")
        model_new = build_keras_model(input_dim=X_train_scaled.shape[1], output_dim=y_train.shape[1], lr=args.lr)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model_new.fit(X_train_scaled, y_train, validation_split=0.2, epochs=int(args.epochs), batch_size=int(args.batch_size), callbacks=[es], verbose=1)
        save_model_any(model_new)
    else:
        print("Training sklearn fallback model (MultiOutput RandomForest)...")
        model_new = build_sklearn_model()
        model_new.fit(X_train_scaled, y_train)
        save_model_any(model_new)

    # save scaler
    joblib.dump(scaler_new, SCALER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")

    # Evaluation on test
    probs_test, y_pred = predict_and_format_any(model_new, X_test_scaled, threshold=args.threshold)
    exact_acc = np.mean(np.all(y_test==y_pred, axis=1))
    hl = hamming_loss(y_test, y_pred)
    try:
        jacc = jaccard_score(y_test, y_pred, average='samples')
    except Exception:
        jacc = None
    print(f"Exact-match accuracy (test): {exact_acc:.4f}")
    print(f"Hamming Loss (test): {hl:.4f}")
    if jacc is not None:
        print(f"Jaccard (samples avg) (test): {jacc:.4f}")


def cli_generate_synthetic(args):
    df = generate_synthetic_dataset(n_samples=args.n, random_state=args.random_state)
    out_path = args.output or 'synthetic_dataset.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic dataset to {out_path}")


def run_basic_tests():
    print("Running basic internal tests...")
    df = generate_synthetic_dataset(n_samples=200)
    try:
        Xs, scaler = preprocess_X(df, scaler=None, fit_scaler=True)
        assert Xs.shape[0] == 200 and Xs.shape[1] == len(FEATURE_NAMES)
        print("preprocess_X test passed")

        # Fit sklearn model as a smoke test
        model = build_sklearn_model()
        y = df[LABEL_NAMES].astype(int).values
        model.fit(Xs, y)
        probs, labels = predict_and_format_any(model, Xs[:5], threshold=0.5)
        assert probs.shape == (5, len(LABEL_NAMES))
        assert labels.shape == (5, len(LABEL_NAMES))
        print("sklearn model train/predict smoke test passed")

        print("All basic tests passed")
    except AssertionError as e:
        print("Assertion failed during tests:", e)
    except Exception:
        print("Exception during tests:\n", traceback.format_exc())

# ---------- Streamlit app (original UI adapted) ----------

def run_streamlit_app():
    # This code only runs if streamlit is installed (USE_STREAMLIT == True)
    st.set_page_config(page_title="Psych Traits MLP - Streamlit UI", layout='centered')
    st.title("Psychological Traits — Multi-label MLP (Streamlit Frontend)")

    st.sidebar.header("App controls")
    mode = st.sidebar.selectbox("Mode", ["Single prediction","Batch prediction (CSV)","Train model from CSV"]) 
    threshold = st.sidebar.slider("Prediction threshold", 0.0, 1.0, 0.7)

    # Load model and scaler if present
    model = load_model_if_exists()
    scaler = load_scaler_if_exists()

    st.sidebar.markdown("---")
    if model is not None:
        st.sidebar.success("Loaded model")
    else:
        st.sidebar.info("No saved model found. Use Train mode or upload a model.")

    if scaler is not None:
        st.sidebar.success("Loaded scaler.joblib")

    # Single prediction UI
    if mode == "Single prediction":
        st.header("Single sample prediction")
        st.markdown("Enter values for the 17 features (the app will scale using saved scaler or fit on these values).")

        user_input = {}
        cols = st.columns(2)
        for i, feat in enumerate(FEATURE_NAMES):
            with cols[i % 2]:
                user_input[feat] = st.number_input(feat, value=50.0, format="%.3f")

        if st.button("Predict"):
            df_input = pd.DataFrame([user_input])
            Xs, _ = preprocess_X(df_input, scaler=scaler, fit_scaler=(scaler is None))
            if model is None:
                st.error("No trained model available. Switch to Train mode or place a trained model 'model.h5' next to this script.")
            else:
                probs, labels_bin = predict_and_format_any(model, Xs, threshold=threshold)
                st.subheader("Predicted probabilities")
                prob_series = pd.Series(probs[0], index=LABEL_NAMES)
                st.dataframe(pd.DataFrame({'probability': prob_series}))

                predicted = [LABEL_NAMES[i] for i,v in enumerate(labels_bin[0]) if v==1]
                st.subheader("Predicted labels (threshold = %.2f)" % threshold)
                if predicted:
                    for p in predicted:
                        st.success(p)
                else:
                    st.info("No label predicted at this threshold.")

    # Batch prediction UI
    elif mode == "Batch prediction (CSV)":
        st.header("Batch prediction — upload CSV")
        st.markdown("CSV must contain the 17 feature columns named exactly as below. Optionally include the 13 label columns for evaluation.")
        st.write(FEATURE_NAMES)

        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head())

            missing_feats = [f for f in FEATURE_NAMES if f not in df.columns]
            if missing_feats:
                st.error("Missing feature columns: " + ", ".join(missing_feats))
            else:
                Xs, _ = preprocess_X(df, scaler=scaler, fit_scaler=(scaler is None))
                if model is None:
                    st.error("No trained model available. Use Train mode or place a trained model 'model.h5'.")
                else:
                    probs, labels_bin = predict_and_format_any(model, Xs, threshold=threshold)
                    out_df = df.copy()
                    for i,label in enumerate(LABEL_NAMES):
                        out_df[label + '_prob'] = probs[:, i]
                        out_df[label + '_pred'] = labels_bin[:, i]
                    st.write("Predictions preview:")
                    st.dataframe(out_df[[col for col in out_df.columns if col.endswith('_prob') or col.endswith('_pred')]].head())

                    # If true labels provided, compute metrics
                    if all(l in df.columns for l in LABEL_NAMES):
                        y_true = df[LABEL_NAMES].astype(int).values
                        y_pred = labels_bin
                        exact_acc = np.mean(np.all(y_true==y_pred, axis=1))
                        st.metric("Exact-match accuracy", f"{exact_acc:.4f}")

                        hl = hamming_loss(y_true, y_pred)
                        st.write(f"Hamming loss: {hl:.4f}")
                        try:
                            jacc = jaccard_score(y_true, y_pred, average='samples')
                            st.write(f"Jaccard (samples avg): {jacc:.4f}")
                        except Exception:
                            st.write("Jaccard score could not be computed")

                        cr = classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0, output_dict=True)
                        st.write(pd.DataFrame(cr).transpose())

                        # ROC-AUC per label (if possible)
                        aucs = {}
                        for i,label in enumerate(LABEL_NAMES):
                            try:
                                if len(np.unique(y_true[:, i])) == 2:
                                    auc = roc_auc_score(y_true[:, i], probs[:, i])
                                else:
                                    auc = None
                            except Exception:
                                auc = None
                            aucs[label] = auc
                        st.write("ROC-AUC per label:")
                        st.write(pd.Series(aucs))

    # Train UI
    elif mode == "Train model from CSV":
        st.header("Train a new model from CSV")
        st.markdown("Upload a labeled CSV with columns for the 17 features and the 13 labels. The script will split 80/20 and use 20% of train as validation.")
        st.write("Feature columns must be exactly:")
        st.write(FEATURE_NAMES)
        st.write("Label columns must be exactly:")
        st.write(LABEL_NAMES)

        uploaded = st.file_uploader("Upload labeled CSV for training", type=['csv'])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head())

            missing_feats = [f for f in FEATURE_NAMES if f not in df.columns]
            missing_labels = [l for l in LABEL_NAMES if l not in df.columns]
            if missing_feats or missing_labels:
                if missing_feats:
                    st.error("Missing feature columns: " + ", ".join(missing_feats))
                if missing_labels:
                    st.error("Missing label columns: " + ", ".join(missing_labels))
            else:
                test_size = st.slider("Test set size (fraction)", 0.05, 0.4, 0.2)
                batch_size = st.number_input("Batch size", value=32, step=1)
                epochs = st.number_input("Epochs", value=30, step=1)
                lr = st.number_input("Learning rate", value=0.001, format="%.6f")
                backend = st.selectbox("Backend", options=("keras" if HAVE_TF else "sklearn", "sklearn"))

                if st.button("Start Training"):
                    X = df[FEATURE_NAMES].astype(float).values
                    y = df[LABEL_NAMES].astype(int).values

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    scaler_new = StandardScaler()
                    X_train_scaled = scaler_new.fit_transform(X_train)
                    X_test_scaled = scaler_new.transform(X_test)

                    if backend == 'keras' and HAVE_TF:
                        model_new = build_keras_model(input_dim=X_train_scaled.shape[1], output_dim=y_train.shape[1], lr=lr)
                        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        history = model_new.fit(X_train_scaled, y_train, validation_split=0.2, epochs=int(epochs), batch_size=int(batch_size), callbacks=[es], verbose=1)
                        save_model_any(model_new)
                    else:
                        model_new = build_sklearn_model()
                        model_new.fit(X_train_scaled, y_train)
                        save_model_any(model_new)

                    joblib.dump(scaler_new, SCALER_PATH)
                    st.write(f"Saved scaler -> {SCALER_PATH}")

                    y_pred_probs = None
                    try:
                        y_pred_probs, y_pred = predict_and_format_any(model_new, X_test_scaled, threshold=threshold)
                    except Exception:
                        st.write("Could not compute predictions on test set")
                        y_pred = np.zeros_like(y_test)

                    exact_acc = np.mean(np.all(y_test==y_pred, axis=1))
                    hl = hamming_loss(y_test, y_pred)
                    try:
                        jacc = jaccard_score(y_test, y_pred, average='samples')
                    except Exception:
                        jacc = None

                    st.metric("Exact-match accuracy (test)", f"{exact_acc:.4f}")
                    st.write(f"Hamming Loss (test): {hl:.4f}")
                    if jacc is not None:
                        st.write(f"Jaccard (samples avg) (test): {jacc:.4f}")

                    # Plot loss and accuracy if keras
                    if HAVE_TF and 'history' in locals():
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        ax.plot(history.history.get('loss', []), label='train_loss')
                        ax.plot(history.history.get('val_loss', []), label='val_loss')
                        ax.set_title('Loss')
                        ax.legend()
                        st.pyplot(fig)

                    st.balloons()

    st.markdown("---")
    st.caption("Notes: \n- Make sure column names in any CSV exactly match the FEATURE_NAMES and LABEL_NAMES above.\n- The app saves model.h5 (Keras) or model.joblib (sklearn) and scaler.joblib in the current working directory.\n- For production or multi-user deployment, adapt file saving and concurrency handling.\n- This UI is intended for experimentation and educational purposes.")

# ---------- Entrypoint ----------

def main(argv=None):
    if USE_STREAMLIT:
        # If streamlit is available, Streamlit will execute run_streamlit_app when run via `streamlit run`.
        # But allowing direct `python streamlit_frontend.py` to open a simple CLI too.
        if len(sys.argv) > 1 and sys.argv[1] == '--mode' and 'cli' in sys.argv:
            # allow running CLI even when streamlit exists
            pass
        else:
            run_streamlit_app()
            return

    parser = argparse.ArgumentParser(description='CLI fallback for the Psych Traits MLP frontend')
    parser.add_argument('--mode', type=str, default='cli', help='Set to "cli" to use command line mode')
    sub = parser.add_subparsers(dest='cmd')

    p_single = sub.add_parser('single_predict')
    p_single.add_argument('--features_json', type=str, help='Path to JSON with feature values')
    p_single.add_argument('--threshold', type=float, default=0.5)

    p_batch = sub.add_parser('batch_predict')
    p_batch.add_argument('--csv', required=True, help='CSV file with feature columns')
    p_batch.add_argument('--output', required=False, help='Output CSV path')
    p_batch.add_argument('--threshold', type=float, default=0.5)

    p_train = sub.add_parser('train')
    p_train.add_argument('--csv', required=True, help='Labeled CSV for training')
    p_train.add_argument('--backend', choices=['sklearn', 'keras'], default='sklearn', help='Backend to train with')
    p_train.add_argument('--epochs', type=int, default=30)
    p_train.add_argument('--batch_size', type=int, default=32)
    p_train.add_argument('--lr', type=float, default=0.001)
    p_train.add_argument('--test_size', type=float, default=0.2)
    p_train.add_argument('--threshold', type=float, default=0.5)

    p_gen = sub.add_parser('gen_synthetic')
    p_gen.add_argument('--n', type=int, default=1200)
    p_gen.add_argument('--output', type=str, default='synthetic_dataset.csv')
    p_gen.add_argument('--random_state', type=int, default=42)

    p_tests = sub.add_parser('run_tests')

    args = parser.parse_args(argv)
    if args.cmd == 'single_predict':
        cli_single_predict(args)
    elif args.cmd == 'batch_predict':
        cli_batch_predict(args)
    elif args.cmd == 'train':
        cli_train(args)
    elif args.cmd == 'gen_synthetic':
        cli_generate_synthetic(args)
    elif args.cmd == 'run_tests':
        run_basic_tests()
    else:
        print("No command provided. Use --help to list available commands.")


if __name__ == '__main__':
    # If streamlit is available, it typically runs the file differently. We still support CLI fallback.
    try:
        main(sys.argv[1:])
    except Exception:
        print("Unhandled exception:\n", traceback.format_exc())


#!/usr/bin/env python3
"""
Benchmark Script: LightGBM Credit Card Fraud Detection
Lab 16: Cloud AI Environment Setup - Task 4.4
"""

import os
import sys
import time
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
import lightgbm as lgb


def find_dataset():
    """Locate or extract creditcard.csv from common directories."""
    possible_paths = [
        "creditcard.csv",
        os.path.expanduser("~/ml-benchmark/creditcard.csv"),
        "/home/ubuntu/ml-benchmark/creditcard.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "creditcard.csv"),
    ]

    for p in possible_paths:
        if os.path.isfile(p):
            return p

    # Check for zip file
    zip_paths = [
        "creditcardfraud.zip",
        os.path.expanduser("~/ml-benchmark/creditcardfraud.zip"),
        "/home/ubuntu/ml-benchmark/creditcardfraud.zip",
    ]
    for zp in zip_paths:
        if os.path.isfile(zp):
            extract_dir = os.path.dirname(zp) or "."
            print(f"[*] Found zip archive at {zp}, extracting to {extract_dir}...")
            with zipfile.ZipFile(zp, "r") as z:
                z.extractall(extract_dir)
            target = os.path.join(extract_dir, "creditcard.csv")
            if os.path.isfile(target):
                return target

    return None


def main():
    print("=" * 65)
    print("   LIGHTGBM BENCHMARK: CREDIT CARD FRAUD DETECTION (TASK 4.4)")
    print("=" * 65)

    data_path = find_dataset()
    if not data_path:
        print("\n[!] ERROR: creditcard.csv not found!")
        print("    Vui lòng tải dataset từ Kaggle bằng lệnh sau:")
        print("    cd ~/ml-benchmark")
        print("    kaggle datasets download -d mlg-ulb/creditcardfraud --unzip")
        sys.exit(1)

    print(f"[*] Dataset located at: {data_path}")

    # 1. Load Dataset & Train/Test Split
    print("\n[1/5] Loading dataset...")
    t_load_start = time.perf_counter()
    df = pd.read_csv(data_path)
    t_load_end = time.perf_counter()
    load_time_sec = t_load_end - t_load_start

    total_rows, total_cols = df.shape
    fraud_count = int(df["Class"].sum())
    normal_count = total_rows - fraud_count
    print(f"    - Total rows: {total_rows:,}, Columns: {total_cols}")
    print(f"    - Class distribution: Normal = {normal_count:,}, Fraud = {fraud_count:,} ({fraud_count/total_rows*100:.3f}%)")
    print(f"    - Load time: {load_time_sec:.4f} seconds")

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y = df["Class"]

    print("[*] Splitting dataset (80% Train, 20% Test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    - Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Model Training with LightGBM
    print("\n[2/5] Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    t_train_start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    t_train_end = time.perf_counter()
    train_time_sec = t_train_end - t_train_start

    best_iteration = int(model.best_iteration_) if hasattr(model, "best_iteration_") and model.best_iteration_ else model.n_estimators
    print(f"    - Training time: {train_time_sec:.4f} seconds")
    print(f"    - Best iteration: {best_iteration}")

    # 3. Model Evaluation on Test Set
    print("\n[3/5] Evaluating model on test set...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_roc = float(roc_auc_score(y_test, y_prob))
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))

    print(f"    - AUC-ROC  : {auc_roc:.4f}")
    print(f"    - Accuracy : {accuracy * 100:.2f}% ({accuracy:.4f})")
    print(f"    - F1-Score : {f1:.4f}")
    print(f"    - Precision: {precision:.4f}")
    print(f"    - Recall   : {recall:.4f}")

    # 4. Inference Benchmark (Latency & Throughput)
    print("\n[4/5] Benchmarking inference latency & throughput...")

    # Single-row latency (Warmup + 200 runs)
    sample_single = X_test.iloc[[0]]
    for _ in range(20):  # Warmup
        _ = model.predict(sample_single)

    single_times = []
    n_single_trials = 200
    for _ in range(n_single_trials):
        idx = np.random.randint(0, len(X_test))
        row = X_test.iloc[[idx]]
        t0 = time.perf_counter()
        _ = model.predict(row)
        t1 = time.perf_counter()
        single_times.append(t1 - t0)

    avg_single_latency_ms = float(np.mean(single_times) * 1000)
    p95_single_latency_ms = float(np.percentile(single_times, 95) * 1000)

    # 1000-row batch throughput (Warmup + 30 runs)
    batch_size = 1000
    sample_batch = X_test.iloc[:batch_size]
    for _ in range(5):  # Warmup
        _ = model.predict(sample_batch)

    batch_times = []
    n_batch_trials = 30
    for _ in range(n_batch_trials):
        start_idx = np.random.randint(0, len(X_test) - batch_size)
        batch = X_test.iloc[start_idx : start_idx + batch_size]
        t0 = time.perf_counter()
        _ = model.predict(batch)
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)

    avg_batch_time_sec = float(np.mean(batch_times))
    throughput_rows_per_sec = float(batch_size / avg_batch_time_sec)
    batch_latency_ms = float(avg_batch_time_sec * 1000)

    print(f"    - 1-row latency (avg): {avg_single_latency_ms:.3f} ms (p95: {p95_single_latency_ms:.3f} ms)")
    print(f"    - 1000-row batch time: {batch_latency_ms:.2f} ms")
    print(f"    - 1000-row throughput: {throughput_rows_per_sec:,.1f} rows/sec")

    # 5. Export JSON
    print("\n[5/5] Exporting results...")
    results = {
        "dataset": {
            "name": "creditcard.csv",
            "total_rows": total_rows,
            "total_cols": total_cols,
            "fraud_ratio_pct": round(fraud_count / total_rows * 100, 3)
        },
        "metrics": {
            "data_load_time_sec": round(load_time_sec, 4),
            "training_time_sec": round(train_time_sec, 4),
            "best_iteration": best_iteration,
            "auc_roc": round(auc_roc, 4),
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "inference_latency_1_row_ms": round(avg_single_latency_ms, 3),
            "inference_latency_1_row_p95_ms": round(p95_single_latency_ms, 3),
            "inference_throughput_1000_rows_per_sec": round(throughput_rows_per_sec, 1),
            "inference_batch_1000_latency_ms": round(batch_latency_ms, 2)
        },
        "system": {
            "python_version": sys.version.split()[0],
            "lightgbm_version": lgb.__version__
        }
    }

    output_dir = os.path.dirname(data_path)
    output_json_path = os.path.join(output_dir, "benchmark_result.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also save to current directory if different
    if os.path.abspath(output_json_path) != os.path.abspath("benchmark_result.json"):
        with open("benchmark_result.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"    - Results saved to: {output_json_path}")

    # Print summary table formatted exactly as requested in README
    print("\n" + "=" * 65)
    print("               BENCHMARK RESULT TABLE (COPY TO README)")
    print("=" * 65)
    print(f"| {'Metric':<32} | {'Kết quả':<26} |")
    print(f"|{'-' * 34}|{'-' * 28}|")
    print(f"| {'Thời gian load data':<32} | {load_time_sec:.4f} s{'':<18} |")
    print(f"| {'Thời gian training':<32} | {train_time_sec:.4f} s{'':<18} |")
    print(f"| {'Best iteration':<32} | {best_iteration:<26} |")
    print(f"| {'AUC-ROC':<32} | {auc_roc:.4f}{'':<20} |")
    print(f"| {'Accuracy':<32} | {accuracy * 100:.2f}% ({accuracy:.4f}){'':<9} |")
    print(f"| {'F1-Score':<32} | {f1:.4f}{'':<20} |")
    print(f"| {'Precision':<32} | {precision:.4f}{'':<20} |")
    print(f"| {'Recall':<32} | {recall:.4f}{'':<20} |")
    print(f"| {'Inference latency (1 row)':<32} | {avg_single_latency_ms:.3f} ms / row{'':<11} |")
    print(f"| {'Inference throughput (1000 rows)':<32} | {throughput_rows_per_sec:,.0f} rows/s ({batch_latency_ms:.1f}ms) |")
    print("=" * 65)
    print("\n[+] Benchmark completed successfully!")


if __name__ == "__main__":
    main()

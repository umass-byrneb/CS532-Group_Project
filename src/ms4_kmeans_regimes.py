#!/usr/bin/env python3
"""
MS4: K-Means market regimes (Arrow-safe version)

Key changes vs. prior version:
- Never calls toPandas(); uses collect()/pure-Python instead.
- Disables Arrow to avoid DirectByteBuffer / sun.misc.Unsafe issues on some JDKs.
- Derives regime labels from KMeans cluster centers directly (and, if standardized,
  inverts the scaling so labels are based on the original feature space).

CLI
----
python -m src.ms4_kmeans_regimes \
  --features_path artifacts/features_ml_daily \
  --out_dir artifacts/ms4 \
  --kmin 3 --kmax 8 --seed 42 --standardize true

Outputs
-------
- {out_dir}/metrics.json                : sweep metrics and best-k summary
- {out_dir}/centroids.csv               : cluster centers in original and (if scaled) scaled space
- {out_dir}/predictions.parquet         : per-day per-coin cluster + regime labels
- {out_dir}/models/k{K}                 : Spark KMeansModel for best K
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.linalg import DenseVector, VectorUDT

# Project utilities
from src.common import config as C
from src.common.spark import get_spark
from src.common.io_utils import ensure_dir, read_parquet, save_parquet

FEATURE_COLS = [
    "ret_1d_w",
    "zscore_7",
    "ret_std_7",
    "ret_std_30",
    "ma_gap_7",
    "ma_gap_30",
]


@dataclass
class SweepResult:
    k: int
    silhouette: float
    sizes: Dict[str, int]
    model_path: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True, help="Parquet dir of ML features (partitioned by symbol)")
    p.add_argument("--out_dir", required=True, help="Output directory for models/metrics/predictions")
    p.add_argument("--kmin", type=int, default=3)
    p.add_argument("--kmax", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--standardize", type=str, default="true", help="true/false: StandardScaler with mean+std")
    return p.parse_args()


def load_features(spark: SparkSession, path: str):
    df = read_parquet(spark, path)
    # Minimal schema check
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in {path}: {missing}")
    # quick banner
    span = df.agg(F.min("Date").alias("minD"), F.max("Date").alias("maxD")).first()
    syms = [r[0] for r in df.select("symbol").distinct().orderBy("symbol").collect()]
    print(
        f"MS4 input: rows={df.count():,}, symbols={len(syms)}, span={span['minD']} → {span['maxD']}\n"
        f"Symbols: {', '.join(syms)}"
    )
    # percentiles (optional quick sanity)
    for col in FEATURE_COLS:
        p = df.select(
            F.expr(f"percentile_approx({col}, array(0.01,0.50,0.99), 10000)").alias("p")
        ).first()[0]
        print(f"    {col:<10}| p01={p[0]}  p50={p[1]}  p99={p[2]}")
    return df


def assemble_features(df):
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    out = assembler.transform(df)
    return out, assembler


def make_scaler():
    # withMean True requires Dense vectors; VectorAssembler provides Dense by default
    return StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)


def invert_center(center: np.ndarray, scaler_model: StandardScaler | None):
    """Return center in original feature space if scaler_model is provided, else as-is.
    Spark's StandardScalerModel uses: scaled = (v - mean) / stdev when withMean & withStd.
    So inverse is: v = scaled * stdev + mean.
    """
    if scaler_model is None:
        return center
    sm = scaler_model
    # Accessing fitted model parameters: after fit, return type is StandardScalerModel
    # But we pass the fitted model in when available. We'll duck type: look for std/mean attributes
    std = np.array(sm.std.toArray()) if hasattr(sm, "std") else None
    mean = np.array(sm.mean.toArray()) if hasattr(sm, "mean") else None
    v = center.copy()
    if std is not None:
        v = v * std
    if mean is not None:
        v = v + mean
    return v


def describe_centers(model: KMeans, scaler_model: StandardScaler | None) -> List[dict]:
    """Build a list of dicts describing both scaled and (if applicable) original-space centers."""
    centers_scaled = [np.array(c) for c in model.clusterCenters()]
    rows = []
    for idx, c in enumerate(centers_scaled):
        orig = invert_center(c, scaler_model)
        row = {"cluster": idx}
        # attach per-feature values for both spaces
        for i, name in enumerate(FEATURE_COLS):
            row[f"center_scaled_{name}"] = float(c[i])
            row[f"center_orig_{name}"] = float(orig[i])
        rows.append(row)
    return rows


def derive_regime_labels_from_centers(centers_orig: List[np.ndarray]) -> Dict[int, str]:
    """Heuristic mapping from center vectors (original space) to regime names.

    Features order:
      0 ret_1d_w, 1 zscore_7, 2 ret_std_7, 3 ret_std_30, 4 ma_gap_7, 5 ma_gap_30

    Rough rules:
      - Blow-off rally / parabolic: high positive z and large +MA gap (esp. 30d)
      - Sell-off / drawdown: strong negative z and large -MA gap
      - Bull impulse: positive z, moderate +MA gap, volatility elevated but not extreme
      - Calm / normal: near-zero z & MA gaps, *lowest* volatility (ret_std_30) among leftovers
      - Choppy high-vol sideways: near-zero z & MA gaps, but higher volatility
    """
    z = np.array([c[1] for c in centers_orig])
    vol = np.array([c[3] for c in centers_orig])  # ret_std_30
    magap30 = np.array([c[5] for c in centers_orig])

    K = len(centers_orig)
    labels: Dict[int, str] = {}
    used = set()

    # 1) Blow-off
    blow_idx = None
    if K >= 5:
        blow_idx = int(np.argmax(magap30 + 0.75 * z))  # tilt toward positive z
    else:
        blow_idx = int(np.argmax(magap30))
    labels[blow_idx] = "Blow-off rally / parabolic"
    used.add(blow_idx)

    # 2) Sell-off
    sell_idx = int(np.argmin(magap30 + 0.75 * z))  # most negative
    if sell_idx in used:
        # fallback: pick most negative z if tie
        sell_idx = int(np.argmin(z))
    labels[sell_idx] = "Sell-off / drawdown"
    used.add(sell_idx)

    # 3) Bull impulse (positive tilt, not the blow-off)
    remaining = [i for i in range(K) if i not in used]
    if remaining:
        bull_scores = [(i, z[i] + 0.5 * magap30[i]) for i in remaining]
        bull_idx = max(bull_scores, key=lambda t: t[1])[0]
        labels[bull_idx] = "Bull impulse"
        used.add(bull_idx)

    # 4) Remaining: choose Calm (lowest vol), others -> Choppy
    remaining = [i for i in range(K) if i not in used]
    if remaining:
        calm_idx = min(remaining, key=lambda i: vol[i])
        labels[calm_idx] = "Calm / normal"
        used.add(calm_idx)

    remaining = [i for i in range(K) if i not in used]
    for i in remaining:
        labels[i] = "Choppy high-vol sideways"

    return labels


def fit_and_score(
    df_vec,
    feature_col: str,
    k: int,
    seed: int,
) -> Tuple[KMeans, object, float, Dict[str, int]]:
    kmeans = KMeans(k=k, seed=seed, featuresCol=feature_col, predictionCol="cluster")
    model = kmeans.fit(df_vec)
    pred = model.transform(df_vec)
    evaluator = ClusteringEvaluator(featuresCol=feature_col, predictionCol="cluster", metricName="silhouette", distanceMeasure="squaredEuclidean")
    sil = float(evaluator.evaluate(pred))
    sizes = {f"c{i}": int(r[1]) for i, r in enumerate(pred.groupBy("cluster").count().orderBy("cluster").collect())}
    return model, pred, sil, sizes


def main():
    args = parse_args()

    # Spark session (Arrow off to avoid JDK DirectBuffer issues)
    spark = get_spark("MS4_KMeans")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    ensure_dir(args.out_dir)

    base = load_features(spark, args.features_path).cache()

    # Assemble features
    df_vec, assembler = assemble_features(base)
    feature_col = "features"

    # Optional standardization
    scaler_model = None
    if str(args.standardize).lower() in {"1", "true", "yes", "y"}:
        scaler = make_scaler()
        scaler_model = scaler.fit(df_vec)
        df_vec = scaler_model.transform(df_vec)
        feature_col = "scaledFeatures"
        print("Using StandardScaler(withMean=True, withStd=True)")

    # Brief sanity (retention)
    before = df_vec.count()
    after = df_vec.select(feature_col).na.drop().count()
    print("\n=== ML dataset sanity ===")
    print(f"Rows before: {before:,}")
    print(f"Rows after : {after:,} ({after/before*100:.1f}% retained)")
    df_vec = df_vec.na.drop(subset=[feature_col]).cache()

    # K sweep
    results: List[SweepResult] = []
    best = None
    for k in range(args.kmin, args.kmax + 1):
        model, pred, sil, sizes = fit_and_score(df_vec, feature_col, k, args.seed)
        print(f"k={k}: silhouette={sil:.4f} | cluster sizes: {sizes}")
        # Save temporary models? We'll only save the best.
        r = SweepResult(k=k, silhouette=sil, sizes=sizes)
        results.append(r)
        if best is None or sil > best.silhouette:
            best = r
            best_model = model
            best_pred = pred

    assert best is not None
    print(f"\nBest k={best.k} (silhouette={best.silhouette:.4f})\n")

    # Describe centers (both scaled & original)
    centers_scaled = [np.array(c) for c in best_model.clusterCenters()]
    centers_orig = [invert_center(c, scaler_model) for c in centers_scaled]

    # Label regimes from original-space centers
    label_map = derive_regime_labels_from_centers(centers_orig)

    # Save model
    model_dir = os.path.join(args.out_dir, f"models/k{best.k}")
    ensure_dir(model_dir)
    # Overwrite if exists
    if os.path.exists(model_dir):
        import shutil
        shutil.rmtree(model_dir)
    best_model.save(model_dir)

    # Save centroids (CSV)
    cent_rows = []
    for idx, (sc, og) in enumerate(zip(centers_scaled, centers_orig)):
        row = {"cluster": idx, "label": label_map[idx]}
        for j, name in enumerate(FEATURE_COLS):
            row[f"scaled_{name}"] = float(sc[j])
            row[f"orig_{name}"] = float(og[j])
        cent_rows.append(row)

    centroids_df = spark.createDataFrame(cent_rows)
    centroids_csv = os.path.join(args.out_dir, "centroids.csv")
    (centroids_df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(centroids_csv))
    print(f"Saved centroids → {centroids_csv}")

    # Attach labels to predictions and save
    mapping_broadcast = spark.sparkContext.broadcast(label_map)
    map_udf = F.udf(lambda c: mapping_broadcast.value.get(int(c), "Unknown"))

    preds = (best_pred
             .withColumn("regime", map_udf(F.col("cluster")))
             .select("Date", "symbol", "cluster", "regime"))

    preds_out = os.path.join(args.out_dir, "predictions.parquet")
    ensure_dir(preds_out)
    save_parquet(preds, preds_out, partitionBy=["symbol"], mode=C.SPARK_WRITE_MODE)
    print(f"Saved predictions → {preds_out}")

    # Summaries per regime
    summary = (best_pred
               .withColumn("regime", map_udf(F.col("cluster")))
               .groupBy("regime").count().orderBy(F.desc("count")))
    summary.show(20, truncate=False)

    # Metrics JSON
    metrics = {
        "kmin": args.kmin,
        "kmax": args.kmax,
        "seed": args.seed,
        "standardize": str(args.standardize),
        "results": [
            {
                "k": r.k,
                "silhouette": r.silhouette,
                "sizes": r.sizes,
            }
            for r in results
        ],
        "best": {
            "k": best.k,
            "silhouette": best.silhouette,
            "sizes": best.sizes,
            "label_map": label_map,
            "model_dir": model_dir,
        },
        "feature_cols": FEATURE_COLS,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {os.path.join(args.out_dir, 'metrics.json')}")

    # Friendly peek: by symbol, first 10 rows
    (preds.orderBy("symbol", "Date").show(10, truncate=False))


if __name__ == "__main__":
    main()

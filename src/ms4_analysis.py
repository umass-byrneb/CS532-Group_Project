# Milestone 4B analysis
# Usage:
#   python -m src.ms4_analysis \
#     --predictions_path artifacts/ms4/predictions.parquet \
#     --features_ml_path artifacts/features_ml_daily \
#     --out_dir artifacts/ms4b \
#     --steps 1,2,3,4,5,6,7
import os, json, argparse, uuid, time
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from pyspark.sql import SparkSession, Window, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vector as MLVector

try:
    from pyspark.ml.functions import vector_to_array
    _HAVE_V2A = True
except Exception:
    _HAVE_V2A = False
    vec1_udf = F.udf(
        lambda v: float(v[1]) if isinstance(v, (list, tuple)) and len(v) > 1
        else (float(v.toArray()[1]) if isinstance(v, MLVector) and v.size > 1 else None),
        DoubleType(),
    )

def proba1_col(prob_col):
    """Return P(y=1) from a VectorUDT probability column."""
    return (vector_to_array(prob_col).getItem(1)) if _HAVE_V2A else vec1_udf(prob_col)

REGIME_ORDER = [
    "Calm / normal",
    "Choppy high-vol sideways",
    "Sell-off / drawdown",
    "Blow-off rally / parabolic",
    "Bull impulse",
]

NUMERIC_COLS_DEFAULT: List[str] = [
    "ret_1d_w",
    "zscore_7",
    "ret_std_7",
    "ret_std_30",
    "ma_gap_7",
    "ma_gap_30",
]

def _safe_get(spark, key, default=None):
    try:
        return spark.conf.get(key)
    except Exception:
        return default

class _StepTimer:
    def __init__(self, spark, args, step_num: int, step_name: str, extras: Optional[Dict[str, Any]] = None):
        self.spark, self.args = spark, args
        self.step_num, self.step_name = step_num, step_name
        self.extras, self.t0 = (extras or {}), None
        self.timings_path = os.path.join(args.out_dir, "_timings.jsonl")
        self.run_id = os.environ.get("MS5_RUN_ID") or os.environ.get("RUN_ID") or str(uuid.uuid4())

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self.extras

    def __exit__(self, exc_type, exc, tb):
        wall_s = (time.perf_counter() - self.t0) if self.t0 is not None else None
        sc = self.spark.sparkContext
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "step": self.step_num,
            "step_name": self.step_name,
            "wall_s": wall_s,
            "master": getattr(sc, "master", None),
            "app_id": getattr(sc, "applicationId", None),
            "serializer": _safe_get(self.spark, "spark.serializer"),
            "aqe": _safe_get(self.spark, "spark.sql.adaptive.enabled"),
            "shuffle_partitions": _safe_get(self.spark, "spark.sql.shuffle.partitions"),
            "auto_broadcast": _safe_get(self.spark, "spark.sql.autoBroadcastJoinThreshold"),
            "whole_stage": _safe_get(self.spark, "spark.sql.codegen.wholeStage"),
            "predictions_path": getattr(self.args, "predictions_path", None),
            "features_ml_path": getattr(self.args, "features_ml_path", None),
            "out_dir": self.args.out_dir,
            **self.extras,
        }
        os.makedirs(self.args.out_dir, exist_ok=True)
        with open(self.timings_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[TIMING] Step {self.step_num} '{self.step_name}' completed in {wall_s:.2f}s → {self.timings_path}")
        return False

def step_timer(spark, args, step_num: int, step_name: str, extras: Optional[Dict[str, Any]] = None):
    return _StepTimer(spark, args, step_num, step_name, extras)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_parquet_and_csv(df: DataFrame, base_path: str):
    df.coalesce(1).write.mode("overwrite").parquet(base_path)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(base_path + "_csv")

def daily_ret_stats(df: DataFrame, ret_col: str, group_cols: List[str], suffix: str = "") -> DataFrame:
    g = (
        df.groupBy(*group_cols)
          .agg(
              F.count(F.lit(1)).alias("n"),
              F.mean(F.col(ret_col)).alias(f"mean_ret{suffix}"),
              F.expr(f"percentile_approx({ret_col}, 0.5)").alias(f"median_ret{suffix}"),
              F.stddev(F.col(ret_col)).alias(f"std_ret{suffix}"),
              F.mean(F.when(F.col(ret_col) > 0, 1).otherwise(0)).alias(f"hit_rate{suffix}"),
          )
    )
    return g.withColumn(
        f"sharpe_d{suffix}",
        F.when(F.col(f"std_ret{suffix}") > 0, F.col(f"mean_ret{suffix}") / F.col(f"std_ret{suffix}")).otherwise(F.lit(None)),
    )

def print_header(title: str):
    print("\n" + title)
    print("-" * len(title))
# -------------------------
# Input & Validation
# -------------------------
def run_step1(spark, args, pred: DataFrame):
    with step_timer(spark, args, 1, "Step 1: Input & validation", extras={"rows": None, "symbols": None}) as meta:
        print("Final schema:")
        pred.printSchema()

        span = pred.agg(
            F.min("Date").alias("min_d"),
            F.max("Date").alias("max_d"),
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct("symbol").alias("n_syms"),
        ).first()
        rows, n_syms = span.rows, span.n_syms
        print(f"\nMS4B Step1 — input summary: rows={rows:,}, symbols={n_syms}, span={span.min_d} → {span.max_d}")
        syms = [r.symbol for r in pred.select("symbol").distinct().orderBy("symbol").collect()]
        print("Symbols: " + ", ".join(syms))

        print("\nCounts by regime:")
        pred.groupBy("regime").count().orderBy(F.desc("count")).show(100, False)

        print("\nEvent check — 2020-03-12:")
        (pred.where(F.col("Date") == "2020-03-12")
             .groupBy("regime").count().orderBy(F.desc("count")).show(100, False))

        print("\nPer-coin distribution (top 3 per coin):")
        wcnt = Window.partitionBy("symbol").orderBy(F.desc("count"))
        (pred.groupBy("symbol", "regime").count()
             .withColumn("rnk", F.row_number().over(wcnt))
             .where(F.col("rnk") <= 3)
             .orderBy("symbol", F.desc("count"))
             .select("symbol", "regime", "count")
             .show(200, False))

        with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
            json.dump(
                {
                    "rows": int(rows),
                    "symbols": syms,
                    "span": [str(span.min_d), str(span.max_d)],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        meta["rows"], meta["symbols"] = int(rows), int(n_syms)
# -------------------------
# Transition Stats
# -------------------------
def run_step2(spark, args, pred: DataFrame, total_rows_hint=None):
    with step_timer(spark, args, 2, "Step 2: Transition stats"):
        w = Window.partitionBy("symbol").orderBy("Date")
        pred2 = (
            pred.withColumn("regime_next", F.lead("regime").over(w))
                .withColumn("regime_prev", F.lag("regime").over(w))
                .withColumn("is_transition", (F.col("regime_prev").isNotNull()) & (F.col("regime_prev") != F.col("regime")))
                .withColumn("transition", F.concat_ws(" → ", F.col("regime_prev"), F.col("regime")))
        )

        print_header("Top transitions by count:")
        (pred2.where(F.col("regime_prev").isNotNull())
              .groupBy("regime", "regime_next").count()
              .orderBy(F.desc("count")).show(20, False))

        counts = (pred2.where(F.col("regime_prev").isNotNull())
                        .groupBy("regime", "regime_next").count())
        row_totals = counts.groupBy("regime").agg(F.sum("count").alias("row_count"))
        probs = counts.join(row_totals, "regime").withColumn("prob", F.col("count") / F.col("row_count"))

        mat_counts = counts.groupBy("regime").pivot("regime_next", REGIME_ORDER).agg(F.first("count")).fillna(0)
        mat_probs  = probs.groupBy("regime").pivot("regime_next", REGIME_ORDER).agg(F.first("prob")).fillna(0.0)

        print_header("Transition probability matrix (rows sum ≈ 1.0):")
        mat_probs.orderBy("regime").show(60, False)

        print_header("Row-sum check (should be ~1.0):")
        mat_probs.select("regime", sum([F.col(c) for c in mat_probs.columns if c != "regime"]).alias("row_sum")).show(60, False)

        stay = probs.where(F.col("regime") == F.col("regime_next")).select("regime", F.col("prob").alias("stay_prob"))
        print_header("Diagonal 'stay' probabilities:")
        stay.orderBy("regime").show(100, False)

        pred_runs = pred2.withColumn(
            "regime_group_id",
            F.sum(F.when(F.col("is_transition"), 1).otherwise(0)).over(w.rowsBetween(Window.unboundedPreceding, 0)),
        )
        runs = (pred_runs.groupBy("symbol", "regime", "regime_group_id")
                         .agg(F.count(F.lit(1)).alias("run_len")))

        dwell_overall = (
            runs.groupBy("regime")
                .agg(
                    F.count(F.lit(1)).alias("num_runs"),
                    F.mean("run_len").alias("mean_len"),
                    F.expr("percentile_approx(run_len, 0.5)").alias("p50_len"),
                    F.expr("percentile_approx(run_len, 0.9)").alias("p90_len"),
                    F.min("run_len").alias("min_len"),
                    F.max("run_len").alias("max_len"),
                )
                .orderBy("regime")
        )
        dwell_by_symbol = (
            runs.groupBy("symbol", "regime")
                .agg(
                    F.count(F.lit(1)).alias("num_runs"),
                    F.mean("run_len").alias("mean_len"),
                    F.expr("percentile_approx(run_len, 0.5)").alias("p50_len"),
                    F.expr("percentile_approx(run_len, 0.9)").alias("p90_len"),
                    F.max("run_len").alias("max_len"),
                )
                .orderBy("symbol", "regime")
        )

        print_header("Dwell-time stats (overall):")
        dwell_overall.show(100, False)

        print_header("Dwell-time per symbol (sample 50 rows):")
        dwell_by_symbol.show(50, False)

        total_run_len = runs.agg(F.sum("run_len")).first()[0]
        total_rows = total_rows_hint if total_rows_hint is not None else pred.count()
        print("\nsum equals total rows." if total_run_len == total_rows else f"\nWARNING — Dwell sum {total_run_len} != total rows {total_rows}.")

        save_parquet_and_csv(counts.orderBy(F.desc("count")), os.path.join(args.out_dir, "transitions_counts"))
        save_parquet_and_csv(probs.orderBy("regime", "regime_next"), os.path.join(args.out_dir, "transitions_prob"))
        save_parquet_and_csv(mat_counts.orderBy("regime"), os.path.join(args.out_dir, "matrix_counts"))
        save_parquet_and_csv(mat_probs.orderBy("regime"), os.path.join(args.out_dir, "matrix_prob"))
        save_parquet_and_csv(stay.orderBy("regime"), os.path.join(args.out_dir, "stay_prob"))
        save_parquet_and_csv(dwell_overall, os.path.join(args.out_dir, "dwell_overall"))
        save_parquet_and_csv(dwell_by_symbol, os.path.join(args.out_dir, "dwell_by_symbol"))

        with open(os.path.join(args.out_dir, "step2_manifest.json"), "w") as f:
            json.dump({"ok": True, "generated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)

def _join_features(spark: SparkSession, pred: DataFrame, features_ml_path: str) -> DataFrame:
    raw = spark.read.parquet(features_ml_path)
    cols = set(raw.columns)
    if "ret_1d_w" not in cols:
        raise ValueError("features_ml dataset must contain 'ret_1d_w'.")
    present_numeric = [c for c in NUMERIC_COLS_DEFAULT if c in cols]
    sel = ["Date", "symbol"] + list(sorted(set(["ret_1d_w"] + present_numeric)))
    raw2 = raw.select(*sel).withColumn("Date", F.to_date("Date"))
    return pred.join(raw2, on=["Date", "symbol"], how="inner")
# -------------------------
# Cluster/Regime returns:
# -------------------------
def run_step3(spark, args, pred: DataFrame, features_ml_path: str, out_dir: str, total_rows_hint=None):
    with step_timer(spark, args, 3, "Step 3: Regime returns"):
        df = _join_features(spark, pred, features_ml_path)
        rows_pred = total_rows_hint if total_rows_hint is not None else pred.count()
        join_rows = df.count()
        null_same = df.where(F.col("ret_1d_w").isNull()).count()
        print(f"\nMS4B Step3 — joined rows={join_rows:,} (pred rows={rows_pred:,}), null ret_1d_w={null_same}")

        ret_overall = daily_ret_stats(df, "ret_1d_w", ["regime"]).orderBy(F.desc("mean_ret"))
        print_header("Regime returns — overall (sorted by mean_ret desc):")
        ret_overall.show(100, False)

        ret_by_symbol = daily_ret_stats(df, "ret_1d_w", ["symbol", "regime"]).orderBy("regime", F.desc("mean_ret"))
        print_header("Regime returns — per symbol (top rows by regime):")
        ret_by_symbol.show(100, False)

        save_parquet_and_csv(ret_overall, os.path.join(out_dir, "returns_overall"))
        save_parquet_and_csv(ret_by_symbol, os.path.join(out_dir, "returns_by_symbol"))

        # Forward 1d return
        w2 = Window.partitionBy("symbol").orderBy("Date")
        df_fwd = df.withColumn("ret_fwd1", F.lead("ret_1d_w").over(w2)).where(F.col("ret_fwd1").isNotNull())
        print(f"\nMS4B Step3 (FORWARD) — joined rows={df_fwd.count():,} (pred rows={rows_pred:,}), null ret_fwd1={df_fwd.where(F.col('ret_fwd1').isNull()).count()}")

        ret_fwd_overall = daily_ret_stats(df_fwd, "ret_fwd1", ["regime"], suffix="_fwd1").orderBy(F.desc("mean_ret_fwd1"))
        print_header("Regime → next-day returns — overall (sorted by mean_ret_fwd1 desc):")
        ret_fwd_overall.show(100, False)

        ret_fwd_by_symbol = daily_ret_stats(df_fwd, "ret_fwd1", ["symbol", "regime"], suffix="_fwd1").orderBy("regime", F.desc("mean_ret_fwd1"))
        print_header("Regime → next-day returns — per symbol (top rows by regime):")
        ret_fwd_by_symbol.show(100, False)

        save_parquet_and_csv(ret_fwd_overall, os.path.join(out_dir, "returns_fwd1_overall"))
        save_parquet_and_csv(ret_fwd_by_symbol, os.path.join(out_dir, "returns_fwd1_by_symbol"))
# -------------------------
# Transition returns per cluster
# -------------------------
def run_step4(spark, args, pred: DataFrame, features_ml_path: str, out_dir: str):
    with step_timer(spark, args, 4, "Step 4 — Transition-conditioned returns"):
        df = _join_features(spark, pred, features_ml_path)
        w2 = Window.partitionBy("symbol").orderBy("Date")
        df_fwd = df.withColumn("ret_fwd1", F.lead("ret_1d_w").over(w2)).where(F.col("ret_fwd1").isNotNull())
        df_fwd_prev = (df_fwd
            .withColumn("regime_prev", F.lag("regime").over(w2))
            .withColumn("has_prev", F.col("regime_prev").isNotNull())
            .withColumn("is_transition", (F.col("has_prev")) & (F.col("regime_prev") != F.col("regime")))
            .withColumn("transition", F.concat_ws(" → ", F.col("regime_prev"), F.col("regime"))))

        rows_with_prev = df_fwd_prev.where("has_prev").count()
        rows_transition = df_fwd_prev.where("is_transition").count()
        print_header("STEP 4 — Transition-conditioned forward returns")
        print(f"MS4B Step4 — rows_with_prev={rows_with_prev:,}, transitions={rows_transition:,}, stays={rows_with_prev - rows_transition:,}")
        print(f"Null ret_fwd1 within transitions: {df_fwd_prev.where('is_transition AND ret_fwd1 IS NULL').count()}")

        print("\nTop transitions by frequency (change points only):")
        (df_fwd_prev.where("is_transition").groupBy("transition").count().orderBy(F.desc("count")).show(25, False))

        trans_overall = daily_ret_stats(df_fwd_prev.where("is_transition"), "ret_fwd1", ["transition"], suffix="_fwd1").orderBy(F.desc("mean_ret_fwd1"))
        print("\nTransition → next-day returns — overall (sorted by mean_ret_fwd1 desc):")
        trans_overall.show(40, False)

        trans_by_symbol = daily_ret_stats(df_fwd_prev.where("is_transition"), "ret_fwd1", ["symbol", "transition"], suffix="_fwd1").orderBy("transition", F.desc("mean_ret_fwd1"))
        stay_vs_change = daily_ret_stats(df_fwd_prev.where("has_prev"), "ret_fwd1", ["is_transition"], suffix="_fwd1").orderBy(F.desc("mean_ret_fwd1"))

        print("\nStay vs Change (is_transition=1 means regime changed today → t+1 return):")
        stay_vs_change.show(10, False)

        save_parquet_and_csv(trans_overall,   os.path.join(out_dir, "transitions_fwd1_overall"))
        save_parquet_and_csv(trans_by_symbol, os.path.join(out_dir, "transitions_fwd1_by_symbol"))
        save_parquet_and_csv(stay_vs_change,  os.path.join(out_dir, "transitions_fwd1_stay_vs_change"))
# -------------------------
# Cluster and BTC correlations
# -------------------------
def run_step5(spark, args, pred: DataFrame, features_ml_path: str, out_dir: str):
    with step_timer(spark, args, 5, "Step 5 — Breadth series & BTC correlations"):
        print_header("STEP 5 — Breadth series & BTC t+1 correlations")

        breadth_counts = pred.groupBy("Date", "regime").agg(F.countDistinct("symbol").alias("n_syms_regime"))
        total_per_date = pred.groupBy("Date").agg(F.countDistinct("symbol").alias("total_syms"))

        breadth_pivot = (breadth_counts.groupBy("Date")
            .pivot("regime", REGIME_ORDER)
            .agg(F.first("n_syms_regime")).fillna(0)
            .join(total_per_date, "Date", "inner"))

        breadth_frac = (breadth_pivot
            .withColumn("frac_calm",    F.col("Calm / normal") / F.col("total_syms"))
            .withColumn("frac_choppy",  F.col("Choppy high-vol sideways") / F.col("total_syms"))
            .withColumn("frac_sello",   F.col("Sell-off / drawdown") / F.col("total_syms"))
            .withColumn("frac_blowoff", F.col("Blow-off rally / parabolic") / F.col("total_syms"))
            .withColumn("frac_bull",    F.col("Bull impulse") / F.col("total_syms"))
            .orderBy("Date"))

        print("\nBreadth sample (first 12 rows):")
        breadth_frac.show(12, False)

        df = _join_features(spark, pred, features_ml_path)
        w2 = Window.partitionBy("symbol").orderBy("Date")
        df_fwd = df.withColumn("ret_fwd1", F.lead("ret_1d_w").over(w2)).where(F.col("ret_fwd1").isNotNull())
        btc_fwd = df_fwd.where(F.col("symbol") == "BTC").select("Date", F.col("ret_fwd1").alias("btc_fwd1"))
        breadth_with_btc = breadth_frac.join(btc_fwd, on="Date", how="inner")

        sum_check = breadth_with_btc.select(
            "Date",
            "total_syms",
            (F.col("Calm / normal") + F.col("Choppy high-vol sideways") + F.col("Sell-off / drawdown") + F.col("Blow-off rally / parabolic") + F.col("Bull impulse")).alias("sum_counts"),
        )
        bad_rows = sum_check.where(F.col("sum_counts") != F.col("total_syms")).count()
        print("VALIDATION OK — per-date regime counts sum to total symbols." if bad_rows == 0 else f"WARNING — {bad_rows} dates where regime-count sum != total_syms.")

        corr_items = []
        for nm in ["frac_blowoff", "frac_bull", "frac_calm", "frac_choppy", "frac_sello"]:
            val = breadth_with_btc.agg(F.corr(nm, "btc_fwd1").alias("c")).first()["c"]
            corr_items.append((nm, float(val) if val is not None else None))

        print("\nCorrelations: frac_{regime} vs BTC t+1 return:")
        for k, v in sorted(corr_items, key=lambda x: -abs(x[1] if x[1] is not None else 0.0)):
            print(f"  {k:12s}  corr = {v}")

        save_parquet_and_csv(breadth_frac, os.path.join(out_dir, "breadth_daily"))
        with open(os.path.join(out_dir, "breadth_btc_corr.json"), "w") as f:
            json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "corr_frac_vs_btc_fwd1": {k: v for k, v in corr_items}}, f, indent=2)
# -------------------------
# Cluster characterization
# -------------------------
def _available_numeric_cols(df: DataFrame, want: List[str]) -> List[str]:
    present = [c for c in want if c in set(df.columns)]
    if not present:
        raise ValueError(f"No expected numeric features found. Wanted one of: {want}")
    return present

def run_step6(spark, args, pred: DataFrame, features_ml_path: str, out_dir: str):
    """Cluster characterization."""
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pyspark.sql import types as T
    from math import sqrt

    with step_timer(spark, args, 6, "Step 6 — Cluster characterization (centroids + silhouette)"):
        print_header("Step 6 — Cluster characterization")

        feats_raw = spark.read.parquet(features_ml_path).withColumn("Date", F.to_date("Date"))
        NUM_FEATS = _available_numeric_cols(feats_raw, NUMERIC_COLS_DEFAULT)
        feats = feats_raw.select("Date", "symbol", *NUM_FEATS)
        df = pred.join(feats, on=["Date", "symbol"], how="inner").dropna(subset=NUM_FEATS)

        print(f"Numeric features used ({len(NUM_FEATS)}): {NUM_FEATS}")

        cluster_centroids = (df.groupBy("cluster").agg(*[F.avg(c).alias(f"avg_{c}") for c in NUM_FEATS], F.count(F.lit(1)).alias("n")).orderBy("cluster"))
        cluster_centroids.show(20, truncate=False)
        save_parquet_and_csv(cluster_centroids, os.path.join(out_dir, "step6_cluster_centroids"))

        regime_centroids = (df.groupBy("regime").agg(*[F.avg(c).alias(f"avg_{c}") for c in NUM_FEATS], F.count(F.lit(1)).alias("n")).orderBy("regime"))
        regime_centroids.show(20, truncate=False)
        save_parquet_and_csv(regime_centroids, os.path.join(out_dir, "step6_regime_centroids"))

        assembler = VectorAssembler(inputCols=NUM_FEATS, outputCol="features")
        assembled = assembler.transform(df).select("features")

        k = 5
        kmodel = KMeans(k=k, seed=42, featuresCol="features", predictionCol="kpred").fit(assembled)
        kpred = kmodel.transform(assembled)

        sil_eval = ClusteringEvaluator(featuresCol="features", predictionCol="kpred", metricName="silhouette")
        silhouette = sil_eval.evaluate(kpred)
        print(f"Silhouette (KMeans overall): {silhouette:.3f}")

        centers = []
        for v in kmodel.clusterCenters():
            if hasattr(v, "toArray"): centers.append(list(v.toArray()))
            elif hasattr(v, "tolist"): centers.append(v.tolist())
            else: centers.append(list(v))
        cent_rows = [{"k": i, **{fname: float(vec[j]) for j, fname in enumerate(NUM_FEATS)}} for i, vec in enumerate(centers)]
        save_parquet_and_csv(spark.createDataFrame(cent_rows), os.path.join(out_dir, "step6_kmeans_centroids"))

        bc_centers = centers
        @F.udf(returnType=T.DoubleType())
        def _approx_silhouette(features, kpred):
            x = list(features) if not hasattr(features, "toArray") else list(features.toArray())
            def _dist(u, v):
                return sqrt(sum((uu - vv) ** 2 for uu, vv in zip(u, v)))
            a = _dist(x, bc_centers[int(kpred)])
            b = min(_dist(x, c) for idx, c in enumerate(bc_centers) if idx != int(kpred)) if len(bc_centers) > 1 else 0.0
            denom = max(a, b)
            return float((b - a) / denom) if denom > 0 else 0.0

        kpred_s = kpred.withColumn("silhouette", _approx_silhouette(F.col("features"), F.col("kpred")))
        per_cluster_s = (kpred_s.groupBy("kpred").agg(F.count(F.lit(1)).alias("n"), F.avg("silhouette").alias("mean_silhouette")).orderBy("kpred"))
        print_header("Per-cluster silhouette (approx via centroids)")
        per_cluster_s.show(20, truncate=False)
        save_parquet_and_csv(per_cluster_s, os.path.join(out_dir, "step6_per_cluster_silhouette"))

        with open(os.path.join(out_dir, "step6_manifest.json"), "w") as f:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "num_features": len(NUM_FEATS),
                "features": NUM_FEATS,
                "kmeans_k": k,
                "silhouette_overall": float(silhouette),
            }, f, indent=2)
# -------------------------
# Predict sign
# -------------------------
def run_step7(spark, args, pred: DataFrame, features_ml_path: str, out_dir: str):
    """Predict sign(ret_fwd1) with LR, random/chron split, metrics, threshold sweep, and artifacts."""
    with step_timer(spark, args, 7, "Step 7 — Predictive gauge"):
        base = _join_features(spark, pred, features_ml_path)
        NUM_FEATS = [c for c in NUMERIC_COLS_DEFAULT if c in base.columns]
        if "ret_1d_w" not in base.columns:
            raise ValueError("ret_1d_w missing after join; cannot compute forward label.")

        w = Window.partitionBy("symbol").orderBy("Date")
        df = (base.withColumn("ret_fwd1", F.lead("ret_1d_w").over(w))
                    .where(F.col("ret_fwd1").isNotNull())
                    .dropna(subset=NUM_FEATS)
                    .withColumn("label", (F.col("ret_fwd1") > 0).cast("double")))

        assembler = VectorAssembler(inputCols=NUM_FEATS, outputCol="features")
        df_for_split = df.select("Date", "label", *NUM_FEATS)

        if args.split_mode == "chron":
            if args.split_date:
                cut_date = F.to_date(F.lit(args.split_date))
            else:
                df_ts = df_for_split.withColumn("ts", F.col("Date").cast("timestamp").cast("long"))
                cut_ts = df_ts.approxQuantile("ts", [1.0 - float(args.test_frac)], 0.001)[0]
                cut_date_val = df_ts.where(F.col("ts") <= F.lit(cut_ts)).agg(F.max("Date").alias("d")).first()["d"]
                cut_date = F.lit(cut_date_val)
            train_df = df_for_split.where(F.col("Date") <= cut_date)
            test_df  = df_for_split.where(F.col("Date") >  cut_date)
            train = assembler.transform(train_df).select("features", "label")
            test  = assembler.transform(test_df ).select("features", "label")
        else:
            ds = assembler.transform(df_for_split).select("features", "label")
            train, test = ds.randomSplit([1.0 - float(args.test_frac), float(args.test_frac)], seed=42)

        weight_col = None
        if args.balance_classes:
            pos = float(train.agg(F.avg("label")).first()[0] or 0.5)
            w_pos, w_neg = 0.5 / max(pos, 1e-12), 0.5 / max(1.0 - pos, 1e-12)
            train = train.withColumn("w", F.when(F.col("label") == 1.0, F.lit(w_pos)).otherwise(F.lit(w_neg)))
            weight_col = "w"

        n_tr, n_te = train.count(), test.count()
        pos_rate = test.agg(F.avg("label")).first()[0] if n_te > 0 else None
        print(f"Predictive gauge — train={n_tr:,}, test={n_te:,}, test positive-rate={pos_rate:.3f}" if pos_rate is not None else f"Predictive gauge — train={n_tr:,}, test={n_te:,}")

        lr = LogisticRegression(
            featuresCol="features", labelCol="label",
            predictionCol="prediction", probabilityCol="probability", rawPredictionCol="rawPrediction",
            maxIter=100, regParam=0.0, elasticNetParam=0.0,
        )
        if weight_col:
            lr = lr.setWeightCol(weight_col)

        model = lr.fit(train)
        pred_te = model.transform(test)
        pred_probs = pred_te.withColumn("proba1", proba1_col(F.col("probability")))

        brier = pred_probs.select(F.mean((F.col("proba1") - F.col("label")) ** 2).alias("brier")).first()["brier"]
        prob_q10, prob_q50, prob_q90 = [float(x) for x in pred_probs.approxQuantile("proba1", [0.1, 0.5, 0.9], 0.001)]

        pos = float(pos_rate) if pos_rate is not None else 0.5
        acc_majority = max(pos, 1.0 - pos)
        f1_all_pos = (2 * pos) / (1 + pos) if pos > 0 else 0.0
        f1_all_neg = 0.0
        f1_trivial_best = max(f1_all_pos, f1_all_neg)
        print(f"Baselines — Majority ACC={acc_majority:.3f}, Always-Positive F1={f1_all_pos:.3f}, Brier={brier:.3f}, proba1 q10/50/90=({prob_q10:.3f}, {prob_q50:.3f}, {prob_q90:.3f})")

        roc_auc = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC").evaluate(pred_te)
        pr_auc  = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderPR").evaluate(pred_te)
        cm_default = pred_te.groupBy("label", "prediction").count().orderBy("label", "prediction")
        cm_default.show(20, truncate=False)
        save_parquet_and_csv(cm_default, os.path.join(out_dir, "gauge_confusion_default"))

        cm = {(int(r["label"]), int(r["prediction"])): int(r["count"]) for r in cm_default.collect()}
        TP, FP, TN, FN = cm.get((1, 1), 0), cm.get((0, 1), 0), cm.get((0, 0), 0), cm.get((1, 0), 0)
        precision0 = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        recall0    = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        acc0       = (TP + TN) / max(1, (TP + TN + FP + FN))
        f10        = (2 * precision0 * recall0 / (precision0 + recall0)) if (precision0 + recall0) > 0 else 0.0
        print(f"Predictive gauge metrics — ROC AUC={roc_auc:.3f}, PR AUC={pr_auc:.3f}, ACC={acc0:.3f}, P={precision0:.3f}, R={recall0:.3f}, F1={f10:.3f}")

        coef_df = spark.createDataFrame([{ "feature": f, "weight": float(w)} for f, w in zip(NUM_FEATS, list(model.coefficients))])
        coef_df.orderBy(F.abs("weight").desc()).show(len(NUM_FEATS), truncate=False)
        save_parquet_and_csv(coef_df, os.path.join(out_dir, "gauge_coefficients"))

        pred_probs = pred_te.withColumn("proba1", proba1_col(F.col("probability")))
        sweep_thresholds = [round(x, 2) for x in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]]

        def _metrics_for_threshold(thr: float):
            pr = pred_probs.withColumn("pred_thr", (F.col("proba1") >= F.lit(thr)).cast("int"))
            ag = pr.agg(
                F.sum(F.when((F.col("label") == 1) & (F.col("pred_thr") == 1), 1).otherwise(0)).alias("TP"),
                F.sum(F.when((F.col("label") == 0) & (F.col("pred_thr") == 1), 1).otherwise(0)).alias("FP"),
                F.sum(F.when((F.col("label") == 0) & (F.col("pred_thr") == 0), 1).otherwise(0)).alias("TN"),
                F.sum(F.when((F.col("label") == 1) & (F.col("pred_thr") == 0), 1).otherwise(0)).alias("FN"),
            ).first()
            TP_, FP_, TN_, FN_ = int(ag.TP), int(ag.FP), int(ag.TN), int(ag.FN)
            precision = (TP_ / (TP_ + FP_)) if (TP_ + FP_) > 0 else 0.0
            recall    = (TP_ / (TP_ + FN_)) if (TP_ + FN_) > 0 else 0.0
            acc       = (TP_ + TN_) / max(1, (TP_ + TN_ + FP_ + FN_))
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            return {"threshold": float(thr), "TP": TP_, "FP": FP_, "TN": TN_, "FN": FN_, "precision": float(precision), "recall": float(recall), "f1": float(f1), "accuracy": float(acc)}

        results = [_metrics_for_threshold(t) for t in sweep_thresholds]
        sweep_df = spark.createDataFrame(results)
        sweep_df.orderBy(F.col("threshold")).show(20, truncate=False)
        save_parquet_and_csv(sweep_df, os.path.join(out_dir, "gauge_threshold_sweep"))

        best = max(results, key=lambda r: r["f1"]) if results else {"threshold": 0.5, "f1": f10, "precision": precision0, "recall": recall0, "accuracy": acc0, "TP": TP, "FP": FP, "TN": TN, "FN": FN}
        print(f"Best F1 @ threshold={best['threshold']:.2f} → F1={best['f1']:.3f}, P={best['precision']:.3f}, R={best['recall']:.3f}, ACC={best['accuracy']:.3f}")

        best_cm = spark.createDataFrame([(0, 0, best.get("TN", 0)), (0, 1, best.get("FP", 0)), (1, 0, best.get("FN", 0)), (1, 1, best.get("TP", 0))], ["label", "prediction", "count"])
        save_parquet_and_csv(best_cm, os.path.join(out_dir, "gauge_confusion_best"))

        gauge_metrics = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "num_features": len(NUM_FEATS),
            "features": NUM_FEATS,
            "train_rows": int(n_tr),
            "test_rows": int(n_te),
            "test_positive_rate": float(pos_rate) if pos_rate is not None else None,
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "default_threshold": 0.5,
            "acc_at_default": float(acc0),
            "precision_at_default": float(precision0),
            "recall_at_default": float(recall0),
            "f1_at_default": float(f10),
            "best_threshold": float(best["threshold"]),
            "best_f1": float(best["f1"]),
            "best_precision": float(best["precision"]),
            "best_recall": float(best["recall"]),
            "best_accuracy": float(best["accuracy"]),
            "best_confusion": {k: int(v) for k, v in best.items() if k in ("TP", "FP", "TN", "FN")},
            "threshold_sweep": results,
            "coefficients": [{"feature": f, "weight": float(w)} for f, w in zip(NUM_FEATS, list(model.coefficients))],
            "brier": float(brier),
            "prob_quantiles": {"q10": prob_q10, "q50": prob_q50, "q90": prob_q90},
            "baselines": {"acc_majority": float(acc_majority), "f1_always_positive": float(f1_all_pos), "f1_always_negative": float(f1_all_neg), "f1_trivial_best": float(f1_trivial_best)},
        }
        with open(os.path.join(out_dir, "gauge_metrics.json"), "w") as f:
            json.dump(gauge_metrics, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True, help="Parquet path for predictions [Date,symbol,cluster,regime]")
    parser.add_argument("--features_ml_path", required=False, default=None, help="Parquet path with features incl. ret_1d_w and (optionally) zscore/vol/ma_gap")
    parser.add_argument("--out_dir", required=True, help="Output directory (will be created)")
    parser.add_argument("--steps", required=True, help="Comma-separated steps (e.g. 1,2,3 or 4,5 or all)")
    parser.add_argument("--split_mode", choices=["random", "chron"], default="random", help="random (default) or chronological split by Date")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Test fraction for random/chron quantile split (default 0.2)")
    parser.add_argument("--split_date", type=str, default=None, help="YYYY-MM-DD; if set with --split_mode=chron, trains on <= date, tests on > date")
    parser.add_argument("--balance_classes", action="store_true", help="If set, use class weights (inverse frequency) via weightCol")
    args = parser.parse_args()

    steps = [1,2,3,4,5,6,7] if args.steps.strip().lower() == "all" else [int(s.strip()) for s in args.steps.split(",") if s.strip()]

    ensure_dir(args.out_dir)
    spark = SparkSession.builder.appName("ms4_analysis").getOrCreate()

    pred = (spark.read.parquet(args.predictions_path)
                 .select("Date", "symbol", "cluster", "regime")
                 .withColumn("Date", F.to_date("Date")))

    if not (os.environ.get("MS4_DISABLE_CACHE", "").lower() in ("1", "true", "yes")):
        pred = pred.cache()
    else:
        print("Note: caching disabled via MS4_DISABLE_CACHE.")

    total_rows = pred.count()

    if 1 in steps: run_step1(spark, args, pred)
    if 2 in steps: run_step2(spark, args, pred, total_rows_hint=total_rows)

    needs_features = any(s in steps for s in (3,4,5,6,7))
    if needs_features and not args.features_ml_path:
        raise ValueError("--features_ml_path is required for steps 3–7.")

    if 3 in steps: run_step3(spark, args, pred, args.features_ml_path, args.out_dir, total_rows_hint=total_rows)
    if 4 in steps: run_step4(spark, args, pred, args.features_ml_path, args.out_dir)
    if 5 in steps: run_step5(spark, args, pred, args.features_ml_path, args.out_dir)
    if 6 in steps: run_step6(spark, args, pred, args.features_ml_path, args.out_dir)
    if 7 in steps: run_step7(spark, args, pred, args.features_ml_path, args.out_dir)

    spark.stop()
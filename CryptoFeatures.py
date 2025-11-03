# Milestone 2 and 3: Data Cleaning and Volitility Analysis
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F, Window as W

DATA_DIR           = "data"            
EXCLUDE_SYMBOLS    = {"USDT", "USDC"}  
USE_LOG_RET        = True              # True = log returns; False = simple pct returns
WINSOR_LO          = 0.02              
WINSOR_HI          = 0.98             
SIGMA_FLOOR        = 1e-4              
MIN_PRIOR_OBS_7    = 7                 # must be <= 7, otherwise z_7 is always NULL
MIN_PRIOR_OBS_30   = 10                
BURN_IN_DAYS       = 14                
TOPK               = 10                # rows to show in top-|z|

spark = (
    SparkSession.builder
    .appName("CryptoFeatures")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# =========================
# 1) Load and clean
# =========================
raw = (
    spark.read
         .option("header", "true")
         .option("inferSchema", "false")
         .option("recursiveFileLookup", "true")
         .option("pathGlobFilter", "*.csv")
         .csv(DATA_DIR)
         .withColumn("file_name", F.input_file_name())
)

symbol_from_file = F.upper(
    F.regexp_extract(
        F.regexp_extract("file_name", r"([^/]+)$", 1),
        r"coin_([^\.]+)\.csv",
        1
    )
)

df = (
    raw
    .withColumnRenamed("SNo", "sno")
    .withColumnRenamed("Name", "name")
    .withColumnRenamed("Symbol", "Symbol")
    .withColumnRenamed("Date", "Date")
    .withColumnRenamed("High", "High")
    .withColumnRenamed("Low", "Low")
    .withColumnRenamed("Open", "Open")
    .withColumnRenamed("Close", "Close")
    .withColumnRenamed("Volume", "Volume")
    .withColumnRenamed("Marketcap", "Marketcap")
    .withColumn("symbol", F.upper(F.coalesce(F.col("Symbol"), symbol_from_file)))
)

df = df.filter(~F.col("symbol").isin([s.upper() for s in EXCLUDE_SYMBOLS]))

def clean_num(colname):
    return F.regexp_replace(F.col(colname), r"[^0-9.\-]", "").cast("double")

df = (
    df
    .withColumn("Open",      clean_num("Open"))
    .withColumn("High",      clean_num("High"))
    .withColumn("Low",       clean_num("Low"))
    .withColumn("Close",     clean_num("Close"))
    .withColumn("Volume",    clean_num("Volume"))
    .withColumn("Marketcap", clean_num("Marketcap"))
)


df = df.withColumn("Date", F.to_date(F.substring(F.col("Date").cast("string"), 1, 10), "yyyy-MM-dd"))
df = df.dropna(subset=["symbol", "Date", "Close"])
df = df.select("Date", "symbol", "Open", "High", "Low", "Close", "Volume", "Marketcap")

# =========================
# 2) Basic stats
# =========================
total_rows = df.count()
syms       = [r["symbol"] for r in df.select("symbol").distinct().orderBy("symbol").collect()]
date_span  = df.select(F.min("Date").alias("minD"), F.max("Date").alias("maxD")).first()

print(f"\n✅ Feature rows: {total_rows}")
print(f"\nSymbols ({len(syms)}): {', '.join(syms)}")
print(f"Date range: {date_span['minD']} → {date_span['maxD']}\n")

# =========================
# 3) Rolling features
# =========================
w = Window.partitionBy("symbol").orderBy("Date")
ma7  = F.avg("Close").over(w.rowsBetween(-6, 0))
ma30 = F.avg("Close").over(w.rowsBetween(-29, 0))
close_prev = F.lag("Close", 1).over(w)

if USE_LOG_RET:
    ret_raw = F.when((F.col("Close") > 0) & (close_prev > 0),
                     F.log(F.col("Close")) - F.log(close_prev))
else:
    ret_raw = (F.col("Close") / close_prev) - F.lit(1.0)

feat = (
    df
    .withColumn("ma_close_7",  ma7)
    .withColumn("ma_close_30", ma30)
    .withColumn("ret_1d",      ret_raw)
)

# =========================
# 4) Per-coin winsorization of returns
# =========================
qs = (
    feat.where(F.col("ret_1d").isNotNull())
        .groupBy("symbol")
        .agg(
            F.expr(f"percentile_approx(ret_1d, {WINSOR_LO}, 10000)").alias("q_lo"),
            F.expr(f"percentile_approx(ret_1d, {WINSOR_HI}, 10000)").alias("q_hi"),
        )
)

feat = (
    feat.join(qs, on="symbol", how="left")
        .withColumn(
            "ret_1d_w",
            F.when(F.col("ret_1d") < F.col("q_lo"), F.col("q_lo"))
             .when(F.col("ret_1d") > F.col("q_hi"), F.col("q_hi"))
             .otherwise(F.col("ret_1d"))
        )
)

# =========================
# 5) Rolling stats for z-scores 
# =========================
w_prev7  = w.rowsBetween(-7,  -1)    
w_prev30 = w.rowsBetween(-30, -1)    

ret_for_stats = F.col("ret_1d_w")

feat = (
    feat
    .withColumn("ret_mean_7",  F.avg(ret_for_stats).over(w_prev7))
    .withColumn("ret_std_7",   F.stddev_samp(ret_for_stats).over(w_prev7))
    .withColumn("cnt_7",       F.count(ret_for_stats).over(w_prev7))
    .withColumn("ret_mean_30", F.avg(ret_for_stats).over(w_prev30))
    .withColumn("ret_std_30",  F.stddev_samp(ret_for_stats).over(w_prev30))
    .withColumn("cnt_30",      F.count(ret_for_stats).over(w_prev30))
    .withColumn("rn",          F.row_number().over(w))  # for burn-in gating
)

z7 = F.when(
        (F.col("cnt_7") >= F.lit(MIN_PRIOR_OBS_7)) &
        (F.col("ret_std_7") >= F.lit(SIGMA_FLOOR)) &
        (F.col("rn") >= F.lit(BURN_IN_DAYS)),
        (F.col("ret_1d_w") - F.col("ret_mean_7")) / F.col("ret_std_7")
     )

z30 = F.when(
        (F.col("cnt_30") >= F.lit(MIN_PRIOR_OBS_30)) &
        (F.col("ret_std_30") >= F.lit(SIGMA_FLOOR)) &
        (F.col("rn") >= F.lit(BURN_IN_DAYS)),
        (F.col("ret_1d_w") - F.col("ret_mean_30")) / F.col("ret_std_30")
      )

feat = feat.withColumn("zscore_7", z7).withColumn("zscore_30", z30)

# =========================
# 6) anomaly flag +  % returns
# =========================
ret_pct = F.when(F.lit(USE_LOG_RET), F.exp(F.col("ret_1d")) - F.lit(1.0)).otherwise(F.col("ret_1d"))
feat = (
    feat
    .withColumn("ret_pct", ret_pct)                                   
    .withColumn("is_anom", F.when(F.abs(F.col("zscore_7")) >= 3.0, 1).otherwise(0))
)

# =========================
# 7) Top-N + Anomaly counts
# =========================
print("\nSample (by symbol then date):")
(
    feat.select(
        "Date","symbol","Close",
        "ma_close_7","ma_close_30",
        "ret_1d","ret_1d_w",
        "ret_mean_7","ret_std_7","zscore_7"
    )
    .orderBy("symbol","Date")
    .show(10, truncate=False)
)

print(f"\nTop {TOPK} absolute z-score (7-day window) across all coins (valid z only):")
(
    feat.select("Date","symbol","Close","ret_1d","ret_1d_w","ret_mean_7","ret_std_7","zscore_7")
        .where(F.col("zscore_7").isNotNull())
        .withColumn("abs_z", F.abs(F.col("zscore_7")))
        .orderBy(F.desc("abs_z"))
        .select("Date","symbol","Close","ret_1d","ret_1d_w","ret_mean_7","ret_std_7","zscore_7")
        .show(TOPK, truncate=False)
)

print("\nAnomaly counts per coin (|z|>=3.0 on 7-day window):")
(
    feat.select("symbol", F.when(F.abs(F.col("zscore_7")) >= 3.0, 1).otherwise(0).alias("anom"))
        .groupBy("symbol")
        .agg(F.sum("anom").alias("anom_7_count"))
        .orderBy("symbol")
        .show(100, truncate=False)
)

# =========================
# 8) Top jumps/drops per coin + volatility-over-time
# =========================
# Top 5 biggest jumps and drops per coin 
w_rank_up = W.partitionBy("symbol").orderBy(F.desc("ret_pct"))
w_rank_dn = W.partitionBy("symbol").orderBy(F.asc("ret_pct"))

top5_up = (
    feat.where(F.col("ret_pct").isNotNull())
        .withColumn("rk", F.row_number().over(w_rank_up))
        .where(F.col("rk") <= 5)
        .select("symbol", "Date", "ret_pct", "zscore_7")
        .orderBy("symbol", "rk")
)

top5_down = (
    feat.where(F.col("ret_pct").isNotNull())
        .withColumn("rk", F.row_number().over(w_rank_dn))
        .where(F.col("rk") <= 5)
        .select("symbol", "Date", "ret_pct", "zscore_7")
        .orderBy("symbol", "rk")
)

print("\nTop 5 jumps per coin (by simple % return):")
top5_up.show(100, truncate=False)

print("\nTop 5 drops per coin (by simple % return):")
top5_down.show(100, truncate=False)

# Volatility over time
print("\nSample volatility-over-time (30-day rolling std) per coin:")
(
    feat.select("Date", "symbol", "ret_std_30")
        .orderBy("symbol", "Date")
        .show(20, truncate=False)
)

(
    feat.where(F.col("is_anom") == 1)
        .select("Date", "symbol", "ret_pct", "zscore_7")
        .orderBy(F.desc(F.abs("zscore_7")))
        .write.mode("overwrite").option("header", True).csv("out/anomalies_by_coin")
)
top5_up.write.mode("overwrite").option("header", True).csv("out/top5_up_by_coin")
top5_down.write.mode("overwrite").option("header", True).csv("out/top5_down_by_coin")

# =========================
# 9) Build ML-ready daily dataset for clustering
# =========================
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np

feat_ml = (
    feat
    .withColumn("ma_gap_7",  (F.col("Close") / F.col("ma_close_7"))  - 1)
    .withColumn("ma_gap_30", (F.col("Close") / F.col("ma_close_30")) - 1)
)

ml_cols = ["ret_1d_w", "zscore_7", "ret_std_7", "ret_std_30", "ma_gap_7", "ma_gap_30"]

ml_df = (
    feat_ml
    .where(
        F.col("ret_1d_w").isNotNull() &
        F.col("zscore_7").isNotNull() &
        F.col("ret_std_7").isNotNull()  & (F.col("ret_std_7")  >= F.lit(SIGMA_FLOOR)) &
        F.col("ret_std_30").isNotNull() & (F.col("ret_std_30") >= F.lit(SIGMA_FLOOR)) &
        F.col("ma_gap_7").isNotNull() &
        F.col("ma_gap_30").isNotNull()
    )
    .select("Date", "symbol", *ml_cols)
    .cache()
)

print("\n=== Step 1: ML dataset (clustering) sanity checks ===")
total_feat = feat.count()
total_ml   = ml_df.count()
print(f"Rows before ML filters: {total_feat:,}")
print(f"Rows after  ML filters: {total_ml:,}  ({total_ml/total_feat:.1%} retained)")

for c in ml_cols:
    nulls = ml_df.where(F.col(c).isNull()).count()
    print(f"NULLs in {c}: {nulls}")

def pct(df, col):
    q = df.approxQuantile(col, [0.01, 0.50, 0.99], 0.001)
    return q

print("\nPercentiles (p01, p50, p99) per feature:")
for c in ml_cols:
    q01, q50, q99 = pct(ml_df, c)
    print(f"{c:>12s} | p01={q01:.6g}  p50={q50:.6g}  p99={q99:.6g}")

print("\nSample ML rows:")
ml_df.orderBy(F.rand()).show(10, truncate=False)

# =========================
# 10) K-Means regime clustering 
# =========================
ml_cols = ["ret_1d_w", "zscore_7", "ret_std_7", "ret_std_30", "ma_gap_7", "ma_gap_30"]
assembler = VectorAssembler(inputCols=ml_cols, outputCol="features")
assembled = assembler.transform(ml_df)

scaler = StandardScaler(inputCol="features", outputCol="features_scaled", withStd=True, withMean=True)
scaler_model = scaler.fit(assembled)
ml_scaled = scaler_model.transform(assembled).cache()

evaluator = ClusteringEvaluator(featuresCol="features_scaled",
                                predictionCol="prediction",
                                metricName="silhouette",
                                distanceMeasure="squaredEuclidean")

silhouettes = []
models = {}

print("\n=== Step 2: KMeans search over k ===")
for k in range(3, 9):
    km = KMeans(k=k, featuresCol="features_scaled", seed=42, maxIter=100, tol=1e-4)
    model = km.fit(ml_scaled)
    pred = model.transform(ml_scaled).cache()
    sil = evaluator.evaluate(pred)
    silhouettes.append((k, sil))
    models[k] = (model, pred)

    print(f"\nk={k}: silhouette={sil:.4f}  | cluster sizes:")
    pred.groupBy("prediction").count().orderBy("prediction").show()

best_k, best_sil = sorted(silhouettes, key=lambda x: x[1], reverse=True)[0]
best_model, best_pred = models[best_k]
print(f"\nBest k={best_k} (silhouette={best_sil:.4f})")

std = np.array(scaler_model.std.toArray()) if hasattr(scaler_model, "std") else np.array(scaler_model.stdDev.toArray())
mean = np.array(scaler_model.mean.toArray()) if scaler_model.getWithMean() else np.zeros_like(std)

centers_scaled = np.array(best_model.clusterCenters())
centers_orig = centers_scaled * std + mean

centers_rows = []
for idx, center in enumerate(centers_orig):
    row = {"prediction": idx}
    row.update({col: float(val) for col, val in zip(ml_cols, center)})
    centers_rows.append(row)
centers_df = spark.createDataFrame(centers_rows)

print("\nCluster centroids (ORIGINAL units) — sorted by 30d volatility descending:")
centers_df.orderBy(F.desc("ret_std_30")).show(best_k, truncate=False)

print("\nPer-cluster feature means (original units):")
(
    best_pred
    .groupBy("prediction")
    .agg(*[F.avg(c).alias(f"avg_{c}") for c in ml_cols])
    .orderBy("prediction")
    .show(best_k, truncate=False)
)

print("\nPer-cluster counts by coin (which coins dominate each regime):")
(
    best_pred
    .groupBy("prediction", "symbol")
    .count()
    .orderBy("prediction", F.desc("count"))
    .show(200, truncate=False)
)
# =========================
# 11) labeling & validation 
# =========================

# Build an label for each centroid
centers = centers_df.collect()
labels_by_cluster = {}
for r in centers:
    cid     = r["prediction"]
    mu_ret  = float(r["ret_1d_w"])
    mu_z    = float(r["zscore_7"])
    vol7    = float(r["ret_std_7"])
    vol30   = float(r["ret_std_30"])
    gap7    = float(r["ma_gap_7"])
    gap30   = float(r["ma_gap_30"])

    if (vol30 >= 0.075 and gap30 >= 0.30) or (vol7 >= 0.09 and gap7 >= 0.25):
        label = "Blow-off rally / parabolic"
    elif vol30 >= 0.07 and abs(mu_z) < 0.5:
        label = "Choppy high-vol sideways"
    elif mu_z >= 1.0 and gap30 > 0 and mu_ret > 0:
        label = "Bull impulse"
    elif mu_z <= -1.0 and gap30 < 0 and mu_ret < 0:
        label = "Sell-off / drawdown"
    else:
        label = "Calm / normal"

    labels_by_cluster[cid] = label

print("\nCluster labels:")
for cid in sorted(labels_by_cluster):
    print(f"  cluster {cid}: {labels_by_cluster[cid]}")

# Attach labels to every daily row used in clustering
map_labels = F.udf(lambda i: labels_by_cluster.get(int(i), "Unlabeled"))
labeled = best_pred.withColumn("regime", map_labels(F.col("prediction"))).cache()

print("\nPer-regime counts and feature means:")
(
    labeled
    .groupBy("regime")
    .agg(
        F.count("*").alias("rows"),
        F.avg("ret_1d_w").alias("avg_ret"),
        F.avg("zscore_7").alias("avg_z"),
        F.avg("ret_std_7").alias("avg_vol7"),
        F.avg("ret_std_30").alias("avg_vol30"),
        F.avg("ma_gap_7").alias("avg_gap7"),
        F.avg("ma_gap_30").alias("avg_gap30"),
    )
    .orderBy(F.desc("rows"))
    .show(truncate=False)
)

print("\nPer-regime share by coin (top 100 rows):")
(
    labeled
    .groupBy("symbol","regime")
    .count()
    .withColumn("pct", F.col("count") / F.sum("count").over(Window.partitionBy("symbol")))
    .orderBy("symbol", F.desc("pct"))
    .show(100, truncate=False)
)

# Cross-check: 2020-03-12 (COVID crash) and DOGE 2021-01-28(Pump and Dump)
print("\nEvent check — 2020-03-12 distribution:")
(
    labeled
    .filter(F.col("Date") == F.to_date(F.lit("2020-03-12")))
    .groupBy("regime")
    .count()
    .orderBy(F.desc("count"))
    .show(truncate=False)
)

print("\nEvent check — DOGE 2021-01-28 distribution:")
(
    labeled
    .filter((F.col("Date")==F.to_date(F.lit("2021-01-28"))) & (F.col("symbol")=="DOGE"))
    .groupBy("regime")
    .count()
    .orderBy(F.desc("count"))
    .show(truncate=False)
)

from pyspark.sql import functions as F, Window

REGIME_ORDER = [
    "Calm / normal",
    "Choppy high-vol sideways",
    "Sell-off / drawdown",
    "Bull impulse",
    "Blow-off rally / parabolic",
]

w_sym = Window.partitionBy("symbol").orderBy("Date")

trans_base = (
    labeled
    .select("symbol","Date","regime")
    .withColumn("prev_regime", F.lag("regime").over(w_sym))
    .where(F.col("prev_regime").isNotNull())
)

trans_counts = trans_base.groupBy("prev_regime","regime").count()

print("\nTransition counts (rows = prev_regime, cols = next_regime):")
trans_counts_mat = (
    trans_counts
    .groupBy("prev_regime")
    .pivot("regime", REGIME_ORDER)
    .sum("count")
    .fillna(0)
)
trans_counts_mat.show(100, truncate=False)

row_totals = trans_counts.groupBy("prev_regime").agg(F.sum("count").alias("row_total"))
trans_probs = (
    trans_counts.join(row_totals, "prev_regime")
                .withColumn("prob", F.col("count")/F.col("row_total"))
)

print("\nTransition probabilities (rows sum to 1):")
trans_probs_mat = (
    trans_probs
    .groupBy("prev_regime")
    .pivot("regime", REGIME_ORDER)
    .agg(F.sum("prob"))
    .fillna(0.0)
)
trans_probs_mat.show(100, truncate=False)

stickiness = (
    trans_probs
    .where(F.col("prev_regime")==F.col("regime"))
    .select(F.col("prev_regime").alias("regime"),
            F.col("prob").alias("stay_prob"))
)
print("\nPer-regime stay probability (stickiness):")
stickiness.orderBy("regime").show(truncate=False)

avg_stay = stickiness.agg(F.avg("stay_prob").alias("avg_stay_prob")).first()["avg_stay_prob"]
print(f"Average stay probability across regimes: {avg_stay:.3f}")

prev_reg = F.lag("regime").over(w_sym)
streak_marks = (
    labeled
    .select("symbol","Date","regime")
    .withColumn("is_new",
        F.when(prev_reg.isNull() | (F.col("regime") != prev_reg), 1).otherwise(0))
    .withColumn("streak_id", F.sum("is_new").over(w_sym))
)

streak_lengths = (
    streak_marks
    .groupBy("symbol","regime","streak_id")
    .agg(F.count(F.lit(1)).alias("streak_len"))
)

streak_stats = (
    streak_lengths.groupBy("regime")
    .agg(
        F.count("*").alias("n_streaks"),
        F.expr("percentile_approx(streak_len, 0.25)").alias("p25"),
        F.expr("percentile_approx(streak_len, 0.50)").alias("median"),
        F.expr("percentile_approx(streak_len, 0.75)").alias("p75"),
        F.avg("streak_len").alias("mean")
    )
    .orderBy("regime")
)

print("\nStreak length stats (days) per regime:")
streak_stats.show(truncate=False)

print("\nLongest streaks per regime (top 3 each):")
(
    streak_lengths
    .withColumn("rn", F.row_number().over(Window.partitionBy("regime").orderBy(F.desc("streak_len"))))
    .where(F.col("rn") <= 3)
    .orderBy("regime","rn")
    .select("regime","symbol","streak_len")
    .show(truncate=False)
)

# Monthly counts per regime -> shares per month
btc_month_counts = (
    labeled
    .filter(F.col("symbol")=="BTC")
    .withColumn("month", F.trunc(F.col("Date"), "MM"))  
    .groupBy("month","regime")
    .count()
)

btc_month_shares_long = (
    btc_month_counts.alias("c")
    .join(
        btc_month_counts.groupBy("month").agg(F.sum("count").alias("tot")),
        on="month", how="inner"
    )
    .withColumn("share", F.col("c.count")/F.col("tot"))
    .select("month","regime","share")
)

print("\nBTC monthly regime shares (long form, first 24 rows):")
btc_month_shares_long.orderBy("month","regime").show(24, truncate=False)

btc_month_shares_wide = (
    btc_month_shares_long
    .groupBy("month")
    .pivot("regime", REGIME_ORDER)
    .agg(F.first("share"))
    .fillna(0.0)
    .orderBy("month")
)

print("\nBTC monthly regime share (wide form, first 12 months):")
btc_month_shares_wide.show(12, truncate=False)

try:
    import pandas as pd, matplotlib.pyplot as plt
    pdf = btc_month_shares_wide.toPandas().sort_values("month")
    ax = pdf.plot.area(x="month", y=REGIME_ORDER, figsize=(11,6))
    ax.set_title("BTC — Monthly Regime Share")
    ax.set_ylabel("Share")
    ax.set_xlabel("Month")
    ax.legend(loc="center left", bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    plt.savefig("btc_regime_share.png", dpi=150)
    print("Saved plot → btc_regime_share.png")
except Exception as e:
    print("Plot skipped (likely headless driver or missing libs):", e)

# Per-coin rankings: Blow-off & Sell-off fractions
totals = labeled.groupBy("symbol").agg(F.count("*").alias("n_total"))

blow = (
    labeled.filter(F.col("regime")=="Blow-off rally / parabolic")
           .groupBy("symbol").agg(F.count("*").alias("n_blow"))
)
sell = (
    labeled.filter(F.col("regime")=="Sell-off / drawdown")
           .groupBy("symbol").agg(F.count("*").alias("n_sell"))
)

bias = (
    totals.join(blow, "symbol", "left")
          .join(sell, "symbol", "left")
          .fillna(0, subset=["n_blow","n_sell"])
          .withColumn("blow_frac", F.col("n_blow")/F.col("n_total"))
          .withColumn("sell_frac", F.col("n_sell")/F.col("n_total"))
)

print("\nTop coins by fraction of Blow-off days:")
bias.select("symbol","n_total","n_blow","blow_frac").orderBy(F.desc("blow_frac")).show(25, truncate=False)

print("\nTop coins by fraction of Sell-off days:")
bias.select("symbol","n_total","n_sell","sell_frac").orderBy(F.desc("sell_frac")).show(25, truncate=False)

spark.stop()
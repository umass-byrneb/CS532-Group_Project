# MS2: Rolling features + z-scores
from pyspark.sql import functions as F, Window
from src.common import config as C
from src.common.spark import get_spark
from src.common.io_utils import ensure_dir, read_parquet, save_parquet

def main():
    spark = get_spark("MS2_Features")

    # Load cleaned dataset 
    cleaned_path = f"{C.ART_DIR}/cleaned"
    df = read_parquet(spark, cleaned_path)

    # Rolling windows per coin 
    w = Window.partitionBy("symbol").orderBy("Date")

    ma7  = F.avg("Close").over(w.rowsBetween(-6, 0))
    ma30 = F.avg("Close").over(w.rowsBetween(-29, 0))

    close_prev = F.lag("Close", 1).over(w)

    if C.USE_LOG_RET:
        ret_raw = F.when((F.col("Close") > 0) & (close_prev > 0),
                         F.log(F.col("Close")) - F.log(close_prev))
    else:
        ret_raw = (F.col("Close") / close_prev) - F.lit(1.0)

    # simple % return for top-5s
    ret_pct = (F.col("Close") / close_prev) - F.lit(1.0)

    feat = (
        df
        .withColumn("ma_close_7",  ma7)
        .withColumn("ma_close_30", ma30)
        .withColumn("ret_1d",      ret_raw)
        .withColumn("ret_pct",     ret_pct)
    )

    qs = (
        feat.where(F.col("ret_1d").isNotNull())
            .groupBy("symbol")
            .agg(
                F.expr(f"percentile_approx(ret_1d, {C.WINSOR_LO}, 10000)").alias("q_lo"),
                F.expr(f"percentile_approx(ret_1d, {C.WINSOR_HI}, 10000)").alias("q_hi"),
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

    # Rolling stats for z
    w_prev7  = w.rowsBetween(-7,  -1)
    w_prev30 = w.rowsBetween(-30, -1)

    feat = (
        feat
        .withColumn("ret_mean_7",  F.avg("ret_1d_w").over(w_prev7))
        .withColumn("ret_std_7",   F.stddev_samp("ret_1d_w").over(w_prev7))
        .withColumn("cnt_7",       F.count("ret_1d_w").over(w_prev7))
        .withColumn("ret_mean_30", F.avg("ret_1d_w").over(w_prev30))
        .withColumn("ret_std_30",  F.stddev_samp("ret_1d_w").over(w_prev30))
        .withColumn("cnt_30",      F.count("ret_1d_w").over(w_prev30))
        .withColumn("rn",          F.row_number().over(w))
    )

    z7 = F.when(
            (F.col("cnt_7")  >= F.lit(C.MIN_PRIOR_OBS_7)) &
            (F.col("ret_std_7")  >= F.lit(C.SIGMA_FLOOR)) &
            (F.col("rn") >= F.lit(C.BURN_IN_DAYS)),
            (F.col("ret_1d_w") - F.col("ret_mean_7")) / F.col("ret_std_7")
        )

    z30 = F.when(
            (F.col("cnt_30") >= F.lit(C.MIN_PRIOR_OBS_30)) &
            (F.col("ret_std_30") >= F.lit(C.SIGMA_FLOOR)) &
            (F.col("rn") >= F.lit(C.BURN_IN_DAYS)),
            (F.col("ret_1d_w") - F.col("ret_mean_30")) / F.col("ret_std_30")
        )

    feat = (
        feat
        .withColumn("zscore_7",  z7)
        .withColumn("zscore_30", z30)
        .withColumn("ma_gap_7",  F.col("Close") / F.col("ma_close_7")  - F.lit(1.0))
        .withColumn("ma_gap_30", F.col("Close") / F.col("ma_close_30") - F.lit(1.0))
    )

    # Save ML-ready full features 
    ensure_dir(f"{C.ART_DIR}/features_full")
    save_parquet(feat, f"{C.ART_DIR}/features_full", partitionBy=["symbol"], mode=C.SPARK_WRITE_MODE)

    ml = (
        feat
        .where(
            F.col("ret_1d_w").isNotNull() &
            F.col("zscore_7").isNotNull() &
            F.col("ret_std_7").isNotNull() &
            F.col("ret_std_30").isNotNull() &
            F.col("ma_gap_7").isNotNull() &
            F.col("ma_gap_30").isNotNull()
        )
        .select("Date","symbol","ret_1d_w","zscore_7","ret_std_7","ret_std_30","ma_gap_7","ma_gap_30")
    )
    ensure_dir(f"{C.ART_DIR}/features_ml_daily")
    save_parquet(ml, f"{C.ART_DIR}/features_ml_daily", partitionBy=["symbol"], mode=C.SPARK_WRITE_MODE)

    total_rows = feat.count()
    syms = [r["symbol"] for r in feat.select("symbol").distinct().orderBy("symbol").collect()]
    span = feat.select(F.min("Date").alias("minD"), F.max("Date").alias("maxD")).first()
    print(f"\nFeature rows: {total_rows}")
    print(f"\nSymbols ({len(syms)}): {', '.join(syms)}")
    print(f"Date range: {span['minD']} → {span['maxD']}\n")

    print("Sample (by symbol then date):")
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

    print(f"\nTop {C.TOPK} absolute z-score (7-day window) across all coins (valid z only):")
    (
        feat.select("Date","symbol","Close","ret_1d","ret_1d_w","ret_mean_7","ret_std_7","zscore_7")
            .where(F.col("zscore_7").isNotNull())
            .withColumn("abs_z", F.abs(F.col("zscore_7")))
            .orderBy(F.desc("abs_z"))
            .select("Date","symbol","Close","ret_1d","ret_1d_w","ret_mean_7","ret_std_7","zscore_7")
            .show(C.TOPK, truncate=False)
    )

    print("\nAnomaly counts per coin (|z|>=3.0 on 7-day window):")
    (
        feat.select("symbol", F.when(F.abs(F.col("zscore_7")) >= 3.0, 1).otherwise(0).alias("anom"))
            .groupBy("symbol")
            .agg(F.sum("anom").alias("anom_7_count"))
            .orderBy("symbol")
            .show(100, truncate=False)
    )

    # Top 5 jumps per coin (by simple % return):
    (
        feat.where(F.col("ret_pct").isNotNull())  # <-- add this
            .withColumn("r", F.row_number().over(
                Window.partitionBy("symbol").orderBy(F.desc("ret_pct"))
            ))
            .where(F.col("r") <= 5)
            .select("symbol","Date","ret_pct","zscore_7")
            .orderBy("symbol","Date")
            .show(100, truncate=False)
    )

    # Top 5 drops per coin (by simple % return):
    (
        feat.where(F.col("ret_pct").isNotNull())  # <-- add this
            .withColumn("r", F.row_number().over(
                Window.partitionBy("symbol").orderBy(F.asc("ret_pct"))
            ))
            .where(F.col("r") <= 5)
            .select("symbol","Date","ret_pct","zscore_7")
            .orderBy("symbol","Date")
            .show(100, truncate=False)
    )

    # Volatility-over-time sample (first 20 rows of one coin)
    print("\nSample volatility-over-time (30-day rolling std) per coin:")
    (
        feat.select("Date","symbol","ret_std_30")
            .where(F.col("symbol") == syms[0])  
            .orderBy("Date")
            .show(20, truncate=False)
    )

    before = feat.count()
    after  = ml.count()
    print("\n=== Step 1: ML dataset (clustering) sanity checks ===")
    print(f"Rows before ML filters: {before}")
    print(f"Rows after  ML filters: {after}  ({after/before*100:.1f}% retained)")
    for c in ["ret_1d_w","zscore_7","ret_std_7","ret_std_30","ma_gap_7","ma_gap_30"]:
        nulls = ml.filter(F.col(c).isNull()).count()
        print(f"NULLs in {c}: {nulls}")

    # Percentiles per feature 
    def pct(col):
        return F.expr(f"percentile_approx({col}, array(0.01, 0.50, 0.99), 10000)").alias(col)
    pcts = ml.agg(
        pct("ret_1d_w"),
        pct("zscore_7"),
        pct("ret_std_7"),
        pct("ret_std_30"),
        pct("ma_gap_7"),
        pct("ma_gap_30"),
    ).first()
    labels = ["ret_1d_w","zscore_7","ret_std_7","ret_std_30","ma_gap_7","ma_gap_30"]
    print("\nPercentiles (p01, p50, p99) per feature:")
    for i, lbl in enumerate(labels):
        p01, p50, p99 = pcts[i][0], pcts[i][1], pcts[i][2]
        print(f"    {lbl} | p01={p01}  p50={p50}  p99={p99}")

if __name__ == "__main__":
    main()

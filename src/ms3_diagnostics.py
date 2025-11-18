# MS3: Feature EDA — anomaly & volatility reporting (no clustering)
from pyspark.sql import functions as F, Window
from src.common.spark import get_spark
from src.common.io_utils import ensure_dir, read_parquet, save_parquet
from src.common import config as C

def main():
    spark = get_spark("MS3_Diagnostics")

    ml_path = f"{C.ART_DIR}/features_ml_daily"
    df = read_parquet(spark, ml_path)  

    syms = [r["symbol"] for r in df.select("symbol").distinct().orderBy("symbol").collect()]
    span = df.agg(F.min("Date").alias("minD"), F.max("Date").alias("maxD")).first()
    total = df.count()
    print(f"\nMS3 dataset: rows={total}, symbols={len(syms)}, span={span['minD']} → {span['maxD']}")
    print("Symbols:", ", ".join(syms))

    def pct(col):
        return F.expr(f"percentile_approx({col}, array(0.01,0.5,0.99), 10000)").alias(col)
    pcts = df.agg(
        pct("ret_1d_w"), pct("zscore_7"), pct("ret_std_7"), pct("ret_std_30"), pct("ma_gap_7"), pct("ma_gap_30")
    ).first()
    labels = ["ret_1d_w","zscore_7","ret_std_7","ret_std_30","ma_gap_7","ma_gap_30"]
    print("\nPercentiles (p01, p50, p99):")
    for i, lbl in enumerate(labels):
        p01, p50, p99 = pcts[i][0], pcts[i][1], pcts[i][2]
        print(f"  {lbl:>10} | p01={p01}  p50={p50}  p99={p99}")

    # Anomaly counts per coin and date 
    anom_up  = F.when(F.col("zscore_7") >= 3.0, 1).otherwise(0).alias("anom_up")
    anom_dn  = F.when(F.col("zscore_7") <= -3.0, 1).otherwise(0).alias("anom_dn")
    anoms = df.select("Date","symbol","zscore_7", anom_up, anom_dn)

    by_coin = (
        anoms.groupBy("symbol")
             .agg(F.sum("anom_up").alias("up_3p"), F.sum("anom_dn").alias("dn_3p"))
             .orderBy(F.desc("dn_3p"), F.desc("up_3p"))
    )
    print("\nAnomaly counts per coin (|z|>=3):")
    by_coin.show(100, truncate=False)

    by_date = (
        anoms.groupBy("Date")
             .agg(F.sum("anom_up").alias("up_3p"), F.sum("anom_dn").alias("dn_3p"))
             .orderBy("Date")
    )

    # Notable date check 2020-03-12 and any day with >= 5 down-anomalies
    for d in ["2020-03-12"]:
        row = by_date.where(F.col("Date") == d).first()
        if row:
            print(f"\nEvent check — {d}: up={row['up_3p']}, down={row['dn_3p']} (count across all coins)")

    print("\nDays with widespread down anomalies (dn_3p >= 5):")
    by_date.where(F.col("dn_3p") >= 5).orderBy(F.desc("dn_3p")).show(50, truncate=False)

    # Volatility summaries
    vol_summ = (
        df.groupBy("symbol")
          .agg(
              F.expr("percentile_approx(ret_std_30, 0.5, 10000)").alias("vol30_med"),
              F.max("ret_std_30").alias("vol30_max")
          )
          .orderBy(F.desc("vol30_med"))
    )
    print("\nVolatility (30d) per coin — median & max:")
    vol_summ.show(100, truncate=False)

    # Save artifacts
    out_dir = f"{C.ART_DIR}/ms3"
    ensure_dir(out_dir)
    save_parquet(by_coin, f"{out_dir}/anomaly_counts_by_coin", mode=C.SPARK_WRITE_MODE)
    save_parquet(by_date, f"{out_dir}/anomaly_counts_by_date", mode=C.SPARK_WRITE_MODE)
    save_parquet(vol_summ, f"{out_dir}/vol30_summary", mode=C.SPARK_WRITE_MODE)

    # Plot: BTC 30d vol over time
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        btc = (
            df.where(F.col("symbol")=="BTC")
              .select("Date","ret_std_30")
              .orderBy("Date")
              .toPandas()
        )
        plt.figure()
        plt.plot(pd.to_datetime(btc["Date"]), btc["ret_std_30"])
        plt.xlabel("Date"); plt.ylabel("BTC 30d rolling std (winsorized returns)")
        plt.title("BTC 30d Volatility")
        ensure_dir(f"{C.ART_DIR}/plots")
        plt.tight_layout()
        plt.savefig(f"{C.ART_DIR}/plots/btc_vol30.png", dpi=150)
        print(f"\nSaved plot → {C.ART_DIR}/plots/btc_vol30.png")
    except Exception as e:
        print(f"\n[plot skipped] {e}")

if __name__ == "__main__":
    main()

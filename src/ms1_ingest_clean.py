# Milestone 1: Ingest & Clean → save parquet
from pyspark.sql import functions as F
from src.common import config as C
from src.common.spark import get_spark
from src.common.io_utils import ensure_dir, save_parquet
from src.common.transforms import (
    normalize_schema, clean_numeric, parse_dates, filter_symbols, select_core_columns
)

def main():
    spark = get_spark("MS1_IngestClean")
    raw = (
        spark.read
             .option("header", "true")
             .option("inferSchema", "false")
             .option("recursiveFileLookup", "true")
             .option("pathGlobFilter", "*.csv")
             .csv(C.DATA_DIR)
             .withColumn("file_name", F.input_file_name())
    )

    df = normalize_schema(raw)
    df = clean_numeric(df)
    df = parse_dates(df)
    df = df.dropna(subset=["symbol", "Date", "Close"])
    df = filter_symbols(df, C.EXCLUDE_SYMBOLS)
    df = select_core_columns(df)

    total_rows = df.count()
    syms = [r["symbol"] for r in df.select("symbol").distinct().orderBy("symbol").collect()]
    span = df.select(F.min("Date").alias("minD"), F.max("Date").alias("maxD")).first()

    print(f"\nCleaned rows: {total_rows}")
    print(f"\nSymbols ({len(syms)}): {', '.join(syms)}")
    print(f"Date range: {span['minD']} → {span['maxD']}\n")

    out = f"{C.ART_DIR}/cleaned"
    ensure_dir(out)
    save_parquet(df, out, partitionBy=["symbol"], mode=C.SPARK_WRITE_MODE)

    print("Sample (by symbol then date):")
    (
        df.orderBy("symbol", "Date")
          .select("Date", "symbol", "Close", "Volume", "Marketcap")
          .show(10, truncate=False)
    )

if __name__ == "__main__":
    main()

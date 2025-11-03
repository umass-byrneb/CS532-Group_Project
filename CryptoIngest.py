# Milestone 1: Ingest & Initial Exploration

from pyspark.sql import SparkSession, functions as F
import sys, re

spark = (
    SparkSession.builder
    .appName("SimpleApp")
    .master("local[*]")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

data_path = sys.argv[1] if len(sys.argv) > 1 else "data"

raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.csv")
    .csv(data_path)
    .withColumn("symbol_from_file", F.regexp_extract(F.input_file_name(), r"([^/]+)\.csv$", 1))
)

def norm(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name).lower()

for c in raw.columns:
    raw = raw.withColumnRenamed(c, norm(c))

if "date" in raw.columns:
    raw = raw.withColumn("date", F.trim(F.col("date")))

cols = set(raw.columns)

symbol_candidates = [c for c in ["symbol", "symbolfromfile", "symbol_from_file", "name"] if c in cols]
symbol_expr = (F.coalesce(*[F.col(c) for c in symbol_candidates]) if symbol_candidates else F.lit(None)).alias("symbol")

ts_expr = F.coalesce(
    F.to_timestamp("date", "yyyy-MM-dd HH:mm:ss"),
    F.to_timestamp("date", "yyyy-MM-dd'T'HH:mm:ss"),
    F.to_timestamp("date", "yyyy-MM-dd"),
    F.to_timestamp("date", "MMM dd, yyyy"),
    F.to_timestamp("date", "M/d/yyyy"),
    F.to_timestamp("date", "MM/dd/yyyy")
)
date_expr = F.to_date(ts_expr).alias("Date")

num_candidates = ["open", "high", "low", "close", "volume", "marketcap"]
numeric_cols = [c for c in num_candidates if c in cols]

clean = raw
for c in numeric_cols:
    clean = clean.withColumn(c, F.regexp_replace(F.col(c), r"[^0-9.]", "").cast("double"))

rename_map = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "marketcap": "MarketCap",
}

select_cols = [date_expr, symbol_expr] + [F.col(c).alias(rename_map[c]) for c in numeric_cols]

df = (
    clean.select(*select_cols)
         .filter(F.col("Date").isNotNull())
         .cache()
)

# csv checks
total_rows = df.count()

symbols = [r["symbol"] for r in df.select("symbol").distinct().orderBy("symbol").collect()]
rng = df.agg(F.min("Date").alias("min_date"), F.max("Date").alias("max_date")).first()

print(f"\nLoaded rows: {total_rows}")
print(f"Symbols ({len(symbols)}): {', '.join(symbols)}")
print(f"Overall date range: {rng['min_date']} → {rng['max_date']}\n")

print("Top per-symbol row counts:")
df.groupBy("symbol").count().orderBy(F.desc("count")).show(50, truncate=False)

print("Sample rows (by symbol then date):")
df.orderBy("symbol", "Date").show(10, truncate=False)

spark.stop()

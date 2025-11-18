# cleaning helpers used in Milestone 1
from pyspark.sql import functions as F

def normalize_schema(df):
    """
    Standardize columns: SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap
    """
    file_base = F.regexp_extract(F.col("file_name"), r"([^/\\]+)$", 1)
    token = F.regexp_extract(file_base, r"coin_([^\.]+)\.csv", 1)

    symbol_fallback = F.upper(F.coalesce(F.col("Symbol"), token))

    out = (
        df
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
        .withColumn("symbol", symbol_fallback)
    )
    return out

def clean_numeric(df):
    def _clean(col):
        return F.regexp_replace(F.col(col), r"[^0-9.\-]", "").cast("double")

    return (
        df
        .withColumn("Open",      _clean("Open"))
        .withColumn("High",      _clean("High"))
        .withColumn("Low",       _clean("Low"))
        .withColumn("Close",     _clean("Close"))
        .withColumn("Volume",    _clean("Volume"))
        .withColumn("Marketcap", _clean("Marketcap"))
    )

def parse_dates(df):
    return df.withColumn(
        "Date",
        F.to_date(F.substring(F.col("Date").cast("string"), 1, 10), "yyyy-MM-dd")
    )

def filter_symbols(df, exclude_upper):
    return df.filter(~F.col("symbol").isin([s.upper() for s in exclude_upper]))

def select_core_columns(df):
    return df.select("Date", "symbol", "Open", "High", "Low", "Close", "Volume", "Marketcap")

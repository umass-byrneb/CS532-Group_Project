import os

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_parquet(df, path: str, partitionBy=None, mode="overwrite"):
    writer = df.write.mode(mode)
    if partitionBy:
        writer = writer.partitionBy(partitionBy)
    writer.parquet(path)

def read_parquet(spark, path: str):
    return spark.read.parquet(path)

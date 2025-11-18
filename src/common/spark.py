from pyspark.sql import SparkSession

def get_spark(app="CryptoPipeline"):
    spark = SparkSession.builder.appName(app).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

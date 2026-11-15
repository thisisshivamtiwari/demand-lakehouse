from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def get_spark(master: str = "local[*]", partitions: int = 16, memory: str = "3g") -> SparkSession:
    """A local Spark session with Delta Lake. One session per process: change the master by starting a new process."""
    builder = (
        SparkSession.builder.master(master)
        .appName("demand-lakehouse")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.driver.memory", memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.databricks.delta.snapshotPartitions", "2")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

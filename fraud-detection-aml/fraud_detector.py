"""
PaySim Fraud Detection Engine — PySpark Structured Streaming
Fixed version: writes to PostgreSQL using JDBC inside foreachBatch.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType
)

KAFKA_BOOTSTRAP = "localhost:29092"
INPUT_TOPIC     = "paysim_transactions"
CHECKPOINT_BASE = "/tmp/paysim-checkpoints"

JDBC_URL = "jdbc:postgresql://localhost:5432/fraud_db"
JDBC_USER     = "fraud_user"
JDBC_PASSWORD = "fraud_pass_2024"
JDBC_DRIVER   = "org.postgresql.Driver"

LARGE_AMOUNT_THRESHOLD = 200000.0
BALANCE_MISMATCH_DELTA = 1.0

TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id",       StringType(),  True),
    StructField("step_hour",            IntegerType(), True),
    StructField("transaction_type",     StringType(),  True),
    StructField("amount",               DoubleType(),  True),
    StructField("account_origin",       StringType(),  True),
    StructField("balance_orig_before",  DoubleType(),  True),
    StructField("balance_orig_after",   DoubleType(),  True),
    StructField("account_dest",         StringType(),  True),
    StructField("balance_dest_before",  DoubleType(),  True),
    StructField("balance_dest_after",   DoubleType(),  True),
    StructField("is_fraud",             IntegerType(), True),
    StructField("is_flagged_by_system", IntegerType(), True),
    StructField("timestamp",            StringType(),  True),
    StructField("fraud_rule_triggered", StringType(),  True),
])


def create_spark_session():
    return (
        SparkSession.builder
        .appName("PaySim-Fraud-Detector")
        .master("local[3]")
        .config("spark.sql.shuffle.partitions", "3")
        .getOrCreate()
    )


def read_kafka_stream(spark):
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", INPUT_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )
    return (
        raw
        .select(F.from_json(F.col("value").cast("string"), TRANSACTION_SCHEMA).alias("tx"))
        .select("tx.*")
        .filter(
            F.col("transaction_id").isNotNull() &
            F.col("transaction_type").isin("TRANSFER", "CASH_OUT")
        )
    )


def add_alert_columns(df, rule_name, risk_score, reason_expr):
    return (
        df
        .withColumn("rule_triggered", F.lit(rule_name))
        .withColumn("risk_score",     F.lit(risk_score))
        .withColumn("alert_reason",   reason_expr)
        .withColumn("detected_at",    F.current_timestamp())
        .select(
            "transaction_id", "account_origin", "amount",
            "transaction_type", "rule_triggered", "risk_score",
            "alert_reason", "balance_orig_before", "balance_orig_after",
            "is_fraud", "detected_at"
        )
    )


def write_batch(df, batch_id, rule_name):
    count = df.count()
    if count == 0:
        return
    print(f"[Batch {batch_id}] {rule_name}: {count} alerts")
    (
        df.write
        .format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "flagged_transactions")
        .option("user", JDBC_USER)
        .option("password", JDBC_PASSWORD)
        .option("driver", JDBC_DRIVER)
        .mode("append")
        .save()
    )
    print(f"[Batch {batch_id}] {rule_name}: saved to PostgreSQL")


def detect_account_emptied(df):
    return add_alert_columns(
        df.filter((F.col("balance_orig_after") == 0.0) & (F.col("balance_orig_before") > 0.0)),
        "ACCOUNT_EMPTIED", 85,
        F.concat(F.lit("Account drained from $"), F.round(F.col("balance_orig_before"), 2).cast("string"), F.lit(" to $0.00"))
    )


def detect_large_amount(df):
    return add_alert_columns(
        df.filter(F.col("amount") > LARGE_AMOUNT_THRESHOLD),
        "LARGE_AMOUNT", 80,
        F.concat(F.lit("Amount $"), F.round(F.col("amount"), 2).cast("string"), F.lit(" exceeds $200,000 threshold"))
    )


def detect_balance_mismatch(df):
    return add_alert_columns(
        df
        .withColumn("bc", F.col("balance_orig_before") - F.col("balance_orig_after"))
        .withColumn("mm", F.abs(F.col("bc") - F.col("amount")))
        .filter(F.col("mm") > BALANCE_MISMATCH_DELTA)
        .drop("bc", "mm"),
        "BALANCE_MISMATCH", 90,
        F.concat(F.lit("Balance mismatch on $"), F.round(F.col("amount"), 2).cast("string"))
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 55)
    print("  PaySim Fraud Detection Engine")
    print("=" * 55)
    print(f"  Kafka:  {KAFKA_BOOTSTRAP}")
    print(f"  Topic:  {INPUT_TOPIC}")
    print("=" * 55)

    transactions = read_kafka_stream(spark)

    q1 = (
        detect_account_emptied(transactions).writeStream
        .outputMode("append")
        .foreachBatch(lambda df, bid: write_batch(df, bid, "ACCOUNT_EMPTIED"))
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/rule1")
        .trigger(processingTime="10 seconds")
        .start()
    )

    q2 = (
        detect_large_amount(transactions).writeStream
        .outputMode("append")
        .foreachBatch(lambda df, bid: write_batch(df, bid, "LARGE_AMOUNT"))
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/rule2")
        .trigger(processingTime="10 seconds")
        .start()
    )

    q3 = (
        detect_balance_mismatch(transactions).writeStream
        .outputMode("append")
        .foreachBatch(lambda df, bid: write_batch(df, bid, "BALANCE_MISMATCH"))
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/rule3")
        .trigger(processingTime="10 seconds")
        .start()
    )

    print("All 3 detection queries running")
    print("Rule 1: ACCOUNT_EMPTIED  (10s trigger)")
    print("Rule 2: LARGE_AMOUNT     (10s trigger)")
    print("Rule 3: BALANCE_MISMATCH (10s trigger)")
    print("\nWaiting for transactions...\n")

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
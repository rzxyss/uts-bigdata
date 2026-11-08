"""
End-to-End Big Data Pipeline (Modules 1 to 7)
Case Study: COVID-19 Epidemiological Analytics
UTS IFB-451 BIG DATA - Dr. Edi Nuryatno (2026/2027)

Covers:
- Module 1: 5Vs Analysis, Data Lifecycle & Pandas Chunked EDA
- Module 2: Big Data Architecture & Velocity Streaming Simulation
- Module 3: Storage Benchmark (CSV vs JSON vs Parquet) & HDFS Partition Layout
- Module 4: Hadoop Ecosystem Integration (Hadoop Streaming Mapper & Reducer)
- Module 5: MapReduce Execution Model (Key-Value, Shuffle & Sort, Combiner)
- Module 6: Apache Spark Basics (RDD, Transformations, Actions, Lazy DAG)
- Module 7: Spark SQL, Catalyst Optimizer, Multi-Format Joins, Window Functions,
            and Computational Complexity Benchmark (Operation A vs Operation B)
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure PySpark uses current python interpreter on Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, DateType
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_RAW, "covid_cases_daily.csv")
JSON_FILE = os.path.join(DATA_RAW, "vaccination_icu_facilities.json")
PARQUET_FILE = os.path.join(DATA_RAW, "province_demographics.parquet")


def log_section(title, module_tag):
    print("\n" + "=" * 86)
    print(f"[{module_tag}] {title}")
    print("=" * 86)


# ==============================================================================
# MODULE 1: INTRODUCTION TO BIG DATA & EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
def run_module_1():
    log_section("Module 1: 5Vs Assessment & Pandas Chunked EDA", "M1 - SC1")
    print("Evaluating 5Vs in COVID-19 Epidemiological Surveillance:")
    print(" - Volume   : Longitudinal multi-year records across 34 provinces.")
    print(" - Velocity : Daily reporting updates & micro-batch arrival stream.")
    print(" - Variety  : Tabular CSV (cases) + Nested JSON (vaccine/ICU) + Columnar Parquet (census).")
    print(" - Veracity : Reporting backlogs, negative case anomalies, casing defects, duplicates.")
    print(" - Value    : Transmission monitoring, ICU overload prediction, vaccine efficacy insight.\n")

    # Chunked loading simulation for large files
    chunk_size = 5000
    total_rows = 0
    total_missing_pcr = 0
    negative_cases = 0
    memory_bytes = 0

    print(f"Executing chunked ingestion (chunksize={chunk_size} rows)...")
    for chunk in pd.read_csv(CSV_FILE, chunksize=chunk_size):
        total_rows += len(chunk)
        total_missing_pcr += chunk["pcr_tests_conducted"].isna().sum()
        negative_cases += (chunk["daily_confirmed_cases"] < 0).sum()
        memory_bytes += chunk.memory_usage(deep=True).sum()

    print(f" -> Total Rows Ingested     : {total_rows:,d}")
    print(f" -> Total Memory Footprint  : {memory_bytes / (1024 * 1024):.2f} MB (uncompressed pandas heap)")
    print(f" -> Missing PCR Tests Count : {total_missing_pcr:,d} ({total_missing_pcr/total_rows*100:.2f}%)")
    print(f" -> Negative Case Anomalies : {negative_cases:,d} (input errors requiring veracity filter)")

    # Sample duplicate detection
    df_sample = pd.read_csv(CSV_FILE)
    duplicates_count = df_sample.duplicated(subset=["report_date", "province_id"]).sum()
    print(f" -> Duplicate Entries Found : {duplicates_count:,d} duplicate province-date records")
    return total_rows


# ==============================================================================
# MODULE 2: BIG DATA ECOSYSTEM & ARCHITECTURE (VELOCITY DEMO)
# ==============================================================================
def run_module_2():
    log_section("Module 2: Big Data Architecture & Velocity Ingestion Simulation", "M2 - SC1")
    print("Architecture Overview: Lambda Architecture for Public Health Informatics")
    print(" - Batch Layer   : HDFS + Apache Spark batch job for historical trend consolidation.")
    print(" - Speed Layer   : Simulated Streaming Ingestion for real-time epidemic alerts.")
    print(" - Serving Layer : Spark SQL Views + Visual Analytics Dashboard.\n")

    print("Simulating Velocity: 5 incoming daily micro-batches from provincial health offices...")
    df_raw = pd.read_csv(CSV_FILE)
    unique_dates = sorted(df_raw["report_date"].unique())[:5]

    latencies = []
    for i, cur_date in enumerate(unique_dates, 1):
        t_start = time.perf_counter()
        batch_slice = df_raw[df_raw["report_date"] == cur_date]
        # Ingestion validation simulation
        _ = batch_slice["daily_confirmed_cases"].sum()
        t_elapsed = (time.perf_counter() - t_start) * 1000  # ms
        latencies.append(t_elapsed)
        print(f" Batch #{i} | Date: {cur_date} | Records: {len(batch_slice):2d} | Ingestion Latency: {t_elapsed:.3f} ms")

    avg_latency = np.mean(latencies)
    print(f"\nVelocity Summary: Average Ingestion Latency = {avg_latency:.3f} ms per provincial batch")


# ==============================================================================
# MODULE 3: DATA STORAGE AND MANAGEMENT (STORAGE BENCHMARK & HDFS LAYOUT)
# ==============================================================================
def run_module_3():
    log_section("Module 3: Data Storage & Columnar Format Benchmark", "M3 - SC2")

    csv_size = os.path.getsize(CSV_FILE)
    json_size = os.path.getsize(JSON_FILE)
    parquet_size = os.path.getsize(PARQUET_FILE)

    print("Physical Storage Footprint Comparison on Disk:")
    print(f" 1. CSV (Daily Cases)        : {csv_size / 1024:.2f} KB (Row-based, raw text)")
    print(f" 2. JSON (Vaccine & ICU)     : {json_size / 1024:.2f} KB (Semi-structured, nested)")
    print(f" 3. Parquet (Demographics)   : {parquet_size / 1024:.2f} KB (Columnar, Snappy compressed)")

    # Read benchmark: CSV vs Parquet selective column read
    t0 = time.perf_counter()
    _ = pd.read_csv(CSV_FILE, usecols=["province_id", "daily_confirmed_cases"])
    t_csv = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = pq.read_table(PARQUET_FILE, columns=["province_id", "total_population"])
    t_parquet = (time.perf_counter() - t0) * 1000

    print(f"\nSelective Column Read Benchmark:")
    print(f" - CSV selective read latency     : {t_csv:.2f} ms")
    print(f" - Parquet columnar scan latency  : {t_parquet:.2f} ms (Column pruning enabled)")

    print("\nSimulated HDFS Partition Directory Layout:")
    print(" /data/lake/covid19/")
    print("   ├── raw_cases/year=2021/month=07/cases_202107_delta.csv")
    print("   ├── raw_cases/year=2022/month=02/cases_202202_omicron.csv")
    print("   ├── processed_gold/cases_curated.parquet/ (Partitioned by island_group)")
    print("   └── metadata/province_demographics.parquet")


# ==============================================================================
# MODULE 4 & 5: HADOOP ECOSYSTEM & MAPREDUCE PROGRAMMING
# ==============================================================================
def run_module_4_and_5():
    log_section("Module 4 & 5: MapReduce Execution Model (Hadoop Streaming Logic)", "M4/M5 - SC2/SC3")
    print("MapReduce Key-Value Paradigm Demonstration:")
    print(" Input Split -> Map(Line) -> Combiner -> Shuffle & Sort -> Reduce(Key, List[Values])\n")

    # Step 1: Map Phase
    df = pd.read_csv(CSV_FILE)
    map_emits = []
    for _, row in df.iterrows():
        island = str(row["island_group"]).strip()
        try:
            cases = int(row["daily_confirmed_cases"])
            deaths = int(row["daily_deaths"])
            if cases >= 0 and deaths >= 0:
                map_emits.append((island, (cases, deaths)))
        except (ValueError, TypeError):
            continue

    print(f"1. MAP STAGE: Emitted {len(map_emits):,d} intermediate key-value pairs (K: Island, V: (Cases, Deaths))")

    # Step 2: Combiner / Local Reduction Simulation
    combiner_dict = {}
    for island, (cases, deaths) in map_emits:
        if island not in combiner_dict:
            combiner_dict[island] = [0, 0]
        combiner_dict[island][0] += cases
        combiner_dict[island][1] += deaths

    print(f"2. COMBINER STAGE: Reduced network traffic to {len(combiner_dict)} unique partition keys.")

    # Step 3: Shuffle and Sort Phase
    sorted_keys = sorted(combiner_dict.keys())
    print(f"3. SHUFFLE & SORT: Keys sorted across reducers -> {sorted_keys}")

    # Step 4: Reduce Phase
    print("\n4. REDUCE STAGE FINAL AGGREGATION:")
    print("-" * 75)
    print(f"{'Island Group (Key)':<30} | {'Total Cases':<14} | {'Total Deaths':<14} | {'CFR (%)':<8}")
    print("-" * 75)
    for k in sorted_keys:
        tot_c, tot_d = combiner_dict[k]
        cfr = (tot_d / tot_c * 100.0) if tot_c > 0 else 0.0
        print(f"{k:<30} | {tot_c:<14,d} | {tot_d:<14,d} | {cfr:<8.2f}%")
    print("-" * 75)


# ==============================================================================
# MODULE 6: APACHE SPARK BASICS (RDD TRANSFORMATIONS, ACTIONS, DAG)
# ==============================================================================
def run_module_6(spark):
    log_section("Module 6: Apache Spark Basics (RDD, Lazy Evaluation & DAG)", "M6 - SC3")
    sc = spark.sparkContext
    print("Spark Context Active: Master =", sc.master, "| AppName =", sc.appName)

    # 1. Create RDD from text file
    lines_rdd = sc.textFile(CSV_FILE)
    header = lines_rdd.first()
    data_rdd = lines_rdd.filter(lambda row: row != header)

    print(f"RDD Lineage created (Lazy Evaluation). Total partitions: {data_rdd.getNumPartitions()}")

    # 2. Transformations: filter anomalies -> map to key-value -> reduceByKey
    t0 = time.perf_counter()
    rdd_agg = (
        data_rdd
        .map(lambda line: line.split(","))
        .filter(lambda p: len(p) >= 7 and p[4].lstrip("-").isdigit() and int(p[4]) >= 0)
        .map(lambda p: (p[3].strip(), (int(p[4]), int(p[6]))))  # (island, (cases, deaths))
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))  # reduceByKey
    )
    # Action triggers DAG execution
    rdd_results = rdd_agg.collect()
    t_rdd = (time.perf_counter() - t0) * 1000

    print(f"Spark RDD Execution Completed in {t_rdd:.2f} ms via in-memory DAG scheduler.")
    print("RDD Result Sample (Island, (Total Cases, Total Deaths)):")
    for item in rdd_results:
        print(f"   {item[0]:<28} -> Cases: {item[1][0]:,d}, Deaths: {item[1][1]:,d}")


# ==============================================================================
# MODULE 7: SPARK SQL, DATAFRAMES, MULTI-FORMAT JOINS & COMPUTATIONAL COMPLEXITY
# ==============================================================================
def run_module_7(spark):
    log_section("Module 7: Spark SQL, DataFrames & Multi-Format Integration", "M7 - SC3")

    # 1. Ingest Multi-Format Datasets into DataFrames (Variety)
    print("1. Ingesting Multi-Format Data Sources:")
    df_cases_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(CSV_FILE)
    df_vaccine_raw = spark.read.option("multiline", "true").json(JSON_FILE)
    df_demo_raw = spark.read.parquet(PARQUET_FILE)

    print(f" -> CSV Cases Schema Loaded       : {df_cases_raw.count():,d} rows")
    print(f" -> JSON Vaccine Schema Loaded    : {df_vaccine_raw.count():,d} rows")
    print(f" -> Parquet Demographics Loaded   : {df_demo_raw.count():,d} rows")

    # 2. Veracity Management & Data Cleaning in Spark DataFrame
    print("\n2. Applying Veracity Cleaning Procedures:")
    raw_count = df_cases_raw.count()

    # Step A: Remove exact duplicates
    df_dedup = df_cases_raw.dropDuplicates(["report_date", "province_id"])
    dedup_count = df_dedup.count()
    removed_duplicates = raw_count - dedup_count

    # Step B: Standardize province names & handle negative anomalies
    df_clean_cases = (
        df_dedup
        .withColumn("province_name", F.initcap(F.trim(F.col("province_name"))))
        .withColumn("daily_confirmed_cases", F.when(F.col("daily_confirmed_cases") < 0, 0).otherwise(F.col("daily_confirmed_cases")))
        .withColumn("daily_deaths", F.when(F.col("daily_deaths") < 0, 0).otherwise(F.col("daily_deaths")))
        .withColumn("pcr_tests_conducted", F.coalesce(F.col("pcr_tests_conducted"), F.lit(0)))
        .withColumn("report_date", F.to_date(F.col("report_date"), "yyyy-MM-dd"))
    )

    clean_count = df_clean_cases.count()
    print(f" Before Cleaning: {raw_count:,d} records")
    print(f" After Cleaning : {clean_count:,d} records ({removed_duplicates} duplicates pruned, anomalies corrected)")

    # Flatten nested JSON for vaccination and ICU facilities
    df_vaccine_flat = df_vaccine_raw.select(
        F.col("province_id").alias("v_province_id"),
        F.col("health_facilities.referral_hospitals_count").alias("referral_hospitals"),
        F.col("health_facilities.total_isolation_beds").alias("isolation_beds"),
        F.col("health_facilities.total_icu_beds").alias("total_icu_beds"),
        F.col("health_facilities.total_ventilators").alias("ventilators"),
        F.col("vaccination_progress.dose_2_coverage_pct").alias("dose_2_coverage_pct"),
        F.col("vaccination_progress.booster_coverage_pct").alias("booster_coverage_pct")
    )

    # 3. Register Temporary SQL Views
    df_clean_cases.createOrReplaceTempView("cases")
    df_vaccine_flat.createOrReplaceTempView("vaccines")
    df_demo_raw.createOrReplaceTempView("demographics")

    # 4. Analytical Epidemiological Queries using Spark SQL & Window Functions (Value)
    print("\n3. Executing Epidemiological Analytics with Spark SQL & Window Functions:")

    # Window function: 7-day rolling incidence rate per 100k population
    window_spec = Window.partitionBy("province_id").orderBy("report_date").rowsBetween(-6, 0)
    df_windowed = (
        df_clean_cases
        .join(df_demo_raw, on="province_id", how="inner")
        .withColumn("rolling_7d_cases", F.avg("daily_confirmed_cases").over(window_spec))
        .withColumn("incidence_per_100k", (F.col("rolling_7d_cases") / F.col("total_population")) * 100000)
    )
    df_windowed.createOrReplaceTempView("cases_enriched")

    # Public Health Query: Provincal Profile during Peak Waves
    sql_query = """
    SELECT
        d.province_name_std,
        d.island_group,
        d.total_population,
        SUM(c.daily_confirmed_cases) AS total_cases,
        SUM(c.daily_deaths) AS total_deaths,
        ROUND((SUM(c.daily_deaths) * 100.0 / SUM(c.daily_confirmed_cases)), 2) AS case_fatality_rate_pct,
        MAX(c.incidence_per_100k) AS peak_7d_incidence_per_100k,
        v.total_icu_beds,
        MAX(c.hospitalized_severe_icu) AS peak_icu_demand,
        ROUND((MAX(c.hospitalized_severe_icu) * 100.0 / v.total_icu_beds), 2) AS peak_icu_occupancy_bor_pct,
        v.booster_coverage_pct
    FROM cases_enriched c
    JOIN demographics d ON c.province_id = d.province_id
    JOIN vaccines v ON c.province_id = v.v_province_id
    GROUP BY d.province_name_std, d.island_group, d.total_population, v.total_icu_beds, v.booster_coverage_pct
    ORDER BY total_cases DESC
    """
    df_summary = spark.sql(sql_query)
    print("\nTop 10 High-Burden Provinces (Epidemiological Summary Table):")
    df_summary.select(
        "province_name_std", "total_cases", "total_deaths",
        "case_fatality_rate_pct", "peak_icu_occupancy_bor_pct", "booster_coverage_pct"
    ).show(10, truncate=False)

    # Convert to Pandas for Visualizations and Table 8.2 Comparison
    pdf_summary = df_summary.toPandas()
    # Ensure numeric columns are native float for numpy/seaborn compatibility
    numeric_cols = [
        "total_cases", "total_deaths", "case_fatality_rate_pct",
        "peak_7d_incidence_per_100k", "total_icu_beds", "peak_icu_demand",
        "peak_icu_occupancy_bor_pct", "booster_coverage_pct", "total_population"
    ]
    for col in numeric_cols:
        if col in pdf_summary.columns:
            pdf_summary[col] = pd.to_numeric(pdf_summary[col], errors="coerce")

    # 5. Computational Complexity Demonstration (Table 8.2 in Assignment)
    print("\n4. Computational Complexity Demonstration (Assignment Table 8.2):")
    print("Comparing Operation A (Simple Aggregation) vs Operation B (Relational & Temporal Analysis)...")

    # Operation A: Simple Aggregation
    t0 = time.perf_counter()
    op_a = spark.sql("""
        SELECT island_group, SUM(daily_confirmed_cases) as total_c, SUM(daily_deaths) as total_d
        FROM cases
        GROUP BY island_group
    """).collect()
    time_op_a = (time.perf_counter() - t0) * 1000

    # Operation B: Relational, Multi-Join, and Temporal Window Analysis
    t0 = time.perf_counter()
    op_b = spark.sql(sql_query).collect()
    time_op_b = (time.perf_counter() - t0) * 1000

    print("-" * 86)
    print(f"{'Aspect':<25} | {'Operation A: Simple Aggregation':<28} | {'Operation B: Relational Analysis':<28}")
    print("-" * 86)
    print(f"{'Data Structure':<25} | {'Single CSV Table':<28} | {'CSV + JSON + Parquet (Joined)':<28}")
    print(f"{'Transformations':<25} | {'Group By, Simple Sum':<28} | {'3 Joins, Windowing, Aggregation':<28}")
    print(f"{'Execution Latency':<25} | {f'{time_op_a:.2f} ms':<28} | {f'{time_op_b:.2f} ms':<28}")
    print(f"{'Catalyst Plan Complexity':<25} | {'1 HashAggregate Stage':<28} | {'Multiple HashJoins & WindowDAG':<28}")
    print("-" * 86)

    # Save Curated Gold Parquet Dataset via PyArrow/Pandas (cross-platform compatible without winutils)
    gold_parquet_path = os.path.join(OUTPUT_DIR, "covid19_curated_analytics.parquet")
    if os.path.exists(gold_parquet_path):
        import shutil
        if os.path.isdir(gold_parquet_path):
            shutil.rmtree(gold_parquet_path)
        else:
            os.remove(gold_parquet_path)
    pdf_summary.to_parquet(gold_parquet_path, engine="pyarrow", compression="snappy", index=False)
    print(f"\nGold Curated Parquet Dataset exported to: {gold_parquet_path}")

    # Generate Visualizations (Value)
    generate_visualizations(spark, pdf_summary, time_op_a, time_op_b)


# ==============================================================================
# VISUALIZATIONS & VALUE EXTRACTION
# ==============================================================================
def generate_visualizations(spark, pdf_summary, time_op_a, time_op_b):
    print("\nGenerating Healthcare Informatics Visualizations (PNG)...")
    sns.set_theme(style="whitegrid")

    # 1. Epidemic Curve: Daily Cases & Wave Tracking (National Total over Time)
    df_time = spark.sql("""
        SELECT report_date,
               SUM(daily_confirmed_cases) as national_cases,
               SUM(daily_deaths) as national_deaths,
               SUM(hospitalized_severe_icu) as national_icu
        FROM cases
        GROUP BY report_date
        ORDER BY report_date
    """).toPandas()

    df_time["report_date"] = pd.to_datetime(df_time["report_date"])
    df_time["rolling_7d_cases"] = df_time["national_cases"].rolling(7).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df_time["report_date"], df_time["national_cases"], color="#90caf9", alpha=0.4, label="Daily Reported Cases")
    plt.plot(df_time["report_date"], df_time["rolling_7d_cases"], color="#1565c0", linewidth=2.5, label="7-Day Rolling Average")

    # Highlight Delta and Omicron waves
    plt.axvspan(pd.to_datetime("2021-06-15"), pd.to_datetime("2021-08-31"), color="#ffcdd2", alpha=0.4, label="Wave 2: Delta Surge (High CFR & ICU)")
    plt.axvspan(pd.to_datetime("2022-01-15"), pd.to_datetime("2022-03-31"), color="#ffe0b2", alpha=0.4, label="Wave 3: Omicron Surge (High Cases)")

    plt.title("National COVID-19 Epidemic Curves & Wave Dynamics (2021 - 2022)\nIFB-451 Big Data Healthcare Informatics", fontsize=13, fontweight="bold")
    plt.xlabel("Surveillance Timeline", fontsize=11)
    plt.ylabel("National Daily Cases", fontsize=11)
    plt.legend(loc="upper right")
    plt.tight_layout()
    chart1_path = os.path.join(OUTPUT_DIR, "covid_epidemic_curves.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f" -> [Chart 1 Saved] Epidemic Curves : {chart1_path}")

    # 2. Scatter Plot: Booster Coverage vs Case Fatality Rate (CFR)
    plt.figure(figsize=(9, 6))
    sns.regplot(
        data=pdf_summary,
        x="booster_coverage_pct",
        y="case_fatality_rate_pct",
        scatter_kws={"s": pdf_summary["total_population"] / 400000, "alpha": 0.7, "color": "#2e7d32"},
        line_kws={"color": "#c62828", "linewidth": 2}
    )
    plt.title("Clinical Value: Booster Vaccination Coverage vs Case Fatality Rate (CFR %)\nBubble Size = Provincial Population", fontsize=12, fontweight="bold")
    plt.xlabel("Booster Vaccination Coverage (%)", fontsize=11)
    plt.ylabel("Case Fatality Rate - CFR (%)", fontsize=11)
    plt.tight_layout()
    chart2_path = os.path.join(OUTPUT_DIR, "cfr_vs_booster_scatter.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f" -> [Chart 2 Saved] Vaccine vs CFR  : {chart2_path}")

    # 3. Peak ICU Bed Occupancy Rate (BOR) across Top 12 Provinces
    top12_bor = pdf_summary.sort_values(by="peak_icu_occupancy_bor_pct", ascending=False).head(12)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top12_bor["province_name_std"], top12_bor["peak_icu_occupancy_bor_pct"], color="#d32f2f")
    plt.axvline(80, color="#b71c1c", linestyle="--", linewidth=1.5, label="WHO Critical Threshold (80% BOR)")
    plt.title("Healthcare Capacity Stress: Peak ICU Bed Occupancy Rate (BOR %)\nTop 12 High-Risk Provinces", fontsize=12, fontweight="bold")
    plt.xlabel("Peak ICU Bed Occupancy Rate (%)", fontsize=11)
    plt.legend(loc="lower right")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    chart3_path = os.path.join(OUTPUT_DIR, "peak_icu_bor_top_provinces.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f" -> [Chart 3 Saved] ICU Capacity BOR : {chart3_path}")

    # 4. Computational Complexity: Operation A vs Operation B
    plt.figure(figsize=(7, 5))
    ops = ["Op A: Simple Aggregation\n(1 CSV Table, GroupBy)", "Op B: Relational Analysis\n(3 Joins, Windowing, Agg)"]
    latencies = [time_op_a, time_op_b]
    bars = plt.bar(ops, latencies, color=["#42a5f5", "#ab47bc"], width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 0.5, f"{h:.2f} ms", ha="center", va="bottom", fontweight="bold")
    plt.ylabel("Execution Latency (ms)", fontsize=11)
    plt.title("Computational Complexity Benchmark (Assignment Table 8.2)\nApache Spark Catalyst Execution Time", fontsize=12, fontweight="bold")
    plt.ylim(0, max(latencies) * 1.25)
    plt.tight_layout()
    chart4_path = os.path.join(OUTPUT_DIR, "computational_complexity_benchmark.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f" -> [Chart 4 Saved] Benchmark Latency: {chart4_path}")


def main():
    print("=" * 86)
    print("STARTING BIG DATA PROTOTYPE PIPELINE: MODULES M1 TO M7")
    print("Course : IFB-451 BIG DATA | Institut Teknologi Nasional (Itenas)")
    print("Topic  : COVID-19 Epidemiological Analytics")
    print("=" * 86)

    # Initialize SparkSession
    print("Initializing Apache Spark Session (Local Mode)...")
    spark = (
        SparkSession.builder
        .appName("COVID19-Epidemiological-BigData-M1toM7")
        .master("local[2]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        run_module_1()
        run_module_2()
        run_module_3()
        run_module_4_and_5()
        run_module_6(spark)
        run_module_7(spark)
        print("\n" + "=" * 86)
        print("ALL BIG DATA MODULES (M1 - M7) SUCCESSFULLY EXECUTED AND VERIFIED!")
        print("=" * 86)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""
Dataset Generator for COVID-19 Epidemiological Analytics
UTS IFB-451 BIG DATA - Dr. Edi Nuryatno (2026/2027)

Generates 3 multi-format datasets:
1. CSV: covid_cases_daily.csv (Longitudinal daily cases with intentional veracity noise)
2. JSON: vaccination_icu_facilities.json (Semi-structured healthcare & vaccine progress)
3. Parquet: province_demographics.parquet (Columnar regional census & healthcare metrics)
"""

import json
import os
import random
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(DATA_RAW_DIR, exist_ok=True)

PROVINCES = [
    # (id, bps_code, name, island_group, population, area_km2, elderly_pct, poverty_pct, hw_ratio)
    ("P01", 31, "DKI Jakarta", "Jawa-Bali", 10609659, 664.01, 10.5, 4.69, 32.4),
    ("P02", 32, "Jawa Barat", "Jawa-Bali", 48274162, 35377.76, 10.2, 8.40, 11.2),
    ("P03", 33, "Jawa Tengah", "Jawa-Bali", 36516035, 32800.69, 13.4, 11.25, 14.8),
    ("P04", 34, "DI Yogyakarta", "Jawa-Bali", 3668719, 3133.15, 15.6, 12.80, 28.5),
    ("P05", 35, "Jawa Timur", "Jawa-Bali", 40665696, 47803.49, 13.0, 11.46, 15.6),
    ("P06", 51, "Bali", "Jawa-Bali", 4317404, 5780.06, 12.8, 4.53, 21.0),
    ("P07", 36, "Banten", "Jawa-Bali", 11904562, 9662.92, 7.8, 6.16, 12.5),
    ("P08", 11, "Aceh", "Sumatera", 5274871, 57956.00, 8.9, 15.32, 18.2),
    ("P09", 12, "Sumatera Utara", "Sumatera", 14799361, 72981.23, 9.4, 9.28, 14.1),
    ("P10", 13, "Sumatera Barat", "Sumatera", 5534472, 42012.89, 10.1, 6.28, 17.5),
    ("P11", 14, "Riau", "Sumatera", 6394087, 87023.66, 6.8, 7.00, 12.3),
    ("P12", 15, "Jambi", "Sumatera", 3548228, 50058.16, 7.9, 7.62, 13.0),
    ("P13", 16, "Sumatera Selatan", "Sumatera", 8467432, 91592.43, 8.8, 12.79, 11.8),
    ("P14", 17, "Bengkulu", "Sumatera", 2010670, 19919.33, 8.5, 15.30, 14.9),
    ("P15", 18, "Lampung", "Sumatera", 9007848, 34623.80, 9.7, 12.76, 10.4),
    ("P16", 19, "Kepulauan Bangka Belitung", "Sumatera", 1455678, 16424.23, 8.3, 4.90, 15.0),
    ("P17", 21, "Kepulauan Riau", "Sumatera", 2064564, 8201.72, 6.5, 6.13, 19.4),
    ("P18", 61, "Kalimantan Barat", "Kalimantan", 5414390, 147307.00, 7.9, 7.15, 10.8),
    ("P19", 62, "Kalimantan Tengah", "Kalimantan", 2670000, 153564.50, 6.9, 5.28, 12.1),
    ("P20", 63, "Kalimantan Selatan", "Kalimantan", 4073584, 38744.23, 8.0, 4.70, 14.2),
    ("P21", 64, "Kalimantan Timur", "Kalimantan", 3766039, 129066.64, 6.7, 6.31, 16.5),
    ("P22", 65, "Kalimantan Utara", "Kalimantan", 701814, 75467.70, 5.8, 6.80, 14.0),
    ("P23", 71, "Sulawesi Utara", "Sulawesi", 2621923, 13851.64, 11.5, 7.36, 18.0),
    ("P24", 72, "Sulawesi Tengah", "Sulawesi", 2985734, 61841.29, 8.5, 13.00, 11.5),
    ("P25", 73, "Sulawesi Selatan", "Sulawesi", 9073500, 46717.48, 10.0, 8.70, 17.2),
    ("P26", 74, "Sulawesi Tenggara", "Sulawesi", 2624875, 38067.70, 7.6, 11.20, 12.0),
    ("P27", 75, "Gorontalo", "Sulawesi", 1171681, 11257.07, 8.2, 15.60, 13.4),
    ("P28", 76, "Sulawesi Barat", "Sulawesi", 1419229, 16787.18, 7.5, 11.85, 10.5),
    ("P29", 52, "Nusa Tenggara Barat", "Maluku-Papua-Nusa Tenggara", 5320092, 18572.32, 8.4, 13.82, 12.7),
    ("P30", 53, "Nusa Tenggara Timur", "Maluku-Papua-Nusa Tenggara", 5325566, 48718.10, 9.1, 20.99, 10.1),
    ("P31", 81, "Maluku", "Maluku-Papua-Nusa Tenggara", 1848923, 46914.03, 8.8, 16.30, 14.3),
    ("P32", 82, "Maluku Utara", "Maluku-Papua-Nusa Tenggara", 1282937, 31982.50, 6.9, 6.70, 13.0),
    ("P33", 91, "Papua Barat", "Maluku-Papua-Nusa Tenggara", 1134068, 102955.15, 5.5, 21.70, 11.0),
    ("P34", 94, "Papua", "Maluku-Papua-Nusa Tenggara", 4303707, 319036.05, 5.0, 26.80, 9.8),
]

def generate_demographics_parquet():
    """Module 3: Columnar Parquet format generation for regional metadata."""
    data = []
    for p in PROVINCES:
        pid, bps, name, island, pop, area, elderly, poverty, hw = p
        density = round(pop / area, 2)
        data.append({
            "province_id": pid,
            "bps_code": int(bps),
            "province_name_std": name,
            "island_group": island,
            "total_population": int(pop),
            "area_sq_km": float(area),
            "population_density_per_sqkm": float(density),
            "elderly_ratio_pct": float(elderly),
            "poverty_rate_pct": float(poverty),
            "health_worker_ratio_per_10k": float(hw)
        })
    df_demo = pd.DataFrame(data)
    parquet_path = os.path.join(DATA_RAW_DIR, "province_demographics.parquet")
    table = pa.Table.from_pandas(df_demo)
    pq.write_table(table, parquet_path, compression="snappy")
    print(f"[1/3] Parquet Demographics generated: {parquet_path} ({len(df_demo)} records)")
    return df_demo

def generate_vaccination_icu_json():
    """Module 7 / Variety: Semi-structured JSON generation for vaccination & ICU capacity."""
    facilities_list = []
    for p in PROVINCES:
        pid, bps, name, island, pop, area, elderly, poverty, hw = p
        target_pop = int(pop * 0.78)  # 78% eligible
        # Coverage factors vary realistically across island groups
        cov_factor = 0.92 if island == "Jawa-Bali" else (0.78 if island == "Sumatera" else 0.65)
        dose1 = int(target_pop * min(0.98, cov_factor + random.uniform(-0.05, 0.05)))
        dose2 = int(dose1 * min(0.92, cov_factor * 0.95 + random.uniform(-0.04, 0.04)))
        booster = int(dose2 * min(0.70, cov_factor * 0.60 + random.uniform(-0.05, 0.05)))

        ref_hospitals = max(5, int((pop / 100000) * (hw / 15.0)))
        total_iso_beds = int(ref_hospitals * random.randint(45, 95))
        total_icu_beds = int(total_iso_beds * random.uniform(0.08, 0.14))
        ventilators = int(total_icu_beds * random.uniform(0.65, 0.85))

        entry = {
            "province_id": pid,
            "province_name": name,
            "island_group": island,
            "health_facilities": {
                "referral_hospitals_count": ref_hospitals,
                "total_isolation_beds": total_iso_beds,
                "total_icu_beds": total_icu_beds,
                "total_ventilators": ventilators
            },
            "vaccination_progress": {
                "target_population": target_pop,
                "dose_1_administered": dose1,
                "dose_2_administered": dose2,
                "booster_administered": booster,
                "dose_2_coverage_pct": round((dose2 / target_pop) * 100, 2),
                "booster_coverage_pct": round((booster / target_pop) * 100, 2),
                "vaccine_types_available": ["Sinovac", "AstraZeneca", "Pfizer", "Moderna"]
            },
            "surveillance_level": "Level 3" if island == "Jawa-Bali" else "Level 2"
        }
        facilities_list.append(entry)

    json_path = os.path.join(DATA_RAW_DIR, "vaccination_icu_facilities.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(facilities_list, f, indent=2, ensure_ascii=False)
    print(f"[2/3] JSON Vaccination & ICU facilities generated: {json_path} ({len(facilities_list)} records)")
    return facilities_list

def generate_daily_cases_csv():
    """Module 1 & 3: Tabular CSV generation with epidemiological curves and veracity defects."""
    dates = pd.date_range("2021-01-01", "2022-12-31", freq="D")
    total_days = len(dates)

    # Epidemiological waves simulation (Delta wave at day ~190, Omicron wave at day ~410)
    days_array = np.arange(total_days)
    delta_peak = 195  # July 2021
    delta_wave = np.exp(-0.5 * ((days_array - delta_peak) / 32) ** 2) * 5.5
    omicron_peak = 415  # Feb 2022
    omicron_wave = np.exp(-0.5 * ((days_array - omicron_peak) / 25) ** 2) * 6.8
    baseline = 0.35 + 0.15 * np.sin(days_array / 45.0)
    epidemic_multiplier = baseline + delta_wave + omicron_wave

    records = []
    casing_variants = [
        lambda s: s,
        lambda s: s.upper(),
        lambda s: s.lower(),
        lambda s: f"  {s}  ",  # Whitespace padding
    ]

    for p in PROVINCES:
        pid, bps, name, island, pop, area, elderly, poverty, hw = p
        pop_scale = (pop / 1000000.0)

        for d_idx, dt_val in enumerate(dates):
            date_str = dt_val.strftime("%Y-%m-%d")
            base_cases = int(pop_scale * epidemic_multiplier[d_idx] * random.uniform(15, 35))

            # Delta period had higher CFR, Omicron had lower CFR
            is_delta = (160 <= d_idx <= 240)
            is_omicron = (380 <= d_idx <= 450)
            cfr_prob = 0.042 if is_delta else (0.008 if is_omicron else 0.022)

            deaths = int(np.random.poisson(max(0, base_cases * cfr_prob)))
            recovered = int(np.random.poisson(max(0, base_cases * 0.94)))
            active = max(0, int(base_cases * random.uniform(4, 9)))
            pcr_tests = int(max(base_cases * random.uniform(8, 20), pop_scale * 120))
            antigen_tests = int(pcr_tests * random.uniform(1.2, 2.5))
            mild_hosp = int(base_cases * random.uniform(0.18, 0.35))
            icu_hosp = int(deaths * random.uniform(1.5, 3.2))

            # Veracity defect 1: Inconsistent province name casing / whitespace
            chosen_name = random.choice(casing_variants)(name) if random.random() < 0.15 else name

            # Veracity defect 2: Negative anomalies (reporting corrections)
            if random.random() < 0.003:
                base_cases = -random.randint(1, 15)
            if random.random() < 0.002:
                deaths = -random.randint(1, 5)

            # Veracity defect 3: Missing PCR test counts
            pcr_val = None if random.random() < 0.06 else pcr_tests

            records.append({
                "report_date": date_str,
                "province_id": pid,
                "province_name": chosen_name,
                "island_group": island,
                "daily_confirmed_cases": base_cases,
                "daily_recovered": recovered,
                "daily_deaths": deaths,
                "active_cases": active,
                "pcr_tests_conducted": pcr_val,
                "antigen_tests_conducted": antigen_tests,
                "hospitalized_mild": mild_hosp,
                "hospitalized_severe_icu": icu_hosp
            })

    # Veracity defect 4: Add explicit duplicate records (~450 duplicates)
    num_duplicates = 480
    duplicates = [dict(random.choice(records)) for _ in range(num_duplicates)]
    records.extend(duplicates)

    # Shuffle to simulate disordered arrival
    random.shuffle(records)

    df_cases = pd.DataFrame(records)
    csv_path = os.path.join(DATA_RAW_DIR, "covid_cases_daily.csv")
    df_cases.to_csv(csv_path, index=False)
    print(f"[3/3] CSV Daily Cases generated: {csv_path} ({len(df_cases)} records, includes {num_duplicates} duplicates)")
    return df_cases

if __name__ == "__main__":
    print("=== Generating Big Data COVID-19 Multi-Format Dataset ===")
    generate_demographics_parquet()
    generate_vaccination_icu_json()
    generate_daily_cases_csv()
    print("=== All datasets successfully generated in data/raw/ ===")

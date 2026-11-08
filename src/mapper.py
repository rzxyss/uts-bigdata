#!/usr/bin/env python3
"""
Hadoop Streaming Mapper for COVID-19 Epidemiological Analytics
Module 4 (Hadoop Ecosystem) & Module 5 (MapReduce Programming)

Input: Stdin lines from covid_cases_daily.csv
Output: Key-Value pairs -> <island_group>\t<daily_cases>,<daily_deaths>
"""
import sys

def main():
    for line in sys.stdin:
        # Strip UTF-8 BOM and whitespace
        line = line.lstrip("﻿").strip()
        if not line:
            continue

        parts = line.split(",")
        # Check header row
        if "report_date" in parts[0] or "province_id" in parts:
            continue

        # Format: report_date(0), province_id(1), province_name(2), island_group(3),
        # daily_confirmed_cases(4), daily_recovered(5), daily_deaths(6), ...
        if len(parts) >= 7:
            island_group = parts[3].strip()
            if island_group.lower() == "island_group":
                continue
            try:
                cases = int(parts[4].strip())
                deaths = int(parts[6].strip())
                # Veracity filter in mapper: correct negative anomaly records
                if cases < 0:
                    cases = 0
                if deaths < 0:
                    deaths = 0
                print(f"{island_group}\t{cases},{deaths}")
            except ValueError:
                # Malformed numeric row, skip safely
                continue

if __name__ == "__main__":
    main()

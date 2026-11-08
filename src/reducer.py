#!/usr/bin/env python3
"""
Hadoop Streaming Reducer for COVID-19 Epidemiological Analytics
Module 4 (Hadoop Ecosystem) & Module 5 (MapReduce Programming)

Input: Sorted Key-Value pairs from stdin -> <island_group>\t<daily_cases>,<daily_deaths>
Output: Aggregated totals and Case Fatality Rate (CFR) per island group
"""
import sys

def main():
    current_island = None
    total_cases = 0
    total_deaths = 0

    print("================================================================================")
    print("HADOOP STREAMING REDUCER OUTPUT: COVID-19 REGIONAL TOTALS & FATALITY")
    print("================================================================================")
    print(f"{'Island Group':<30} | {'Total Cases':<14} | {'Total Deaths':<14} | {'CFR (%)':<8}")
    print("-" * 80)

    for line in sys.stdin:
        line = line.lstrip("﻿").strip()
        if not line or "\t" not in line:
            continue

        island, values = line.split("\t", 1)
        island = island.strip()
        val_parts = values.split(",")
        if len(val_parts) != 2:
            continue

        try:
            cases = int(val_parts[0])
            deaths = int(val_parts[1])
        except ValueError:
            continue

        if current_island == island:
            total_cases += cases
            total_deaths += deaths
        else:
            if current_island is not None and (total_cases > 0 or total_deaths > 0):
                cfr = (total_deaths / total_cases * 100.0) if total_cases > 0 else 0.0
                print(f"{current_island:<30} | {total_cases:<14,d} | {total_deaths:<14,d} | {cfr:<8.2f}%")
            current_island = island
            total_cases = cases
            total_deaths = deaths

    # Output last key
    if current_island is not None and (total_cases > 0 or total_deaths > 0):
        cfr = (total_deaths / total_cases * 100.0) if total_cases > 0 else 0.0
        print(f"{current_island:<30} | {total_cases:<14,d} | {total_deaths:<14,d} | {cfr:<8.2f}%")
    print("================================================================================")

if __name__ == "__main__":
    main()

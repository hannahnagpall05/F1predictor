"""
fetch_data.py  (v2 — pulls 1990–2025, predicts for 2026)
=========================================================
Downloads race results and standings from the Jolpi/Ergast API.

Changes from v1:
  - end_year bumped to 2025 (full 2025 season now available in API)
  - train.py cutoff moves to 2023 so 2024+2025 form model's test set
  - predict.py uses 2025 end-of-season stats as the proxy for 2026

2025 season summary (used for predict.py defaults):
  WDC: Lando Norris (McLaren) — edged Verstappen by 2 pts at Abu Dhabi
  WCC: McLaren
  Final race winner: Max Verstappen (Abu Dhabi GP)
"""

import requests
import pandas as pd
import time
import os

BASE_URL = "https://api.jolpi.ca/ergast/f1"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def get_json(url: str, retries=6) -> dict:
    for i in range(retries):
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 429:
                wait = min(2 ** (i + 2), 120)
                print(f"  ⚠️  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            time.sleep(1.5)
            return response.json()
        except requests.exceptions.RequestException as e:
            if i == retries - 1:
                raise RuntimeError(f"Failed after {retries} retries: {url}") from e
            wait = min(2 ** (i + 1), 60)
            print(f"  ⚠️  {type(e).__name__}: retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Rate limit not resolved after {retries} retries: {url}")


def fetch_all_pages(endpoint: str, result_key: str, sub_key: str = "Races") -> list:
    limit, offset, all_items = 100, 0, []
    while True:
        url = f"{BASE_URL}/{endpoint}.json?limit={limit}&offset={offset}"
        data = get_json(url)
        items = data["MRData"][result_key].get(sub_key, [])
        if not items:
            break
        all_items.extend(items)
        total = int(data["MRData"]["total"])
        offset += limit
        if offset >= total:
            break
    return all_items


def fetch_race_results(start_year: int = 1990, end_year: int = 2025):
    rows = []
    for year in range(start_year, end_year + 1):
        print(f"  Fetching {year} race results...")
        races = fetch_all_pages(f"{year}/results", "RaceTable", "Races")
        for race in races:
            for res in race.get("Results", []):
                rows.append({
                    "season":         int(race["season"]),
                    "round":          int(race["round"]),
                    "race_name":      race["raceName"],
                    "circuit_id":     race["Circuit"]["circuitId"],
                    "driver_id":      res["Driver"]["driverId"],
                    "constructor_id": res["Constructor"]["constructorId"],
                    "grid":           int(res.get("grid", 0)),
                    "finish_pos":     int(res.get("position", 0)),
                    "points":         float(res.get("points", 0)),
                    "status":         res.get("status", "Unknown"),
                })
    df = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, "race_results.csv")
    df.to_csv(path, index=False)
    print(f"  ✅ Saved {len(df):,} rows → {path}")
    return df


def fetch_driver_standings(start_year: int = 1990, end_year: int = 2025):
    rows = []
    for year in range(start_year, end_year + 1):
        print(f"  Fetching {year} driver standings...")
        try:
            items = fetch_all_pages(
                f"{year}/driverStandings", "StandingsTable", "StandingsLists"
            )
        except RuntimeError as e:
            print(f"  ⚠️  Skipping {year}: {e}")
            continue
        for standings_list in items:
            rnd = int(standings_list.get("round", 0))
            for entry in standings_list.get("DriverStandings", []):
                rows.append({
                    "season":       year,
                    "round":        rnd,
                    "driver_id":    entry["Driver"]["driverId"],
                    "champ_points": float(entry.get("points", 0)),
                    "champ_pos":    int(entry.get("position", 99)),
                })
    df = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, "driver_standings.csv")
    df.to_csv(path, index=False)
    print(f"  ✅ Saved {len(df):,} rows → {path}")
    return df


def fetch_constructor_standings(start_year: int = 1990, end_year: int = 2025):
    rows = []
    for year in range(start_year, end_year + 1):
        print(f"  Fetching {year} constructor standings...")
        try:
            items = fetch_all_pages(
                f"{year}/constructorStandings", "StandingsTable", "StandingsLists"
            )
        except RuntimeError as e:
            print(f"  ⚠️  Skipping {year}: {e}")
            continue
        for standings_list in items:
            rnd = int(standings_list.get("round", 0))
            for entry in standings_list.get("ConstructorStandings", []):
                rows.append({
                    "season":         year,
                    "round":          rnd,
                    "constructor_id": entry["Constructor"]["constructorId"],
                    "team_points":    float(entry.get("points", 0)),
                    "team_pos":       int(entry.get("position", 99)),
                })
    df = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, "constructor_standings.csv")
    df.to_csv(path, index=False)
    print(f"  ✅ Saved {len(df):,} rows → {path}")
    return df


if __name__ == "__main__":
    print("=== Fetching F1 Data (1990–2025) ===")
    print("  Training on 1990–2023 | Testing on 2024–2025 | Predicting 2026\n")

    print("[1/3] Race Results")
    fetch_race_results(start_year=1990, end_year=2025)

    print("\n[2/3] Driver Standings")
    fetch_driver_standings(start_year=1990, end_year=2025)

    print("\n[3/3] Constructor Standings")
    fetch_constructor_standings(start_year=1990, end_year=2025)

    print("\n✅ Done. Run: features.py → train.py → predict.py")
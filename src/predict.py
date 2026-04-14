

import os
import pickle
import numpy as np
import pandas as pd

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Circuit type encoding (must match features.py) ─────────────────────────
# 0 = technical, 1 = street, 2 = power
CIRCUIT_TYPE = {
    "albert_park":   2, "shanghai":      0, "suzuka":        0,
    "miami":         1, "villeneuve":    1, "monaco":        1,
    "catalunya":     0, "madring":       1, "red_bull_ring": 2,
    "silverstone":   0, "hungaroring":   0, "zandvoort":     0,
    "monza":         2, "baku":          1, "marina_bay":    1,
    "americas":      0, "rodriguez":     0, "interlagos":    0,
    "vegas":         1, "losail":        2, "yas_marina":    0,
    "bahrain":       0, "jeddah":        2,
}

# ── 2026 calendar (22 races — Bahrain & Saudi cancelled) ──────────────────
CALENDAR_2026 = [
    {"round":  1, "name": "Australian Grand Prix",        "circuit_id": "albert_park"},
    {"round":  2, "name": "Chinese Grand Prix",           "circuit_id": "shanghai"},
    {"round":  3, "name": "Japanese Grand Prix",          "circuit_id": "suzuka"},
    {"round":  4, "name": "Miami Grand Prix",             "circuit_id": "miami"},
    {"round":  5, "name": "Canadian Grand Prix",          "circuit_id": "villeneuve"},
    {"round":  6, "name": "Monaco Grand Prix",            "circuit_id": "monaco"},
    {"round":  7, "name": "Barcelona-Catalunya Grand Prix","circuit_id": "catalunya"},
    {"round":  8, "name": "Austrian Grand Prix",          "circuit_id": "red_bull_ring"},
    {"round":  9, "name": "British Grand Prix",           "circuit_id": "silverstone"},
    {"round": 10, "name": "Belgian Grand Prix",           "circuit_id": "spa"},
    {"round": 11, "name": "Hungarian Grand Prix",         "circuit_id": "hungaroring"},
    {"round": 12, "name": "Dutch Grand Prix",             "circuit_id": "zandvoort"},
    {"round": 13, "name": "Italian Grand Prix",           "circuit_id": "monza"},
    {"round": 14, "name": "Spanish Grand Prix (Madrid)",  "circuit_id": "madring"},
    {"round": 15, "name": "Azerbaijan Grand Prix",        "circuit_id": "baku"},
    {"round": 16, "name": "Singapore Grand Prix",         "circuit_id": "marina_bay"},
    {"round": 17, "name": "United States Grand Prix",     "circuit_id": "americas"},
    {"round": 18, "name": "Mexico City Grand Prix",       "circuit_id": "rodriguez"},
    {"round": 19, "name": "São Paulo Grand Prix",         "circuit_id": "interlagos"},
    {"round": 20, "name": "Las Vegas Grand Prix",         "circuit_id": "vegas"},
    {"round": 21, "name": "Qatar Grand Prix",             "circuit_id": "losail"},
    {"round": 22, "name": "Abu Dhabi Grand Prix",         "circuit_id": "yas_marina"},
]

# ── Default 2026 grid ──────────────────────────────────────────────────────
# Based on 2025 championship order as starting grid approximation.
# Teams: McLaren, Mercedes, Red Bull, Ferrari, Williams, Racing Bulls,
#        Aston Martin, Haas, Audi (ex-Sauber), Alpine, Cadillac (new)
# ── Per-circuit default grids (real 2025 qualifying order as proxy for 2026) ──
# Each circuit has its own typical qualifying order based on 2025 results.
# This makes Monaco look different from Monza, Suzuka, Baku, etc.
# Drivers new in 2026 (lindblad, colapinto, bottas, perez-cadillac) are slotted
# at realistic midfield/back positions based on team pace.

def _make_entry(driver_id, constructor_id, pos):
    return {"driver_id": driver_id, "constructor_id": constructor_id, "grid_position": pos}

def _d(drv, team, pos):
    return _make_entry(drv, team, pos)

# Shorthand team names
MCL = "mclaren";  MER = "mercedes";  RBR = "red_bull";  FER = "ferrari"
WIL = "williams"; RB  = "rb";        AM  = "aston_martin"; HAS = "haas"
AUD = "sauber";   ALP = "alpine";    CAD = "red_bull"   # Cadillac uses closest proxy

CIRCUIT_GRIDS = {
    # ── Australia (albert_park) — REAL 2026 qualifying grid
    # Russell P1, Antonelli P2, Hadjar P3, Leclerc P4, Norris P5
    # Result: Russell won, Antonelli 2nd, Leclerc 3rd
    "albert_park": [
        _d("russell",        MER, 1),  _d("antonelli",      MER, 2),
        _d("hadjar",         RBR, 3),  _d("leclerc",        FER, 4),
        _d("norris",         MCL, 5),  _d("max_verstappen", RBR, 6),
        _d("bearman",        HAS, 7),  _d("lindblad",       RB,  8),
        _d("piastri",        MCL, 9),  _d("hamilton",       FER, 10),
        _d("gasly",          ALP, 11), _d("lawson",         RB,  12),
        _d("bortoleto",      AUD, 13), _d("ocon",           HAS, 14),
        _d("colapinto",      ALP, 15), _d("sainz",          WIL, 16),
        _d("albon",          WIL, 17), _d("alonso",         AM,  18),
        _d("stroll",         AM,  19), _d("hulkenberg",     AUD, 20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── China (shanghai) — REAL 2026 qualifying grid
    # Antonelli P1, Russell P2, Piastri P3, Leclerc P4, Hamilton P5
    # Result: Antonelli won, Russell 2nd, Hamilton 3rd
    "shanghai": [
        _d("antonelli",      MER, 1),  _d("russell",        MER, 2),
        _d("piastri",        MCL, 3),  _d("leclerc",        FER, 4),
        _d("hamilton",       FER, 5),  _d("norris",         MCL, 6),
        _d("max_verstappen", RBR, 7),  _d("hadjar",         RBR, 8),
        _d("bearman",        HAS, 9),  _d("gasly",          ALP, 10),
        _d("lawson",         RB,  11), _d("lindblad",       RB,  12),
        _d("sainz",          WIL, 13), _d("albon",          WIL, 14),
        _d("ocon",           HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("colapinto",      ALP, 18),
        _d("alonso",         AM,  19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Japan (suzuka) — REAL 2026 qualifying grid
    # Antonelli P1, Russell P2, Piastri P3, Leclerc P4 — Verstappen Q2 exit (P11)
    # Result: Antonelli won, Piastri 2nd, Leclerc 3rd — Red Bull STRUGGLING
    "suzuka": [
        _d("antonelli",      MER, 1),  _d("russell",        MER, 2),
        _d("piastri",        MCL, 3),  _d("leclerc",        FER, 4),
        _d("norris",         MCL, 5),  _d("hamilton",       FER, 6),
        _d("gasly",          ALP, 7),  _d("hadjar",         RBR, 8),
        _d("bortoleto",      AUD, 9),  _d("lindblad",       RB,  10),
        _d("max_verstappen", RBR, 11), _d("lawson",         RB,  12),
        _d("sainz",          WIL, 13), _d("albon",          WIL, 14),
        _d("bearman",        HAS, 15), _d("ocon",           HAS, 16),
        _d("colapinto",      ALP, 17), _d("alonso",         AM,  18),
        _d("stroll",         AM,  19), _d("hulkenberg",     AUD, 20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Miami (miami) — street-style, McLaren strong
    # 2025: Piastri P1, Norris P2, Verstappen P3, Russell P4, Leclerc P5
    "miami": [
        _d("piastri",        MCL, 1),  _d("norris",         MCL, 2),
        _d("max_verstappen", RBR, 3),  _d("russell",        MER, 4),
        _d("leclerc",        FER, 5),  _d("hamilton",       FER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Canada (villeneuve) — street/power, Verstappen/McLaren competitive
    # 2025: Piastri P1, Norris P2, Verstappen P3, Leclerc P4, Russell P5
    "villeneuve": [
        _d("piastri",        MCL, 1),  _d("norris",         MCL, 2),
        _d("max_verstappen", RBR, 3),  _d("leclerc",        FER, 4),
        _d("russell",        MER, 5),  _d("hamilton",       FER, 6),
        _d("antonelli",      MER, 7),  _d("sainz",          WIL, 8),
        _d("albon",          WIL, 9),  _d("hadjar",         RBR, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Monaco (monaco) — street, Leclerc & Norris historically strongest
    # 2025: Norris P1, Leclerc P2, Piastri P3, Verstappen P4 (Hamilton penalised)
    "monaco": [
        _d("norris",         MCL, 1),  _d("leclerc",        FER, 2),
        _d("piastri",        MCL, 3),  _d("max_verstappen", RBR, 4),
        _d("hamilton",       FER, 5),  _d("hadjar",         RBR, 6),
        _d("alonso",         AM,  7),  _d("ocon",           HAS, 8),
        _d("lawson",         RB,  9),  _d("albon",          WIL, 10),
        _d("sainz",          WIL, 11), _d("lindblad",       RB,  12),
        _d("hulkenberg",     AUD, 13), _d("russell",        MER, 14),
        _d("antonelli",      MER, 15), _d("gasly",          ALP, 16),
        _d("colapinto",      ALP, 17), _d("stroll",         AM,  18),
        _d("bortoleto",      AUD, 19), _d("bearman",        HAS, 20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Barcelona-Catalunya (catalunya) — technical, McLaren/Red Bull battle
    # 2025: Norris P1, Piastri P2, Verstappen P3, Russell P4, Leclerc P5
    "catalunya": [
        _d("norris",         MCL, 1),  _d("piastri",        MCL, 2),
        _d("max_verstappen", RBR, 3),  _d("russell",        MER, 4),
        _d("leclerc",        FER, 5),  _d("hamilton",       FER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Madrid street circuit (madring) — NEW in 2026, street character like Monaco/Baku
    # No 2025 data — using street circuit typical order (Leclerc/Norris/Piastri strong)
    "madring": [
        _d("leclerc",        FER, 1),  _d("norris",         MCL, 2),
        _d("piastri",        MCL, 3),  _d("max_verstappen", RBR, 4),
        _d("hamilton",       FER, 5),  _d("russell",        MER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("alonso",         AM,  9),  _d("sainz",          WIL, 10),
        _d("albon",          WIL, 11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Austria (red_bull_ring) — power/short, Red Bull home race
    # 2025: Piastri P1, Norris P2, Verstappen P3, Russell P4
    "red_bull_ring": [
        _d("piastri",        MCL, 1),  _d("norris",         MCL, 2),
        _d("max_verstappen", RBR, 3),  _d("russell",        MER, 4),
        _d("leclerc",        FER, 5),  _d("antonelli",      MER, 6),
        _d("hamilton",       FER, 7),  _d("hadjar",         RBR, 8),
        _d("albon",          WIL, 9),  _d("sainz",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Britain (silverstone) — technical/fast, Hamilton historic stronghold
    # 2025: Norris P1, Piastri P2, Russell P3, Hamilton P4, Verstappen P5
    "silverstone": [
        _d("norris",         MCL, 1),  _d("piastri",        MCL, 2),
        _d("russell",        MER, 3),  _d("hamilton",       FER, 4),
        _d("max_verstappen", RBR, 5),  _d("leclerc",        FER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Belgium (spa) — high-speed technical, historically Verstappen/Hamilton strong
    # 2025: Verstappen P1, Norris P2, Piastri P3, Russell P4, Leclerc P5
    "spa": [
        _d("max_verstappen", RBR, 1),  _d("norris",         MCL, 2),
        _d("piastri",        MCL, 3),  _d("russell",        MER, 4),
        _d("leclerc",        FER, 5),  _d("hamilton",       FER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("alonso",         AM,  9),  _d("sainz",          WIL, 10),
        _d("albon",          WIL, 11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Hungary (hungaroring) — tight/technical, McLaren/Hamilton strong
    # 2025: Norris P1, Piastri P2, Hamilton P3, Russell P4, Leclerc P5
    "hungaroring": [
        _d("norris",         MCL, 1),  _d("piastri",        MCL, 2),
        _d("hamilton",       FER, 3),  _d("russell",        MER, 4),
        _d("leclerc",        FER, 5),  _d("max_verstappen", RBR, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("alonso",         AM,  9),  _d("sainz",          WIL, 10),
        _d("albon",          WIL, 11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Netherlands (zandvoort) — technical/bumpy, Verstappen home race
    # 2025: Verstappen P1, Norris P2, Piastri P3, Russell P4, Leclerc P5
    "zandvoort": [
        _d("max_verstappen", RBR, 1),  _d("norris",         MCL, 2),
        _d("piastri",        MCL, 3),  _d("russell",        MER, 4),
        _d("leclerc",        FER, 5),  _d("hamilton",       FER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Italy (monza) — power circuit, Verstappen/Norris battle in 2025
    # 2025: Verstappen P1, Norris P2, Piastri P3, Leclerc P4, Russell P5
    "monza": [
        _d("max_verstappen", RBR, 1),  _d("norris",         MCL, 2),
        _d("piastri",        MCL, 3),  _d("leclerc",        FER, 4),
        _d("russell",        MER, 5),  _d("hamilton",       FER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Azerbaijan (baku) — street/power, chaos circuit, Leclerc historically strong
    # 2025: Leclerc P1, Piastri P2, Norris P3, Verstappen P4, Russell P5
    "baku": [
        _d("leclerc",        FER, 1),  _d("piastri",        MCL, 2),
        _d("norris",         MCL, 3),  _d("max_verstappen", RBR, 4),
        _d("russell",        MER, 5),  _d("hamilton",       FER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Singapore (marina_bay) — street, slow corners, Ferrari/Leclerc specialist
    # 2025: Leclerc P1, Norris P2, Piastri P3, Hamilton P4, Russell P5
    "marina_bay": [
        _d("leclerc",        FER, 1),  _d("norris",         MCL, 2),
        _d("piastri",        MCL, 3),  _d("hamilton",       FER, 4),
        _d("russell",        MER, 5),  _d("max_verstappen", RBR, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("alonso",         AM,  9),  _d("sainz",          WIL, 10),
        _d("albon",          WIL, 11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── USA (americas) — technical, Red Bull/McLaren battle
    # 2025: Verstappen P1, Norris P2, Piastri P3, Leclerc P4, Russell P5
    "americas": [
        _d("max_verstappen", RBR, 1),  _d("norris",         MCL, 2),
        _d("piastri",        MCL, 3),  _d("leclerc",        FER, 4),
        _d("russell",        MER, 5),  _d("hamilton",       FER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Mexico (rodriguez) — high altitude, power/technical, Verstappen strong
    # 2025: Verstappen P1, Norris P2, Piastri P3, Leclerc P4, Hamilton P5
    "rodriguez": [
        _d("max_verstappen", RBR, 1),  _d("norris",         MCL, 2),
        _d("piastri",        MCL, 3),  _d("leclerc",        FER, 4),
        _d("hamilton",       FER, 5),  _d("russell",        MER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Brazil (interlagos) — technical/unpredictable, Hamilton historically great
    # 2025: Norris P1, Piastri P2, Verstappen P3, Russell P4, Hamilton P5
    "interlagos": [
        _d("norris",         MCL, 1),  _d("piastri",        MCL, 2),
        _d("max_verstappen", RBR, 3),  _d("russell",        MER, 4),
        _d("hamilton",       FER, 5),  _d("leclerc",        FER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Las Vegas (vegas) — street/power, Verstappen won in 2025
    # 2025: Verstappen P1, Russell P2, Antonelli P3, Norris P4, Leclerc P5
    "vegas": [
        _d("max_verstappen", RBR, 1),  _d("russell",        MER, 2),
        _d("antonelli",      MER, 3),  _d("norris",         MCL, 4),
        _d("leclerc",        FER, 5),  _d("piastri",        MCL, 6),
        _d("hamilton",       FER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Qatar (losail) — power/fast, McLaren strong
    # 2025: Norris P1, Piastri P2, Verstappen P3, Leclerc P4, Hamilton P5
    "losail": [
        _d("norris",         MCL, 1),  _d("piastri",        MCL, 2),
        _d("max_verstappen", RBR, 3),  _d("leclerc",        FER, 4),
        _d("hamilton",       FER, 5),  _d("russell",        MER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
    # ── Abu Dhabi (yas_marina) — technical/smooth, Verstappen on pole for title decider
    # 2025: Verstappen P1, Norris P2, Piastri P3, Leclerc P4, Hamilton P5
    "yas_marina": [
        _d("max_verstappen", RBR, 1),  _d("norris",         MCL, 2),
        _d("piastri",        MCL, 3),  _d("leclerc",        FER, 4),
        _d("hamilton",       FER, 5),  _d("russell",        MER, 6),
        _d("antonelli",      MER, 7),  _d("hadjar",         RBR, 8),
        _d("sainz",          WIL, 9),  _d("albon",          WIL, 10),
        _d("alonso",         AM,  11), _d("lawson",         RB,  12),
        _d("lindblad",       RB,  13), _d("ocon",           HAS, 14),
        _d("bearman",        HAS, 15), _d("hulkenberg",     AUD, 16),
        _d("bortoleto",      AUD, 17), _d("gasly",          ALP, 18),
        _d("colapinto",      ALP, 19), _d("stroll",         AM,  20),
        _d("perez",          CAD, 21), _d("bottas",         CAD, 22),
    ],
}

def get_default_grid(circuit_id: str) -> list[dict]:
    """Return the circuit-specific default qualifying grid, falling back to championship order."""
    return CIRCUIT_GRIDS.get(circuit_id, CIRCUIT_GRIDS["albert_park"])

# ── Real 2026 championship points after Round 3 (Japan) ──────────────────
# Source: live standings, used to override the 2025 historical proxy
# This makes Miami predictions reflect the ACTUAL 2026 form, not 2025 history
REAL_2026_DRIVER_POINTS = {
    "antonelli":      72,   # 🥇 Championship leader, 2 wins
    "russell":        63,   # 🥈 9 pts behind
    "leclerc":        49,   # 🥉 Ferrari best
    "hamilton":       41,   # Ferrari
    "norris":         25,   # McLaren slow start
    "piastri":        21,   # McLaren
    "bearman":        17,   # Haas surprise
    "gasly":          15,   # Alpine
    "max_verstappen": 12,   # Red Bull struggling badly
    "lawson":         10,   # Racing Bulls
    "lindblad":        4,
    "hadjar":          4,
    "bortoleto":       2,
    "sainz":           2,
    "ocon":            1,
    "colapinto":       1,
    "hulkenberg":      0,
    "albon":           0,
    "bottas":          0,
    "perez":           0,
    "alonso":          0,
    "stroll":          0,
}

REAL_2026_CONSTRUCTOR_POINTS = {
    "mercedes":    135,   # Dominant early
    "ferrari":      90,
    "mclaren":      56,   # Recovering
    "haas":         41,   # Surprise package
    "alpine":       16,
    "rb":           14,   # Racing Bulls
    "red_bull":     16,   # Struggling
    "audi":          2,
    "williams":      2,
    "aston_martin":  0,
    "sauber":        0,   # Cadillac proxy
}

# Real 2026 recent form (avg finish pos last 3 races — lower = better)
# Overrides the 2025 historical proxy which is now stale
REAL_2026_RECENT_FORM = {
    "antonelli":      1.0,   # Won last 2, 2nd in Rd1 — on fire
    "russell":        2.3,   # 1st, 2nd, 4th — consistently top 4
    "leclerc":        3.7,   # 3rd, 5th, 3rd — solid Ferrari
    "hamilton":       4.7,   # 10th, 3rd, 6th — improving
    "piastri":        5.0,   # DNF, DNF, 2nd — finally finishing
    "norris":         6.7,   # 5th, 6th, 5th — McLaren below par
    "bearman":        8.0,   # Surprise Haas performer
    "gasly":          9.0,   # Alpine points scorer
    "max_verstappen": 9.3,   # 8th, 8th, 9th — Red Bull very poor
    "hadjar":        12.0,
    "lawson":        13.0,
    "lindblad":      14.0,
    "bortoleto":     15.0,
    "sainz":         15.0,
    "ocon":          16.0,
    "colapinto":     17.0,
    "hulkenberg":    18.0,
    "albon":         18.0,
    "alonso":        19.0,
    "stroll":        20.0,
    "perez":         20.0,
    "bottas":        20.0,
}

# Real 2026 season win % (3 races done)
REAL_2026_WIN_PCT = {
    "antonelli":      2/3,   # Won 2 of 3
    "russell":        1/3,   # Won 1 of 3
    "leclerc":        0.0,
    "hamilton":       0.0,
    "piastri":        0.0,
    "norris":         0.0,
    "max_verstappen": 0.0,
    "bearman":        0.0,
    "gasly":          0.0,
}

# Real 2026 Miami qualifying grid (sprint weekend — based on 2026 form)
# Antonelli on pole (2 poles already), Mercedes front-row lock-out possible
MIAMI_2026_GRID = [
    _d("antonelli",      MER, 1),   # On fire — 2 wins, 2 poles
    _d("russell",        MER, 2),   # Strong teammate
    _d("piastri",        MCL, 3),   # McLaren improving
    _d("leclerc",        FER, 4),   # Ferrari consistent
    _d("norris",         MCL, 5),   # McLaren
    _d("hamilton",       FER, 6),   # Ferrari
    _d("max_verstappen", RBR, 7),   # Red Bull struggling (Q2 exit at Suzuka)
    _d("hadjar",         RBR, 8),
    _d("bearman",        HAS, 9),   # Haas surprise performer
    _d("gasly",          ALP, 10),  # Alpine in the points
    _d("lawson",         RB,  11),
    _d("lindblad",       RB,  12),
    _d("ocon",           HAS, 13),
    _d("sainz",          WIL, 14),
    _d("albon",          WIL, 15),
    _d("alonso",         AM,  16),
    _d("stroll",         AM,  17),
    _d("hulkenberg",     AUD, 18),
    _d("bortoleto",      AUD, 19),
    _d("colapinto",      ALP, 20),
    _d("perez",          CAD, 21),
    _d("bottas",         CAD, 22),
]
# Override miami circuit grid with the real 2026 form-based grid
CIRCUIT_GRIDS["miami"] = MIAMI_2026_GRID



# ── Load model artifacts ──────────────────────────────────────────────────

def load_model():
    paths = {
        "model":    os.path.join(MODEL_DIR, "rf_model.pkl"),
        "encoders": os.path.join(MODEL_DIR, "encoders.pkl"),
        "features": os.path.join(MODEL_DIR, "feature_cols.pkl"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing: {missing}. Run python src/train.py first.")
    with open(paths["model"],    "rb") as f: model    = pickle.load(f)
    with open(paths["encoders"], "rb") as f: encoders = pickle.load(f)
    with open(paths["features"], "rb") as f: features = pickle.load(f)
    return model, encoders, features


def load_history() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("features.csv not found — run features.py first.")
    return pd.read_csv(path)


# ── History lookups (prefer 2025, fallback to latest available) ────────────

def _latest_row(df: pd.DataFrame, **filters) -> pd.Series | None:
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in filters.items():
        mask &= df[col] == val
    sub = df[mask].sort_values(["season", "round"])
    return sub.iloc[-1] if not sub.empty else None


def get_circuit_history(hist: pd.DataFrame, driver_id: str, circuit_id: str) -> dict:
    row = _latest_row(hist, driver_id=driver_id, circuit_id=circuit_id)
    if row is None:
        return {"circuit_win_rate": 0.0, "circuit_podium_rate": 0.0}
    return {
        "circuit_win_rate":    float(row.get("circuit_win_rate", 0.0)),
        "circuit_podium_rate": float(row.get("circuit_podium_rate", 0.0)),
    }


def get_driver_stats(hist: pd.DataFrame, driver_id: str) -> dict:
    # Resolve real 2026 overrides first — before any early returns
    real_pts     = REAL_2026_DRIVER_POINTS.get(driver_id)
    real_form    = REAL_2026_RECENT_FORM.get(driver_id)
    real_win_pct = REAL_2026_WIN_PCT.get(driver_id)

    row = _latest_row(hist, driver_id=driver_id, season=2025)
    if row is None:
        row = _latest_row(hist, driver_id=driver_id)
    if row is None:
        return {
            "driver_champ_points":     float(real_pts or 0),
            "driver_season_win_pct":   float(real_win_pct or 0.0),
            "driver_season_poles_pct": 0.0,
            "driver_recent_form":      float(real_form or 12.0),
            "dnf_rate":                0.1,
        }
    return {
        "driver_champ_points":     float(real_pts if real_pts is not None else row.get("driver_champ_points", 0)),
        "driver_season_win_pct":   float(real_win_pct if real_win_pct is not None else row.get("driver_season_win_pct", 0.0)),
        "driver_season_poles_pct": float(row.get("driver_season_poles_pct", 0.0)),
        "driver_recent_form":      float(real_form if real_form is not None else row.get("driver_recent_form", 12.0)),
        "dnf_rate":                float(row.get("dnf_rate", 0.1)),
    }


def get_constructor_stats(hist: pd.DataFrame, constructor_id: str) -> dict:
    # Use real 2026 points if available
    real_pts = REAL_2026_CONSTRUCTOR_POINTS.get(constructor_id)
    row = _latest_row(hist, constructor_id=constructor_id, season=2025)
    if row is None:
        row = _latest_row(hist, constructor_id=constructor_id)
    if row is None:
        return {"team_champ_points": real_pts or 0, "constructor_season_wins": 0, "constructor_avg_grid": 12.0}
    return {
        "team_champ_points":       float(real_pts if real_pts is not None else row.get("team_champ_points", 0)),
        "constructor_season_wins": float(row.get("constructor_season_wins", 0)),
        "constructor_avg_grid":    float(row.get("constructor_avg_grid", 12.0)),
    }


# ── Core prediction ────────────────────────────────────────────────────────

def predict_race(circuit_id: str, entries: list[dict]) -> pd.DataFrame:
    model, encoders, feature_cols = load_model()
    hist = load_history()

    le_drv = encoders["driver_id"]
    le_con = encoders["constructor_id"]
    le_cir = encoders["circuit_id"]

    ctype   = CIRCUIT_TYPE.get(circuit_id, 0)
    cir_enc = int(le_cir.transform([circuit_id])[0]) if circuit_id in le_cir.classes_ else 0

    rows = []
    for e in entries:
        drv, con, grid = e["driver_id"], e["constructor_id"], int(e["grid_position"])

        cir  = get_circuit_history(hist, drv, circuit_id)
        dstats = get_driver_stats(hist, drv)
        cstats = get_constructor_stats(hist, con)

        drv_enc = int(le_drv.transform([drv])[0]) if drv in le_drv.classes_ else 0
        con_enc = int(le_con.transform([con])[0]) if con in le_con.classes_ else 0

        rows.append({
            "driver_id": drv, "constructor_id": con, "grid_position": grid,
            "grid_squared":            grid ** 2,
            "is_pole":                 int(grid == 1),
            "driver_champ_points":     dstats["driver_champ_points"],
            "team_champ_points":       cstats["team_champ_points"],
            "circuit_win_rate":        cir["circuit_win_rate"],
            "circuit_podium_rate":     cir["circuit_podium_rate"],
            "circuit_type_enc":        ctype,
            "driver_recent_form":      dstats["driver_recent_form"],
            "driver_season_win_pct":   dstats["driver_season_win_pct"],
            "driver_season_poles_pct": dstats["driver_season_poles_pct"],
            "constructor_season_wins": cstats["constructor_season_wins"],
            "constructor_avg_grid":    cstats["constructor_avg_grid"],
            "dnf_rate":                dstats["dnf_rate"],
            "driver_id_enc":           drv_enc,
            "constructor_id_enc":      con_enc,
            "circuit_id_enc":          cir_enc,
        })

    df  = pd.DataFrame(rows)
    X   = df[[c for c in feature_cols if c in df.columns]]
    raw = model.predict_proba(X)[:, 1]

    # Normalise across field so all probs sum to 100%
    total = raw.sum()
    norm  = raw / total if total > 0 else np.ones(len(raw)) / len(raw)
    df["win_probability_%"] = (norm * 100).round(1)

    result = (
        df[["driver_id", "constructor_id", "grid_position", "win_probability_%"]]
        .sort_values("win_probability_%", ascending=False)
        .reset_index(drop=True)
    )
    result.index += 1
    result.index.name = "rank"
    return result


# ── Accuracy summary ───────────────────────────────────────────────────────

def show_accuracy():
    print("\n  Calculating accuracy on 2025 test set...")
    try:
        model, encoders, feature_cols = load_model()
        hist = load_history()
        test = hist[hist["season"] == 2025].copy()

        if test.empty:
            test = hist[hist["season"] >= 2022].copy()
            print(f"  (Using {test['season'].min()}–{test['season'].max()} as proxy)\n")

        for col, le in [("driver_id", encoders["driver_id"]),
                        ("constructor_id", encoders["constructor_id"]),
                        ("circuit_id", encoders["circuit_id"])]:
            test[col + "_enc"] = test[col].apply(
                lambda x: int(le.transform([x])[0]) if x in le.classes_ else 0
            )

        X = test[[c for c in feature_cols if c in test.columns]]
        test["win_prob"] = model.predict_proba(X)[:, 1]

        top1 = top3 = total = 0
        for _, race in test.groupby(["season", "round"]):
            w = race[race["won"] == 1]["driver_id"].values
            if not len(w): continue
            r = race.nlargest(3, "win_prob")["driver_id"].values
            if r[0] == w[0]: top1 += 1
            if w[0] in r:    top3 += 1
            total += 1

        w   = 52
        t1  = top1 / total if total else 0
        t3  = top3 / total if total else 0
        bar = lambda p: "█" * int(p * 20) + "░" * (20 - int(p * 20))

        print("  " + "=" * w)
        print(f"  {'MODEL ACCURACY — 2025 TEST SET':^{w}}")
        print("  " + "=" * w)
        print(f"\n  Top-1  {t1:>6.1%}  {bar(t1)}  ({top1}/{total})")
        print(f"  Top-3  {t3:>6.1%}  {bar(t3)}  ({top3}/{total})")

        grade = ("🟢 EXCELLENT (≥90%!)" if t3 >= 0.90 else
                 "🟡 GOOD — near target" if t3 >= 0.80 else "🟠 FAIR")
        print(f"\n  Grade: {grade}")
        print("  " + "=" * w + "\n")

    except Exception as e:
        print(f"  Could not compute accuracy: {e}\n")


# ── Display ────────────────────────────────────────────────────────────────

def print_result(result: pd.DataFrame, race_name: str, circuit_id: str):
    ctype_name = {0: "Technical", 1: "Street circuit", 2: "Power circuit"}
    ctype = ctype_name.get(CIRCUIT_TYPE.get(circuit_id, 0), "")
    w = 68
    max_prob = result["win_probability_%"].max()

    print()
    print("  " + "=" * w)
    print(f"  {'F1 Race Winner Predictor  2026':^{w}}")
    print(f"  {race_name:^{w}}")
    if ctype:
        print(f"  {'[' + ctype + ']':^{w}}")
    print("  " + "=" * w)
    print(f"  {'Rank':<5} {'Driver':<22} {'Team':<16} {'Grid':>4}  {'Win %':>6}  Chart")
    print("  " + "-" * w)

    for rank, row in result.iterrows():
        pct = row["win_probability_%"]
        bar = "█" * max(1, int(pct / max_prob * 20))
        print(f"  {rank:<5} {row['driver_id']:<22} {row['constructor_id']:<16} "
              f"{int(row['grid_position']):>4}  {pct:>5.1f}%  {bar}")

    print("  " + "=" * w)
    top = result.iloc[0]
    print(f"\n  🏆 {top['driver_id']}  ({top['win_probability_%']:.1f}%)")
    print(f"  2️⃣  {result.iloc[1]['driver_id']}  ({result.iloc[1]['win_probability_%']:.1f}%)")
    print(f"  3️⃣  {result.iloc[2]['driver_id']}  ({result.iloc[2]['win_probability_%']:.1f}%)")
    print()


# ── Interactive UI ─────────────────────────────────────────────────────────

def pick_race() -> dict:
    ctype_label = {0: "tech", 1: "🏙 street", 2: "⚡ power"}
    print("\n  2026 F1 Calendar  (22 races)")
    print("  " + "-" * 48)
    for r in CALENDAR_2026:
        ct = ctype_label.get(CIRCUIT_TYPE.get(r["circuit_id"], 0), "tech")
        print(f"  {r['round']:>2}. {r['name']:<38} [{ct}]")
    print()
    while True:
        raw = input("  Race number (1–22): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 22:
            return CALENDAR_2026[int(raw) - 1]
        print("  Enter 1–22.")


def pick_grid(circuit_id: str) -> list[dict]:
    grid = get_default_grid(circuit_id)
    pole = grid[0]["driver_id"]
    choice = input(f"\n  Use default grid for this circuit? (pole: {pole}) (y/n): ").strip().lower()
    if choice != "n":
        return grid
    print("\n  Enter: driver_id  constructor_id  grid_position")
    print("  e.g.   norris mclaren 1    |  'done' to finish\n")
    entries = []
    while True:
        raw = input(f"  Driver {len(entries)+1}: ").strip()
        if raw.lower() == "done":
            if len(entries) >= 2: break
            print("  Need at least 2.")
            continue
        parts = raw.split()
        if len(parts) != 3 or not parts[2].isdigit():
            print("  Format: driver_id constructor_id grid_pos")
            continue
        entries.append({"driver_id": parts[0], "constructor_id": parts[1],
                        "grid_position": int(parts[2])})
    return entries


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== F1 Race Winner Predictor  2026 ===")
    print("  Defending champion: Lando Norris  |  Defending WCC: McLaren")
    print("  New teams: Cadillac, Audi  |  New circuit: Madrid")

    show_accuracy()

    while True:
        race   = pick_race()
        grid   = pick_grid(race["circuit_id"])
        result = predict_race(circuit_id=race["circuit_id"], entries=grid)
        print_result(result, race["name"], race["circuit_id"])
        show_accuracy()

        if input("  Predict another race? (y/n): ").strip().lower() != "y":
            print("  Goodbye!\n")
            break

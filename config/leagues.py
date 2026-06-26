"""
League configuration — all 24 leagues with API-Football IDs.
Used by every data collection and model script.
"""

LEAGUES = [
    # ── England ──────────────────────────────────────────────────────────────
    {"id": 39,  "name": "Premier League",        "country": "England",     "div": 1, "tier": "big5"},
    {"id": 40,  "name": "Championship",           "country": "England",     "div": 2, "tier": "big5"},

    # ── Spain ─────────────────────────────────────────────────────────────────
    {"id": 140, "name": "La Liga",                "country": "Spain",       "div": 1, "tier": "big5"},
    {"id": 141, "name": "La Liga 2",              "country": "Spain",       "div": 2, "tier": "big5"},

    # ── Germany ───────────────────────────────────────────────────────────────
    {"id": 78,  "name": "Bundesliga",             "country": "Germany",     "div": 1, "tier": "big5"},
    {"id": 79,  "name": "2. Bundesliga",          "country": "Germany",     "div": 2, "tier": "big5"},

    # ── Italy ─────────────────────────────────────────────────────────────────
    {"id": 135, "name": "Serie A",                "country": "Italy",       "div": 1, "tier": "big5"},
    {"id": 136, "name": "Serie B",                "country": "Italy",       "div": 2, "tier": "big5"},

    # ── France ────────────────────────────────────────────────────────────────
    {"id": 61,  "name": "Ligue 1",                "country": "France",      "div": 1, "tier": "big5"},
    {"id": 62,  "name": "Ligue 2",                "country": "France",      "div": 2, "tier": "big5"},

    # ── Netherlands ───────────────────────────────────────────────────────────
    {"id": 88,  "name": "Eredivisie",             "country": "Netherlands", "div": 1, "tier": "extra_eu"},
    {"id": 89,  "name": "Eerste Divisie",         "country": "Netherlands", "div": 2, "tier": "extra_eu"},

    # ── Portugal ──────────────────────────────────────────────────────────────
    {"id": 94,  "name": "Primeira Liga",          "country": "Portugal",    "div": 1, "tier": "extra_eu"},
    {"id": 95,  "name": "Liga Portugal 2",        "country": "Portugal",    "div": 2, "tier": "extra_eu"},

    # ── Turkey ────────────────────────────────────────────────────────────────
    {"id": 203, "name": "Süper Lig",              "country": "Turkey",      "div": 1, "tier": "extra_eu"},
    {"id": 204, "name": "TFF First League",       "country": "Turkey",      "div": 2, "tier": "extra_eu"},

    # ── Belgium ───────────────────────────────────────────────────────────────
    {"id": 144, "name": "Pro League",             "country": "Belgium",     "div": 1, "tier": "extra_eu"},
    {"id": 145, "name": "First Amateur Division", "country": "Belgium",     "div": 2, "tier": "extra_eu"},

    # ── Austria ───────────────────────────────────────────────────────────────
    {"id": 218, "name": "Austrian Bundesliga",    "country": "Austria",     "div": 1, "tier": "extra_eu"},
    {"id": 219, "name": "Austrian 2. Liga",       "country": "Austria",     "div": 2, "tier": "extra_eu"},

    # ── Americas ──────────────────────────────────────────────────────────────
    {"id": 253, "name": "MLS",                    "country": "USA",         "div": 1, "tier": "americas"},
    {"id": 262, "name": "Liga MX",                "country": "Mexico",      "div": 1, "tier": "americas"},
    {"id": 71,  "name": "Série A",                "country": "Brazil",      "div": 1, "tier": "americas"},
    {"id": 128, "name": "Primera División",       "country": "Argentina",   "div": 1, "tier": "americas"},
]

# Season conventions:
#   European leagues:  2025 = 2025/26 season (Aug 2025 – May 2026)  ← just ended
#   Americas leagues:  2025 = 2025 calendar year (Feb-Dec 2025)     ← ended Dec 2025
#                      2026 = 2026 calendar year (Mar-Dec 2026)      ← live now

SEASONS        = [2021, 2022, 2023, 2024]   # 4 complete training seasons
CURRENT_SEASON = 2025                        # 2025/26 European — used as model test season

# Americas 2026 season — calendar year leagues currently in-season
CURRENT_SEASON_AMERICAS = 2026              # MLS, Brazilian Série A, Liga MX Apertura

# Quick lookups
LEAGUE_BY_ID   = {l["id"]: l for l in LEAGUES}
LEAGUE_IDS     = [l["id"] for l in LEAGUES]
BIG5_IDS       = [l["id"] for l in LEAGUES if l["tier"] == "big5"]
EXTRA_EU_IDS   = [l["id"] for l in LEAGUES if l["tier"] == "extra_eu"]
AMERICAS_IDS   = [l["id"] for l in LEAGUES if l["tier"] == "americas"]
DIVISION1_IDS  = [l["id"] for l in LEAGUES if l["div"] == 1]
DIVISION2_IDS  = [l["id"] for l in LEAGUES if l["div"] == 2]

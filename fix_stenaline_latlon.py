"""
Convert Stenaline LAT / LONG from radians to degrees and
plot GPS track before and after the conversion.

Data source: 9235517_YYYYMM.csv files (one per month, organised in year folders)
Raw columns: LAT (radians), LONG (radians)

Usage:
    python fix_stenaline_latlon.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from pathlib import Path

# ── Path to the Stenaline data directory ─────────────────────────────────────
DATA_DIR = Path(
    "/Users/iroshanij/Library/CloudStorage/"
    "OneDrive-SharedLibraries-UiTOffice365/"
    "O365-TwinShip Stena Data and Information - CSV Data/"
    "Stenaline/9235517_sensor by sensor"
)

RAW_LAT_COL = "LAT"
RAW_LON_COL = "LONG"

OUTPUT_PLOT    = Path("results/01_eda/stenaline/plots/latlon_rad_vs_deg.png")
OUTPUT_MAP     = Path("results/01_eda/stenaline/plots/vessel_track_map.html")

# ── Discover and load all monthly CSVs ───────────────────────────────────────
csv_files = sorted(DATA_DIR.rglob("9235517_*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No 9235517_*.csv files found under: {DATA_DIR}")

print(f"Found {len(csv_files)} CSV file(s) — loading …")

chunks = []
for fp in csv_files:
    try:
        df_part = pd.read_csv(fp, usecols=[RAW_LAT_COL, RAW_LON_COL])
        chunks.append(df_part)
    except Exception as e:
        print(f"  Skipped {fp.name}: {e}")

df = pd.concat(chunks, ignore_index=True)
print(f"Total rows loaded: {len(df):,}")

# ── Print raw stats ───────────────────────────────────────────────────────────
lat_raw = df[RAW_LAT_COL].dropna()
lon_raw = df[RAW_LON_COL].dropna()

print(f"\nRaw LAT  — min: {lat_raw.min():.6f}  max: {lat_raw.max():.6f}  "
      f"median: {lat_raw.median():.6f}")
print(f"Raw LONG — min: {lon_raw.min():.6f}  max: {lon_raw.max():.6f}  "
      f"median: {lon_raw.median():.6f}")

# ── Convert radians → degrees ─────────────────────────────────────────────────
# Median LAT ~0.977 rad → ~56°N confirms radians storage.
# Safe threshold: if median latitude < 2.0 it must be radians.
lat_median = float(lat_raw.median())
is_radians = abs(lat_median) < 2.0

if is_radians:
    print("\nUnit detected: RADIANS → converting to degrees.")
    df["GPS_LAT"] = np.degrees(df[RAW_LAT_COL])
    df["GPS_LON"] = np.degrees(df[RAW_LON_COL])
else:
    print("\nUnit detected: already in DEGREES — copying as-is for comparison.")
    df["GPS_LAT"] = df[RAW_LAT_COL]
    df["GPS_LON"] = df[RAW_LON_COL]

print(f"\nConverted GPS_LAT — min: {df['GPS_LAT'].dropna().min():.4f}°  "
      f"max: {df['GPS_LAT'].dropna().max():.4f}°")
print(f"Converted GPS_LON — min: {df['GPS_LON'].dropna().min():.4f}°  "
      f"max: {df['GPS_LON'].dropna().max():.4f}°")

# ── Plot ──────────────────────────────────────────────────────────────────────
valid_before = df[[RAW_LAT_COL, RAW_LON_COL]].dropna()
valid_after  = df[["GPS_LAT", "GPS_LON"]].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Before ---
axes[0].scatter(
    valid_before[RAW_LON_COL],
    valid_before[RAW_LAT_COL],
    s=0.5,
    alpha=0.3,
    color="steelblue",
    rasterized=True,
)
axes[0].set_title(
    f"Before conversion\n(raw values — {'radians' if is_radians else 'degrees'})",
    fontsize=12,
)
axes[0].set_xlabel(f"{RAW_LON_COL} (raw)", fontsize=10)
axes[0].set_ylabel(f"{RAW_LAT_COL} (raw)", fontsize=10)
axes[0].grid(True, linewidth=0.4)

# --- After ---
axes[1].scatter(
    valid_after["GPS_LON"],
    valid_after["GPS_LAT"],
    s=0.5,
    alpha=0.3,
    color="darkorange",
    rasterized=True,
)
axes[1].set_title("After conversion\n(degrees)", fontsize=12)
axes[1].set_xlabel("Longitude (°E)", fontsize=10)
axes[1].set_ylabel("Latitude (°N)", fontsize=10)
axes[1].grid(True, linewidth=0.4)

fig.suptitle(
    "Stenaline (MMSI 9235517): GPS Track (Lat vs Lon)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()

OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {OUTPUT_PLOT}")
plt.show()

# ── World map (folium) ────────────────────────────────────────────────────────
print("\nBuilding interactive world map …")
coords = valid_after[["GPS_LAT", "GPS_LON"]].dropna()

# Down-sample to ≤5 000 points for a responsive map
step = max(1, len(coords) // 5000)
coords_sampled = coords.iloc[::step].reset_index(drop=True)

center_lat = float(coords_sampled["GPS_LAT"].median())
center_lon = float(coords_sampled["GPS_LON"].median())

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="OpenStreetMap",
)

# Route as a polyline
route_coords = list(zip(coords_sampled["GPS_LAT"], coords_sampled["GPS_LON"]))
folium.PolyLine(
    route_coords,
    color="#e07b00",
    weight=2,
    opacity=0.8,
    tooltip="Stenaline (MMSI 9235517) route",
).add_to(m)

# Start / end markers
folium.Marker(
    location=route_coords[0],
    popup="Start",
    icon=folium.Icon(color="green", icon="play"),
).add_to(m)
folium.Marker(
    location=route_coords[-1],
    popup="End",
    icon=folium.Icon(color="red", icon="stop"),
).add_to(m)

folium.map.LayerControl().add_to(m)

OUTPUT_MAP.parent.mkdir(parents=True, exist_ok=True)
m.save(str(OUTPUT_MAP))
print(f"Map saved  → {OUTPUT_MAP}")

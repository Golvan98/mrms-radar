# backend/app/main.py
import os, io, re, gzip, time, asyncio, requests
import numpy as np
import xarray as xr  # cfgrib provides the engine
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from matplotlib import pyplot as plt

# ----------------------- CONFIG -----------------------
MRMS_URL = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"
PATTERN = re.compile(r"MRMS_ReflectivityAtLowestAltitude_00\.50_\d{8}-\d{6}\.grib2\.gz$")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REFRESH_SECONDS = 180  # 3 minutes
CONUS_BOUNDS = [[24.5, -125.0], [49.5, -66.5]]  # used for Leaflet ImageOverlay

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend origin in prod
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----------------------- UTILITIES -----------------------
def find_latest_filename() -> str:
    """Scrape MRMS directory and return latest RALA filename."""
    r = requests.get(MRMS_URL, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")


    
    names = []
    for a in soup.find_all("a"):
        text = (a.get_text(strip=True) or "")
        href = a.get("href", "")
        if PATTERN.search(text):
            names.append(text)
        elif PATTERN.search(href):
            names.append(href)


    if not names:
        raise RuntimeError("No RALA files found in MRMS directory")
    names.sort()  # lexicographic works because timestamp format is sortable
    return names[-1]


def download_gz(name: str) -> str:
    url = MRMS_URL + name
    gz_path = os.path.join(DATA_DIR, name)
    if not os.path.exists(gz_path):
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(1024 * 64):
                    f.write(chunk)
    # decompress .gz -> .grib2
    grib_path = gz_path[:-3]  # remove .gz
    if not os.path.exists(grib_path):
        with gzip.open(gz_path, "rb") as gzr, open(grib_path, "wb") as out:
            out.write(gzr.read())
    return grib_path


def _first_data_var(ds: xr.Dataset) -> xr.DataArray:
    # MRMS/GRIB2 field names vary; pick first data var robustly.
    for k, v in ds.data_vars.items():
        return v
    raise RuntimeError("No data variables in GRIB dataset")


def grib_to_png(grib_path: str, out_png: str) -> None:
    """
    Load GRIB2 with xarray+cfgrib, colorize reflectivity to PNG.
    Transparent for NaN or very low values.
    """
    # Open with cfgrib
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    da = _first_data_var(ds).astype("float32").load()
    arr = da.values  # 2D numpy array, units dBZ typically

    # Mask invalid/very low values
    mask = ~np.isfinite(arr) | (arr < -5)
    vmin, vmax = 0.0, 75.0
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)

    # Use a nice perceptual colormap
    cmap = plt.get_cmap("turbo")
    rgba = (cmap(norm) * 255).astype("uint8")
    rgba[mask, 3] = 0  # transparent where invalid

    # Save RGBA PNG
    from PIL import Image
    Image.fromarray(rgba, mode="RGBA").save(out_png, optimize=True)


def write_meta(ts_str: str) -> None:
    meta_path = os.path.join(STATIC_DIR, "latest.json")
    import json
    json.dump(
        {"timestamp": ts_str, "bounds": CONUS_BOUNDS},
        open(meta_path, "w"),
    )


def refresh_once() -> str:
    """
    Fetch the newest MRMS RALA, convert to static/latest.png + latest.json.
    Returns timestamp string.
    """
    name = find_latest_filename()  # e.g. MRMS_RALA_00.50_20251006-053530.grib2.gz
    ts = name.split("_")[3].split(".")[0]  # 20251006-053530
    grib = download_gz(name)
    out_png = os.path.join(STATIC_DIR, "latest.png")
    grib_to_png(grib, out_png)
    write_meta(ts)
    return ts


async def refresher_loop():
    """Background task to keep latest.png fresh."""
    while True:
        try:
            ts = refresh_once()
            print(f"[MRMS] refreshed {ts}")
        except Exception as e:
            print(f"[MRMS] refresh failed: {e}")
        await asyncio.sleep(REFRESH_SECONDS)


# ----------------------- ROUTES -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("startup")
async def _startup():
    # Kick off background refresher
    asyncio.create_task(refresher_loop())

@app.get("/api/latest-meta")
def latest_meta():
    """Return timestamp + bounds; frontend uses it to refresh overlay."""
    meta_path = os.path.join(STATIC_DIR, "latest.json")
    if not os.path.exists(meta_path):
        # force one refresh synchronously on cold start
        ts = refresh_once()
        return {"timestamp": ts, "bounds": CONUS_BOUNDS}
    import json
    return json.load(open(meta_path))

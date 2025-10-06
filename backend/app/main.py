# backend/app/main.py
import os, io, re, gzip, time, asyncio, requests
import numpy as np
import xarray as xr  # cfgrib provides the engine
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import gc

# ----------------------- CONFIG -----------------------
MRMS_URL = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"
PATTERN = re.compile(
    r"MRMS_ReflectivityAtLowestAltitude_(?P<res>01\.00|00\.50)_(?P<ts>\d{8}-\d{6})\.grib2\.gz$"
)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REFRESH_SECONDS = 180  # 3 minutes
GRID_DECIMATE = 3   # 1 = full res, 2 = half res each axis (~1/4 pixels)
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
    r = requests.get(MRMS_URL, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    candidates = []
    for a in soup.find_all("a"):
        text = (a.get_text(strip=True) or "")
        href = a.get("href", "") or ""
        for s in (text, href):
            m = PATTERN.search(s)
            if m:
                ts = m.group("ts")      # e.g. 20251006-075838
                res = m.group("res")    # "01.00" or "00.50"
                # Keep the real filename we matched on
                # (prefer href if present; text is fine too)
                name = s
                candidates.append((ts, res, name))
                break

    if not candidates:
        raise RuntimeError("No RALA files found in MRMS directory")

    # Sort by timestamp, then prefer 01.00 over 00.50 when timestamps tie
    candidates.sort(key=lambda x: (x[0], 0 if x[1] == "01.00" else 1))
    return candidates[-1][2]


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
            shutil.copyfileobj(gzr, out, length=1024 * 64)
        try:
            os.remove(gz_path)
        except OSError:
            pass
    return grib_path


def _first_data_var(ds: xr.Dataset) -> xr.DataArray:
    # MRMS/GRIB2 field names vary; pick first data var robustly.
    for k, v in ds.data_vars.items():
        return v
    raise RuntimeError("No data variables in GRIB dataset")


def _make_palette() -> list[int]:
    # Simple blue->cyan->green->yellow->orange->red gradient (256 colors).
    # Index 0 will be transparent, so its color doesn't matter.
    stops = [
        (0.00, (  0,   0,   0)),
        (0.15, ( 30,  30, 140)),
        (0.30, (  0, 120, 255)),
        (0.50, (  0, 210, 120)),
        (0.70, (255, 230,   0)),
        (0.85, (255, 100,   0)),
        (1.00, (180,   0,  80)),
    ]
    pal: list[int] = []
    for i in range(256):
        t = i / 255.0
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t <= t1:
                f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                r = int(c0[0] * (1 - f) + c1[0] * f)
                g = int(c0[1] * (1 - f) + c1[1] * f)
                b = int(c0[2] * (1 - f) + c1[2] * f)
                pal.extend([r, g, b])
                break
    return pal

def grib_to_png(grib_path: str, out_png: str) -> None:
    """Load GRIB2 with xarray+cfgrib and write a paletted PNG (index 0 transparent)."""
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        da = _first_data_var(ds)
        arr = da.values.astype("float32", copy=False)  # ensure 32-bit

        # Optional in-memory decimation to reduce pixels before colorizing
        if GRID_DECIMATE > 1:
            arr = arr[::GRID_DECIMATE, ::GRID_DECIMATE]

        # Mask invalid / very low
        mask = ~np.isfinite(arr) | (arr < -5.0)

        # Normalize 0..75 dBZ -> 0..1
        vmin, vmax = 0.0, 75.0
        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

        # Map to palette indices 1..255 (0 reserved for transparent)
        idx = (norm * 254 + 1).astype("uint8", copy=False)
        idx[mask] = 0  # transparent

        from PIL import Image
        im = Image.fromarray(idx, mode="P")
        im.putpalette(_make_palette())
        # Per-index alpha: index 0 transparent, others opaque
        im.info["transparency"] = bytes([0] + [255] * 255)
        im.save(out_png, optimize=True)
    finally:
        try:
            ds.close()
        except Exception:
            pass
        if 'da' in locals():
            del da
        if 'arr' in locals():
            del arr
        gc.collect()


def write_meta(ts_str: str) -> None:
    meta_path = os.path.join(STATIC_DIR, "latest.json")
    import json
    with open(meta_path, "w") as f:
        json.dump({"timestamp": ts_str, "bounds": CONUS_BOUNDS}, f)


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

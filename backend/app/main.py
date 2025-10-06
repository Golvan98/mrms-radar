# backend/app/main.py
import os, io, re, gzip, time, asyncio, requests
import numpy as np
import xarray as xr  # cfgrib provides the engine
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import gc
import shutil
# ----------------------- CONFIG -----------------------
MRMS_URL = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"
PATTERN = re.compile(
    r"MRMS_ReflectivityAtLowestAltitude_(?P<res>01\.00)_(?P<ts>\d{8}-\d{6})\.grib2\.gz$"
)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REFRESH_SECONDS = 180  # 3 minutes
GRID_DECIMATE = 4   # 1 = full res, 2 = half res each axis (~1/4 pixels)
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
    r = requests.get(MRMS_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    files_100 = []  # 01.00 km
    files_050 = []  # 00.50 km
    for a in soup.find_all("a"):
        for s in ((a.get_text(strip=True) or ""), (a.get("href") or "")):
            m = PATTERN.search(s)
            if not m:
                continue
            ts = m.group("ts")  # e.g. 20251006-095638
            res = m.group("res")
            if res == "01.00":
                files_100.append((ts, s))
            else:
                files_050.append((ts, s))
            break

    if not files_100 and not files_050:
        raise RuntimeError("No RALA files found in MRMS directory")

    if files_100:
        files_100.sort(key=lambda x: x[0])
        return files_100[-1][1]

    files_050.sort(key=lambda x: x[0])
    return files_050[-1][1]


def _looks_like_grib2(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        return head == b"GRIB"
    except Exception:
        return False

def download_gz(name: str) -> str:
    url = MRMS_URL + name
    gz_path = os.path.join(DATA_DIR, name)
    grib_path = gz_path[:-3]  # .grib2

    def _fetch():
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    if chunk:
                        f.write(chunk)
        # gunzip stream -> .grib2
        with gzip.open(gz_path, "rb") as gzr, open(grib_path, "wb") as out:
            shutil.copyfileobj(gzr, out, length=64 * 1024)
        try:
            os.remove(gz_path)
        except OSError:
            pass

    if not os.path.exists(grib_path):
        _fetch()

    # file sanity check; if bad, retry once
    if (not os.path.exists(grib_path)) or os.path.getsize(grib_path) < 1024 or not _looks_like_grib2(grib_path):
        try:
            if os.path.exists(gz_path):
                os.remove(gz_path)
            if os.path.exists(grib_path):
                os.remove(grib_path)
        except OSError:
            pass
        _fetch()

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
    name = find_latest_filename()
    ts = name.split("_")[3].split(".")[0]
    grib = download_gz(name)
    out_png = os.path.join(STATIC_DIR, "latest.png")
    try:
        grib_to_png(grib, out_png)
    except Exception as e:
        # if we picked a 00.50, try latest 01.00 as a fallback
        if "_00.50_" in name:
            try:
                # force 01.00
                old_pattern = r"(?P<res>01\.00|00\.50)"
                # quick re-scan but prefer 01.00 only
                rname = find_latest_filename()  # will pick 01.00 if present due to new logic
                if "_01.00_" in rname:
                    grib = download_gz(rname)
                    grib_to_png(grib, out_png)
                    ts = rname.split("_")[3].split(".")[0]
                else:
                    raise e
            except Exception:
                raise e
        else:
            raise e
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

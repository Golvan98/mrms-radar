# backend/app/main.py
import os, re, gzip, asyncio, requests, gc, shutil, json
import numpy as np
import xarray as xr  # cfgrib provides the engine
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ----------------------- CONFIG -----------------------
UA = "Mozilla/5.0 (mrms-radar/1.0; +render)"
MRMS_HTTP = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"
MRMS_S3 = "https://noaa-mrms-pds.s3.amazonaws.com"
# Fixed S3 prefix - note the different format for S3 vs HTTP
S3_PREFIX = os.getenv("MRMS_S3_PREFIX", "CONUS/ReflectivityAtLowestAltitude/")
PATTERN = re.compile(
    r"MRMS_ReflectivityAtLowestAltitude_(?P<res>01\.00)_(?P<ts>\d{8}-\d{6})\.grib2\.gz$"
)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "180"))
GRID_DECIMATE = int(os.getenv("GRID_DECIMATE", "4"))
CONUS_BOUNDS = [[24.5, -125.0], [49.5, -66.5]]

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
    """Return absolute URL for newest 01.00 RALA .grib2.gz."""
    # Try NOAA HTML directory first
    try:
        r = requests.get(MRMS_HTTP, timeout=20, headers={"User-Agent": UA})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        names = []
        for a in soup.find_all("a"):
            for s in ((a.get_text(strip=True) or ""), (a.get("href") or "")):
                if PATTERN.search(s):
                    names.append(s)
                    break
        if names:
            names.sort()
            return MRMS_HTTP + names[-1]
    except Exception as e:
        print(f"[MRMS] HTTP fetch failed: {e}, trying S3...")

    # Fallback: S3 list-type=2 XML
    params = {
        "list-type": "2",
        "prefix": S3_PREFIX,
        "max-keys": "1000",
    }
    print(f"[MRMS] Searching S3 with prefix: {S3_PREFIX}")
    r = requests.get(f"{MRMS_S3}/", params=params, timeout=20, headers={"User-Agent": UA})
    r.raise_for_status()
    from xml.etree import ElementTree as ET
    root = ET.fromstring(r.text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    def filter_candidates(root_el):
        keys = [el.text for el in root_el.findall(".//s3:Key", ns)]
        matched = [k for k in keys if PATTERN.search(os.path.basename(k))]
        print(f"[MRMS] Found {len(keys)} total keys, {len(matched)} matching pattern")
        return matched

    candidates = filter_candidates(root)
    
    # If nothing matched, try to advance a couple pages
    if not candidates:
        next_token_el = root.find(".//s3:NextContinuationToken", ns)
        tries = 0
        while next_token_el is not None and tries < 2 and not candidates:
            params2 = dict(params)
            params2["continuation-token"] = next_token_el.text
            r2 = requests.get(f"{MRMS_S3}/", params=params2, timeout=20, headers={"User-Agent": UA})
            r2.raise_for_status()
            root2 = ET.fromstring(r2.text)
            candidates = filter_candidates(root2)
            next_token_el = root2.find(".//s3:NextContinuationToken", ns)
            tries += 1

    if not candidates:
        raise RuntimeError(f"No RALA files found in MRMS directory (prefix={S3_PREFIX})")
    
    candidates.sort()
    latest = f"{MRMS_S3}/{candidates[-1]}"
    print(f"[MRMS] Latest file: {latest}")
    return latest


def _looks_like_grib2(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GRIB"
    except Exception:
        return False


def download_gz(url_or_name: str) -> str:
    if url_or_name.startswith("http"):
        url = url_or_name
        name = os.path.basename(url_or_name)
    else:
        url = MRMS_HTTP + url_or_name
        name = url_or_name

    gz_path = os.path.join(DATA_DIR, name)
    grib_path = gz_path[:-3]  # .grib2

    def _fetch():
        print(f"[MRMS] Downloading {url}...")
        with requests.get(url, stream=True, timeout=120, headers={"User-Agent": UA}) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"[MRMS] Downloaded {len(open(gz_path, 'rb').read())} bytes")
        # gunzip to .grib2
        print(f"[MRMS] Decompressing...")
        import shutil as _sh
        with gzip.open(gz_path, "rb") as gzr, open(grib_path, "wb") as out:
            _sh.copyfileobj(gzr, out, length=64 * 1024)
        try:
            os.remove(gz_path)
        except OSError:
            pass
        print(f"[MRMS] Decompressed to {grib_path}")

    if not os.path.exists(grib_path):
        _fetch()
    # sanity check; retry once if corrupt/short
    if (not os.path.exists(grib_path)) or os.path.getsize(grib_path) < 1024 or not _looks_like_grib2(grib_path):
        print(f"[MRMS] GRIB file appears corrupt, re-downloading...")
        for p in (gz_path, grib_path):
            try:
                os.remove(p)
            except OSError:
                pass
        _fetch()

    return grib_path


def _first_data_var(ds: xr.Dataset):
    for _, v in ds.data_vars.items():
        return v
    raise RuntimeError("No data variables in GRIB dataset")


def _make_palette() -> list[int]:
    # Simple blue->cyan->green->yellow->orange->red gradient (256 colors).
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
    print(f"[MRMS] Processing GRIB to PNG...")
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        da = _first_data_var(ds)
        arr = da.values.astype("float32", copy=False)
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
        im.info["transparency"] = bytes([0] + [255] * 255)
        im.save(out_png, optimize=True)
        print(f"[MRMS] PNG saved to {out_png}")
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
    with open(meta_path, "w") as f:
        json.dump({"timestamp": ts_str, "bounds": CONUS_BOUNDS}, f)


def refresh_once() -> str:
    name = find_latest_filename()
    ts = os.path.basename(name).split("_")[3].split(".")[0]
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
    # Don't block startup - do initial refresh in background
    asyncio.create_task(_initial_refresh())


async def _initial_refresh():
    """Non-blocking initial refresh that won't crash the server on startup."""
    await asyncio.sleep(2)  # Let server start first
    try:
        print("[MRMS] Starting initial refresh...")
        refresh_once()
        print("[MRMS] Initial refresh complete")
    except Exception as e:
        print(f"[MRMS] Initial refresh failed (will retry in background): {e}")
    # Start the continuous refresher loop
    asyncio.create_task(refresher_loop())


@app.get("/api/latest-meta")
def latest_meta():
    """Return timestamp + bounds; frontend uses it to refresh overlay."""
    meta_path = os.path.join(STATIC_DIR, "latest.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    # cold start: return basic info
    return {"error": "Data not yet available", "bounds": CONUS_BOUNDS}


# Optional: quick S3 debug to verify prefix/connectivity on Render
@app.get("/api/debug/s3-head")
def debug_s3_head():
    params = {"list-type": "2", "prefix": S3_PREFIX, "max-keys": "200"}
    r = requests.get(f"{MRMS_S3}/", params=params, timeout=20, headers={"User-Agent": UA})
    r.raise_for_status()
    from xml.etree import ElementTree as ET
    root = ET.fromstring(r.text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [el.text for el in root.findall(".//s3:Key", ns)]
    tail = keys[-10:] if len(keys) >= 10 else keys
    return {"prefix": S3_PREFIX, "count": len(keys), "tail": tail}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
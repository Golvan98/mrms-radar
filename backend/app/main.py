# backend/app/main.py
import os, re, gzip, asyncio, requests, gc, shutil, json, time
from typing import Optional

import numpy as np
import xarray as xr
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ----------------------- CONFIG -----------------------
UA = "Mozilla/5.0 (mrms-radar/1.0; +render)"
MRMS_HTTP = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"
MRMS_S3   = "https://noaa-mrms-pds.s3.amazonaws.com"

# CRITICAL: Use 00.25 ONLY - smallest grid to avoid OOM on 512MB
S3_PREFIXES = [
    "CONUS/ReflectivityAtLowestAltitude_00.25/",
]

PATTERN = re.compile(
    r"MRMS_ReflectivityAtLowestAltitude_(?P<res>00\.25|00\.50|01\.00)_(?P<ts>\d{8}-\d{6})\.grib2\.gz$"
)

BASE_DIR    = os.path.dirname(__file__)
STATIC_DIR  = os.path.join(BASE_DIR, "static")
DATA_DIR    = os.path.join(BASE_DIR, "data")
REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "300"))
GRID_DECIMATE   = max(1, int(os.getenv("GRID_DECIMATE", "8")))  # Increased from 4 to 8
CONUS_BOUNDS    = [[24.5, -125.0], [49.5, -66.5]]

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_refresh_lock = asyncio.Lock()


# ----------------------- UTILITIES -----------------------
def _s3_list_latest(prefix: str) -> Optional[str]:
    """Return absolute URL for newest .grib2.gz under a given S3 prefix, else None."""
    params = {"list-type": "2", "prefix": prefix, "max-keys": "500"}
    print(f"[MRMS] S3 list prefix: {prefix}")
    r = requests.get(f"{MRMS_S3}/", params=params, timeout=20, headers={"User-Agent": UA})
    r.raise_for_status()

    from xml.etree import ElementTree as ET
    root = ET.fromstring(r.text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    keys = [el.text for el in root.findall(".//s3:Key", ns)]
    matches = [k for k in keys if PATTERN.search(os.path.basename(k))]
    print(f"[MRMS] keys={len(keys)} matches={len(matches)}")

    if not matches:
        return None
    matches.sort()
    latest = f"{MRMS_S3}/{matches[-1]}"
    print(f"[MRMS] latest from {prefix}: {latest}")
    return latest


def find_latest_filename() -> str:
    """Newest RALA (.grib2.gz), HTML first; else S3."""
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
        print(f"[MRMS] HTTP dir scrape failed: {e}")

    for pfx in S3_PREFIXES:
        url = _s3_list_latest(pfx)
        if url:
            return url
    raise RuntimeError(f"No RALA files found (tried: {', '.join(S3_PREFIXES)})")


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
    grib_path = gz_path[:-3]

    def _fetch():
        print(f"[MRMS] Download {url}")
        with requests.get(url, stream=True, timeout=120, headers={"User-Agent": UA}) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"[MRMS] Decompress {gz_path}")
        with gzip.open(gz_path, "rb") as gzr, open(grib_path, "wb") as out:
            shutil.copyfileobj(gzr, out, length=64 * 1024)
        try:
            os.remove(gz_path)
        except OSError:
            pass

    if not os.path.exists(grib_path):
        _fetch()

    if (not os.path.exists(grib_path)) or os.path.getsize(grib_path) < 1024 or not _looks_like_grib2(grib_path):
        print("[MRMS] GRIB sanity failed; re-fetch")
        for p in (gz_path, grib_path):
            try: os.remove(p)
            except OSError: pass
        _fetch()

    return grib_path


def _make_palette() -> list[int]:
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
    """ULTRA MEMORY EFFICIENT: Process GRIB in smallest chunks possible."""
    print(f"[MRMS] Process GRIB -> PNG (low memory mode)")
    
    # Force garbage collection before opening
    gc.collect()
    
    # Open with minimal memory footprint
    ds = xr.open_dataset(
        grib_path, 
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "errors": "ignore"  # Ignore ECCODES warnings
        }
    )
    
    try:
        # Get first data variable
        for _, v in ds.data_vars.items():
            da = v
            break
        
        # Get shape BEFORE loading data
        shape = da.shape
        print(f"[MRMS] Original shape: {shape}")
        
        # Calculate aggressive decimation
        s = GRID_DECIMATE
        h, w = shape[0] // s, shape[1] // s
        print(f"[MRMS] Decimated shape: ({h}, {w})")
        
        # Load ONLY the decimated data - never load full array
        arr = da.values[::s, ::s].astype("float32", copy=False)
        
        # Close dataset immediately
        ds.close()
        del da, ds
        gc.collect()
        
        # Further downsample if still too large (> 2M elements = ~8MB)
        while arr.size > 2_000_000:
            print(f"[MRMS] Still too large ({arr.size} elements), downsampling 2x")
            arr = arr[::2, ::2]
            gc.collect()
        
        print(f"[MRMS] Final array size: {arr.shape} ({arr.nbytes / 1024 / 1024:.1f} MB)")
        
        # Process to 8-bit palette
        mask = ~np.isfinite(arr) | (arr < -5.0)
        arr = np.clip((arr - 0.0) / 75.0, 0.0, 1.0)
        idx = (arr * 254 + 1).astype("uint8", copy=False)
        idx[mask] = 0
        
        del arr, mask
        gc.collect()
        
        # Save PNG
        from PIL import Image
        im = Image.fromarray(idx, mode="P")
        im.putpalette(_make_palette())
        im.info["transparency"] = bytes([0] + [255] * 255)
        im.save(out_png, optimize=True)
        print(f"[MRMS] Saved {out_png}")
        
        del idx, im
        gc.collect()
        
    except Exception as e:
        print(f"[MRMS] Error processing GRIB: {e}")
        try:
            ds.close()
        except:
            pass
        raise


def _meta_path() -> str:
    return os.path.join(STATIC_DIR, "latest.json")


def write_meta(ts_str: str) -> None:
    with open(_meta_path(), "w") as f:
        json.dump({"timestamp": ts_str, "bounds": CONUS_BOUNDS}, f)


async def refresh_once_async() -> str:
    async with _refresh_lock:
        # Clean up old files first
        for f in os.listdir(DATA_DIR):
            if f.endswith('.grib2'):
                try:
                    os.remove(os.path.join(DATA_DIR, f))
                except:
                    pass
        
        name = find_latest_filename()
        ts = os.path.basename(name).split("_")[3].split(".")[0]
        grib = download_gz(name)
        out_png = os.path.join(STATIC_DIR, "latest.png")
        grib_to_png(grib, out_png)
        write_meta(ts)
        
        # Clean up GRIB file after processing
        try:
            os.remove(grib)
        except:
            pass
        
        gc.collect()
        return ts


async def refresher_loop():
    while True:
        try:
            ts = await refresh_once_async()
            print(f"[MRMS] refreshed {ts}")
        except Exception as e:
            print(f"[MRMS] refresh failed: {e}")
        await asyncio.sleep(REFRESH_SECONDS)


# ----------------------- ROUTES -----------------------
@app.get("/")
def root():
    return {"ok": True, "service": "mrms-radar", "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("startup")
async def _startup():
    print("[MRMS] Starting background refresher...")
    asyncio.create_task(refresher_loop())

@app.get("/api/latest-meta")
async def latest_meta():
    meta = _meta_path()
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f)
    try:
        ts = await refresh_once_async()
        return {"timestamp": ts, "bounds": CONUS_BOUNDS}
    except Exception as e:
        return {"error": f"refresh failed: {e}", "bounds": CONUS_BOUNDS}

@app.get("/api/force-refresh")
async def force_refresh():
    try:
        ts = await refresh_once_async()
        return {"status": "ok", "timestamp": ts}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/debug/s3-head")
def debug_s3_head():
    out = []
    for pfx in S3_PREFIXES:
        try:
            params = {"list-type": "2", "prefix": pfx, "max-keys": "200"}
            r = requests.get(f"{MRMS_S3}/", params=params, timeout=20, headers={"User-Agent": UA})
            r.raise_for_status()
            from xml.etree import ElementTree as ET
            root = ET.fromstring(r.text)
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            keys = [el.text for el in root.findall(".//s3:Key", ns)]
            out.append({"prefix": pfx, "count": len(keys), "tail": keys[-10:]})
        except Exception as e:
            out.append({"prefix": pfx, "error": str(e)})
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
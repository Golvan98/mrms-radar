# backend/app/main.py
import os, re, gzip, asyncio, requests, gc, shutil, json, time
import numpy as np
import xarray as xr  # cfgrib provides the engine
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ----------------------- CONFIG -----------------------
UA = "Mozilla/5.0 (mrms-radar/1.0; +render)"
MRMS_HTTP = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"
MRMS_S3   = "https://noaa-mrms-pds.s3.amazonaws.com"

# Correct S3 prefixes (do NOT change to ...ReflectivityAtLowestAltitude/)
S3_PREFIXES = [
    "CONUS/ReflectivityAtLowestAltitude_01.00/",
    "CONUS/ReflectivityAtLowestAltitude_00.50/",
]

# Accept either 01.00 or 00.50; 01.00 preferred
PATTERN = re.compile(
    r"MRMS_ReflectivityAtLowestAltitude_(?P<res>01\.00|00\.50)_(?P<ts>\d{8}-\d{6})\.grib2\.gz$"
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "180"))
GRID_DECIMATE   = int(os.getenv("GRID_DECIMATE", "4"))
CONUS_BOUNDS    = [[24.5, -125.0], [49.5, -66.5]]

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----------------------- UTILITIES -----------------------
def _s3_list_latest(prefix: str) -> str | None:
    """Return absolute URL for newest .grib2.gz under a given S3 prefix, else None."""
    params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    print(f"[MRMS] S3 list prefix: {prefix}")
    r = requests.get(f"{MRMS_S3}/", params=params, timeout=20, headers={"User-Agent": UA})
    r.raise_for_status()

    from xml.etree import ElementTree as ET
    root = ET.fromstring(r.text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    def filter_candidates(root_el):
        keys = [el.text for el in root_el.findall(".//s3:Key", ns)]
        matches = [k for k in keys if PATTERN.search(os.path.basename(k))]
        print(f"[MRMS] keys={len(keys)} matches={len(matches)}")
        return matches

    candidates = filter_candidates(root)
    # Try a couple continuation pages if first page is empty/mid-list
    next_token_el = root.find(".//s3:NextContinuationToken", ns)
    tries = 0
    while (not candidates) and (next_token_el is not None) and tries < 2:
        params2 = dict(params)
        params2["continuation-token"] = next_token_el.text
        r2 = requests.get(f"{MRMS_S3}/", params=params2, timeout=20, headers={"User-Agent": UA})
        r2.raise_for_status()
        root2 = ET.fromstring(r2.text)
        candidates = filter_candidates(root2)
        next_token_el = root2.find(".//s3:NextContinuationToken", ns)
        tries += 1

    if not candidates:
        return None
    candidates.sort()
    latest = f"{MRMS_S3}/{candidates[-1]}"
    print(f"[MRMS] latest from {prefix}: {latest}")
    return latest


def find_latest_filename() -> str:
    """Newest RALA (.grib2.gz), HTML first; if that fails, S3 01.00 then 00.50."""
    # 1) HTML directory scrape
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

    # 2) S3 listing: prefer 01.00, then 00.50
    for pfx in S3_PREFIXES:
        url = _s3_list_latest(pfx)
        if url:
            return url
    raise RuntimeError(f"No RALA files found in MRMS S3 (tried {', '.join(S3_PREFIXES)})")


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
        print(f"[MRMS] Download {url}")
        with requests.get(url, stream=True, timeout=120, headers={"User-Agent": UA}) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    if chunk:
                        f.write(chunk)
        # gunzip to .grib2
        print(f"[MRMS] Decompress {gz_path} -> {grib_path}")
        with gzip.open(gz_path, "rb") as gzr, open(grib_path, "wb") as out:
            shutil.copyfileobj(gzr, out, length=64 * 1024)
        try:
            os.remove(gz_path)
        except OSError:
            pass

    if not os.path.exists(grib_path):
        _fetch()

    # sanity check; retry once if corrupt/short
    if (not os.path.exists(grib_path)) or os.path.getsize(grib_path) < 1024 or not _looks_like_grib2(grib_path):
        print("[MRMS] GRIB sanity failed; re-fetch")
        for p in (gz_path, grib_path):
            try: os.remove(p)
            except OSError: pass
        _fetch()

    return grib_path


def _first_data_var(ds: xr.Dataset):
    for _, v in ds.data_vars.items():
        return v
    raise RuntimeError("No data variables in GRIB dataset")


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
    """Load GRIB2 with xarray+cfgrib and write a paletted PNG (index 0 transparent)."""
    print(f"[MRMS] Process GRIB -> PNG")
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        da = _first_data_var(ds)
        arr = da.values.astype("float32", copy=False)

        if GRID_DECIMATE > 1:
            arr = arr[::GRID_DECIMATE, ::GRID_DECIMATE]

        mask = ~np.isfinite(arr) | (arr < -5.0)
        vmin, vmax = 0.0, 75.0
        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        idx = (norm * 254 + 1).astype("uint8", copy=False)
        idx[mask] = 0

        from PIL import Image
        im = Image.fromarray(idx, mode="P")
        im.putpalette(_make_palette())
        im.info["transparency"] = bytes([0] + [255] * 255)
        im.save(out_png, optimize=True)
        print(f"[MRMS] Saved {out_png}")
    finally:
        try: ds.close()
        except Exception: pass
        if 'da' in locals(): del da
        if 'arr' in locals(): del arr
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
    while True:
        try:
            ts = refresh_once()
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
    # Do not block startup; kick background loop
    asyncio.create_task(refresher_loop())

@app.get("/api/latest-meta")
def latest_meta():
    meta_path = os.path.join(STATIC_DIR, "latest.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    # cold start: try once synchronously
    try:
        ts = refresh_once()
        return {"timestamp": ts, "bounds": CONUS_BOUNDS}
    except Exception as e:
        # graceful response; frontend can retry soon
        return {"error": f"refresh failed: {e}", "bounds": CONUS_BOUNDS}, 503

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

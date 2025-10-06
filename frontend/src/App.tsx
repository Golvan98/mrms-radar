import { useEffect, useState } from "react";
import { MapContainer, TileLayer, ImageOverlay } from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

// If not set, default to local FastAPI
const API_BASE = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

type Meta = { timestamp: string; bounds: LatLngBoundsExpression };

// ---- Fixed right-side legend (always visible) ----
function SideLegend() {
  // matches backend palette (low→high)
  const gradient = `linear-gradient(to top,
    rgb(30,30,140) 0%,
    rgb(0,120,255) 20%,
    rgb(0,210,120) 40%,
    rgb(255,230,0) 60%,
    rgb(255,100,0) 80%,
    rgb(180,0,80) 100%)`;

  const ticks = [0, 10, 20, 30, 40, 50, 60, 70, 75];
  const pct = (dbz: number) => `${(dbz / 75) * 100}%`; // 0 at bottom, 75 at top

  return (
    <div className="fixed right-4 top-1/2 -translate-y-1/2 z-[10000] pointer-events-none">
      <div className="bg-white/90 backdrop-blur rounded-xl shadow p-3 w-28">
        <div className="text-xs font-medium mb-2">Reflectivity (dBZ)</div>

        <div className="flex gap-2 items-stretch">
          {/* color bar */}
          <div className="relative h-64 w-4 rounded" style={{ background: gradient }}>
            {ticks.map((t) => (
              <div key={t} className="absolute left-0 w-4" style={{ bottom: pct(t) }}>
                <div className="h-[1px] w-3 bg-gray-700" />
              </div>
            ))}
          </div>

          {/* labels */}
          <div className="relative h-64 text-[10px] text-gray-700">
            {ticks.map((t) => (
              <div key={t} className="absolute -translate-y-1/2" style={{ bottom: pct(t) }}>
                {t}
              </div>
            ))}
          </div>
        </div>

        <div className="mt-1 text-[10px] text-gray-600">
          Transparent ≈ &lt; -5 dBZ / NoData
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/latest-meta`);
        const j: Meta = await r.json();
        setMeta(j);
      } catch (e) {
        console.error("meta fetch failed", e);
      }
    };
    load();
    const timer = setInterval(load, 120000);
    return () => clearInterval(timer);
  }, []);

  const bounds: LatLngBoundsExpression =
    meta?.bounds || [
      [24.5, -125.0],
      [49.5, -66.5],
    ];

  // cache-bust the PNG with timestamp so Leaflet re-loads it
  const radarUrl = meta
    ? `${API_BASE}/static/latest.png?t=${encodeURIComponent(meta.timestamp)}`
    : "";

  return (
    <div className="h-screen w-screen">
      <MapContainer
        center={[37.5, -96]}
        zoom={4}
        style={{ height: "100%", width: "100%", zIndex: 0 }} // keep map under overlays
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />

        {radarUrl && (
          <ImageOverlay
            url={radarUrl}
            bounds={bounds}
            opacity={0.7}
            crossOrigin="anonymous"
          />
        )}
      </MapContainer>

      {/* Status badge (top-left) */}
      <div className="fixed top-3 left-3 z-[10000] pointer-events-none bg-white/80 backdrop-blur px-3 py-1 rounded shadow text-xs">
        <div className="font-medium">MRMS RALA</div>
        <div>{meta ? `Updated: ${meta.timestamp}Z` : "Loading…"}</div>
      </div>

      {/* Always-on right-side legend */}
      <SideLegend />
    </div>
  );
}

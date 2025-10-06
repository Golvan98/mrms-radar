import { useEffect, useState } from "react";
import { MapContainer, TileLayer, ImageOverlay } from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

// If not set, default to local FastAPI
const API_BASE = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

type Meta = { timestamp: string; bounds: LatLngBoundsExpression };

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);

  useEffect(() => {
    let timer: any;

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
    // refresh metadata every 2 minutes
    timer = setInterval(load, 120000);
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
        style={{ height: "100%", width: "100%" }}
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

      <div className="absolute top-3 left-3 bg-white/80 backdrop-blur px-3 py-1 rounded shadow text-xs">
        <div className="font-medium">MRMS RALA</div>
        <div>{meta ? `Updated: ${meta.timestamp}Z` : "Loading…"}</div>
      </div>
    </div>
  );
}

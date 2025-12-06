"use client";

import { useEffect, useRef, useState } from "react";

type PersonAlert = {
  person_box: [number, number, number, number];
  helmet_detected: boolean;
  timestamp: string;
};

type AlertPayload = {
  timestamp: string;
  helmet_detected: boolean | null;
  alerts: PersonAlert[];
};

export default function Page() {
  const [latest, setLatest] = useState<AlertPayload | null>(null);
  const [history, setHistory] = useState<AlertPayload[]>([]);
  const [connected, setConnected] = useState(false);
  const mounted = useRef(true);

  const VIDEO_URL = "http://localhost:8000/video_feed";
  const ALERTS_URL = "http://localhost:8000/alerts";
  const HISTORY_URL = "http://localhost:8000/alerts/history";

  useEffect(() => {
    mounted.current = true;

    const pollLatest = async () => {
      try {
        const res = await fetch(ALERTS_URL, { cache: "no-store" });
        if (!res.ok) throw new Error("Failed to fetch latest alerts");
        const data: AlertPayload = await res.json();
        if (!mounted.current) return;
        setLatest(data);
        setConnected(true);
      } catch (err) {
        console.error("Latest alerts poll error:", err);
        setConnected(false);
      } finally {
        if (mounted.current) setTimeout(pollLatest, 1000);
      }
    };

    const pollHistory = async () => {
      try {
        const res = await fetch(HISTORY_URL, { cache: "no-store" });
        if (!res.ok) throw new Error("Failed to fetch alert history");
        const data: AlertPayload[] = await res.json();
        if (!mounted.current) return;
        setHistory(data.slice(-10).reverse()); // last 10 newest first
      } catch (err) {
        console.error("Alert history poll error:", err);
      } finally {
        if (mounted.current) setTimeout(pollHistory, 3000);
      }
    };

    pollLatest();
    pollHistory();

    return () => {
      mounted.current = false;
    };
  }, []);

  return (
    <main className="min-h-screen bg-slate-900 text-slate-100 flex flex-col items-center p-6">
      <div className="max-w-5xl w-full">
        <header className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-semibold">Helmet Detection Dashboard</h1>
          <div className="text-sm">
            Backend:{" "}
            <span
              className={`py-1 px-2 rounded ${
                connected ? "bg-green-600" : "bg-red-600"
              }`}
            >
              {connected ? "connected" : "offline"}
            </span>
          </div>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Video */}
          <div className="md:col-span-2 bg-black rounded shadow p-2">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg font-medium">Live Feed</h2>
              <div className="text-xs text-slate-400">MJPEG stream from backend</div>
            </div>
            <div className="border border-slate-700 rounded overflow-hidden">
              <img
                src={VIDEO_URL}
                alt="Live feed"
                width={960}
                height={540}
                style={{ width: "100%", height: "auto", display: "block" }}
              />
            </div>
          </div>

          {/* Stats / Alerts */}
          <div className="bg-slate-800 rounded shadow p-4">
            <h2 className="text-lg font-medium mb-3">Alerts</h2>

            
            <div className="mb-3">
              <div className="text-sm text-slate-400">Latest summary</div>
              <div className="mt-2 flex items-center justify-between">
                <div>
                  <div className="text-xl font-semibold">
                    {latest === null || latest.helmet_detected === null
                      ? "No data"
                      : latest.helmet_detected
                      ? "✅ All helmets detected"
                      : "🚨 Some missing helmets"}
                  </div>
                  <div className="text-xs text-slate-400">
                    {latest?.timestamp ? new Date(latest.timestamp).toLocaleString() : "---"}
                  </div>
                </div>
              </div>
            </div>

            {/* Recent per-person alerts */}
            <div>
              <div className="text-sm text-slate-400 mb-2">Recent history (last 10)</div>
              <div className="space-y-2 max-h-[320px] overflow-auto pr-2">
                {history.length === 0 && (
                  <div className="text-sm text-slate-500">No history yet</div>
                )}
                {history.map((alert, i) => (
                  <div
                    key={i}
                    className="bg-slate-700/30 p-2 rounded"
                  >
                    <div className="text-xs text-slate-400 mb-1">
                      {alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : "---"}
                    </div>
                    {alert.alerts.length === 0 && (
                      <div className="text-sm">No persons detected</div>
                    )}
                    {alert.alerts.map((personAlert, pi) => (
                      <div
                        key={pi}
                        className={`flex items-center justify-between p-1 rounded ${
                          personAlert.helmet_detected
                            ? "bg-green-700/50"
                            : "bg-red-700/50"
                        }`}
                      >
                        <div>
                          Person #{pi + 1}
                        </div>
                        <div className="font-semibold">
                          {personAlert.helmet_detected ? "✅ Helmet" : "🚨 No Helmet"}
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <footer className="mt-6 text-sm text-slate-500">
          Make sure your FastAPI backend permits CORS from{" "}
          <code>http://localhost:3000</code> and is reachable at{" "}
          <code>http://localhost:8000</code>.
        </footer>
      </div>
    </main>
  );
}

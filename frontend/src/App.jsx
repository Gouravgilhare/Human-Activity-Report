import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  Camera,
  ChartNoAxesCombined,
  Database,
  History,
  LayoutDashboard,
  Play,
  RefreshCw,
  Square,
  Target,
  Video,
  Wifi,
  WifiOff,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
const API = "http://127.0.0.1:8000";
const initialStats = {
  total_samples: 0,
  total_sessions: 0,
  activities: { sitting: 0, standing: 0, walking: 0 },
};
const navigation = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Live Detection", icon: Video },
  { label: "Dataset", icon: Database },
  { label: "Model Metrics", icon: ChartNoAxesCombined },
  { label: "History", icon: History },
];
function StatCard({ icon: Icon, label, value, detail }) {
  return (
    <div className="rounded-xl border border-[#25352b] bg-[#121b16] p-5">
      <div className="flex items-center justify-between gap-3 text-sm text-[#93a49a]">
        <span>{label}</span>
        <Icon size={19} className="text-emerald-300" />
      </div>
      <div className="mt-4 break-words text-2xl font-semibold tracking-tight text-white">
        {value}
      </div>
      <p className="mt-2 text-xs text-[#718278]">{detail}</p>
    </div>
  );
}
function Panel({ children, className = "" }) {
  return (
    <section
      className={`min-w-0 rounded-xl border border-[#25352b] bg-[#121b16] p-5 ${className}`}
    >
      {children}
    </section>
  );
}
function Detail({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-[#25352b] py-3 text-xs last:border-b-0">
      <span className="text-[#93a49a]">{label}</span>
      <strong className="break-words text-right text-[#e8f0eb]">{value}</strong>
    </div>
  );
}
export default function App() {
  const [page, setPage] = useState("Overview");
  const [online, setOnline] = useState(false);
  const [cameraRunning, setCameraRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("Connecting to Python backend...");
  const [stats, setStats] = useState(initialStats);
  const [metrics, setMetrics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [lastUpdated, setLastUpdated] = useState(null);
  const loadDashboard = useCallback(async (showConnectionMessage = false) => {
    try {
      const healthResponse = await fetch(`${API}/api/health`);
      if (!healthResponse.ok) {
        throw new Error(`Backend returned HTTP ${healthResponse.status}`);
      }
      const health = await healthResponse.json();
      setOnline(health.status === "online");
      setCameraRunning(Boolean(health.camera_running));
      if (showConnectionMessage) {
        setMessage("Python backend connected successfully.");
      }
      // Optional dashboard endpoints should not override a successful health check.
      const endpointResults = await Promise.allSettled([
        fetch(`${API}/api/stats`),
        fetch(`${API}/api/metrics`),
        fetch(`${API}/api/history`),
        fetch(`${API}/api/prediction`),
      ]);
      const [statsResult, metricsResult, historyResult, predictionResult] = endpointResults;
      if (statsResult.status === "fulfilled" && statsResult.value.ok) {
        const result = await statsResult.value.json();
        setStats({
          ...initialStats,
          ...result,
          activities: result.activities ?? initialStats.activities,
        });
      }
      if (metricsResult.status === "fulfilled" && metricsResult.value.ok) {
        setMetrics(await metricsResult.value.json());
      }
      if (historyResult.status === "fulfilled" && historyResult.value.ok) {
        const result = await historyResult.value.json();
        setHistory(Array.isArray(result) ? result : result.items ?? []);
      }
      if (predictionResult.status === "fulfilled" && predictionResult.value.ok) {
        const result = await predictionResult.value.json();
        setPrediction(result?.activity ? result : null);
        if (result?.camera_running !== undefined) {
          setCameraRunning(Boolean(result.camera_running));
        }
      }
      setLastUpdated(new Date());
    } catch (error) {
      setOnline(false);
      setCameraRunning(false);
      setMessage(
        error?.message
          ? `Python backend unavailable: ${error.message}`
          : "Python backend is offline. Start the API to load live data.",
      );
    }
  }, []);
  async function controlDetection(action) {
    setBusy(true);
    setMessage(`${action === "start" ? "Starting" : "Stopping"} detection...`);
    try {
      const response = await fetch(`${API}/api/detection/${action}`, {
        method: "POST",
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || result.status === "error") {
        throw new Error(result.message || result.detail || "Detection request failed.");
      }
      setMessage(result.message || "Detection request accepted.");
      if (action === "start") setCameraRunning(true);
      if (action === "stop") setCameraRunning(false);
      await loadDashboard();
    } catch (error) {
      setMessage(error.message || "Could not contact the Python backend.");
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    loadDashboard(true);
    const interval = window.setInterval(() => loadDashboard(), 3000);
    return () => window.clearInterval(interval);
  }, [loadDashboard]);
  const activities = useMemo(
    () =>
      Object.entries(stats.activities || {}).map(([activity, samples]) => ({
        activity: activity.charAt(0).toUpperCase() + activity.slice(1),
        samples: Number(samples) || 0,
      })),
    [stats.activities],
  );
  const accuracy =
    typeof metrics?.accuracy === "number"
      ? `${(metrics.accuracy * 100).toFixed(1)}%`
      : "—";
  const confidence =
    typeof prediction?.confidence === "number"
      ? `${(prediction.confidence * 100).toFixed(1)}%`
      : "—";
  return (
    <div className="min-h-screen bg-[#0b110e] text-[#e8f0eb] md:flex">
      <aside className="flex w-full shrink-0 flex-col border-b border-[#25352b] bg-[#0e1511] p-4 md:min-h-screen md:w-60 md:border-r md:border-b-0 md:p-5">
        <div className="flex items-center gap-3 px-2 py-3 md:pb-9">
          <div className="grid size-11 place-items-center rounded-xl bg-emerald-950 text-emerald-300">
            <BrainCircuit size={24} />
          </div>
          <div>
            <h1 className="font-semibold tracking-tight">HAR Studio</h1>
            <p className="mt-1 text-xs text-[#93a49a]">Activity intelligence</p>
          </div>
        </div>
        <p className="mb-3 mt-4 px-3 text-[10px] font-bold tracking-[0.18em] text-[#789487]">
          WORKSPACE
        </p>
        <nav className="grid grid-cols-2 gap-1 md:grid-cols-1">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              onClick={() => setPage(label)}
              className={`flex items-center gap-3 rounded-lg border px-3 py-3 text-left text-sm transition ${
                page === label
                  ? "border-[#294535] bg-[#182a20] text-emerald-200"
                  : "border-transparent text-[#a4b3aa] hover:bg-[#17221b]"
              }`}
            >
              <Icon size={18} />
              {label}
            </button>
          ))}
        </nav>
        <div className="mt-5 rounded-xl border border-[#25352b] p-3 md:mt-auto">
          <div className="flex items-center gap-2 text-xs font-medium">
            {online ? (
              <Wifi size={14} className="text-emerald-300" />
            ) : (
              <WifiOff size={14} className="text-amber-400" />
            )}
            {online ? "API connected" : "API disconnected"}
          </div>
          <p className="mt-2 text-[11px] text-[#93a49a]">
            {online ? "Python service online" : "Start the FastAPI server"}
          </p>
        </div>
        <p className="mt-4 px-1 text-[10px] text-[#617369]">
          HAR Project · Local environment
        </p>
      </aside>
      <main className="mx-auto w-full max-w-[1600px] min-w-0 flex-1 p-4 sm:p-6 lg:p-9">
        <header className="mb-6 flex items-center justify-between gap-4">
          <div>
            <p className="mb-2 text-xs text-[#93a49a]">Workspace / {page}</p>
            <h2 className="text-2xl font-semibold tracking-tight">{page}</h2>
            <p className="mt-2 text-[11px] text-[#718278]">
              {lastUpdated
                ? `Last refreshed at ${lastUpdated.toLocaleTimeString()}`
                : "Waiting for first API response"}
            </p>
          </div>
          <button
            onClick={() => loadDashboard(true)}
            className="flex items-center gap-2 rounded-lg border border-[#25352b] bg-[#121b16] px-3 py-2 text-xs hover:bg-[#17221b]"
          >
            <RefreshCw size={15} />
            Refresh
          </button>
        </header>
        <div
          className={`mb-6 flex items-center gap-3 rounded-lg border px-4 py-3 text-xs ${
            online
              ? "border-[#294b37] bg-[#12271c] text-emerald-200"
              : "border-[#453b2a] bg-[#211c13] text-amber-200"
          }`}
        >
          <span
            className={`size-2 shrink-0 rounded-full ${
              online ? "bg-emerald-300" : "bg-amber-400"
            }`}
          />
          <span>{message}</span>
        </div>
        {page === "Overview" && (
          <>
            <section className="mb-5 flex items-center justify-between gap-5 rounded-2xl border border-[#294637] bg-gradient-to-r from-[#17271e] to-[#122019] p-6 sm:p-8">
              <div>
                <p className="text-[10px] font-bold tracking-[0.18em] text-emerald-300">
                  HUMAN ACTIVITY RECOGNITION
                </p>
                <h3 className="mt-3 max-w-xl text-2xl font-semibold tracking-tight sm:text-3xl">
                  Understand movement in real time.
                </h3>
                <p className="mt-3 max-w-xl text-sm leading-6 text-[#93a49a]">
                  Monitor predictions, review your dataset, and evaluate your trained
                  machine-learning model.
                </p>
              </div>
              <div className="hidden size-16 shrink-0 place-items-center rounded-2xl bg-[#18372a] text-emerald-300 sm:grid">
                <Activity size={34} />
              </div>
            </section>
            <section className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <StatCard
                icon={Database}
                label="Dataset samples"
                value={(stats.total_samples ?? 0).toLocaleString()}
                detail="Recorded pose frames"
              />
              <StatCard
                icon={LayoutDashboard}
                label="Sessions"
                value={(stats.total_sessions ?? 0).toLocaleString()}
                detail="Unique recording sessions"
              />
              <StatCard
                icon={Target}
                label="Test accuracy"
                value={accuracy}
                detail="Saved model evaluation"
              />
              <StatCard
                icon={Activity}
                label="Current activity"
                value={prediction?.activity || "Waiting"}
                detail={`Confidence: ${confidence}`}
              />
            </section>
            <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              <Panel>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="font-semibold">Live detection</h3>
                    <p className="mt-1 text-xs text-[#93a49a]">
                      Control the webcam activity recognizer
                    </p>
                  </div>
                  <span className="flex items-center gap-2 text-[10px] font-bold tracking-wider">
                    <span
                      className={`size-2 rounded-full ${
                        cameraRunning ? "bg-emerald-300" : "bg-[#68776e]"
                      }`}
                    />
                    {cameraRunning ? "RUNNING" : online ? "READY" : "OFFLINE"}
                  </span>
                </div>
                <div className="relative mt-5 flex min-h-56 items-center justify-center overflow-hidden rounded-xl border border-[#34473a] bg-[#070b08]">
                  {cameraRunning ? (
                    <img
                      key="har-live-video"
                      src={`${API}/api/video`}
                      alt="Live webcam with pose detection"
                      className="absolute inset-0 h-full w-full object-contain"
                    />
                  ) : (
                    <div className="flex flex-col items-center p-6 text-center">
                      <Camera size={34} className="mb-3 text-emerald-300" />
                      <strong className="text-sm">Camera is not running</strong>
                      <p className="mt-2 max-w-xs text-xs leading-5 text-[#93a49a]">
                        Start detection to open your webcam and see the live pose
                        prediction here.
                      </p>
                    </div>
                  )}
                  {cameraRunning && prediction && (
                    <div className="absolute bottom-3 left-3 rounded-lg border border-emerald-900 bg-[#0b110e]/90 px-3 py-2 text-xs text-emerald-200">
                      {prediction.activity} · {confidence}
                    </div>
                  )}
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    disabled={!online || busy || cameraRunning}
                    onClick={() => controlDetection("start")}
                    className="flex items-center gap-2 rounded-lg bg-emerald-300 px-4 py-2.5 text-xs font-semibold text-[#0d2116] hover:bg-emerald-200 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    <Play size={15} /> {busy ? "Please wait..." : "Start detection"}
                  </button>
                  <button
                    disabled={!online || busy || !cameraRunning}
                    onClick={() => controlDetection("stop")}
                    className="flex items-center gap-2 rounded-lg border border-[#25352b] bg-[#17221b] px-4 py-2.5 text-xs hover:bg-[#203127] disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    <Square size={14} /> Stop
                  </button>
                </div>
              </Panel>
              <Panel>
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold">Dataset distribution</h3>
                    <p className="mt-1 text-xs text-[#93a49a]">
                      Number of samples by activity
                    </p>
                  </div>
                  <Database size={19} className="text-[#93a49a]" />
                </div>
                <div className="mt-5 h-72 w-full">
                  {activities.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={activities}
                        margin={{ top: 12, right: 8, left: -18, bottom: 0 }}
                      >
                        <CartesianGrid
                          stroke="#26332e"
                          strokeDasharray="3 3"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="activity"
                          tick={{ fill: "#9baaa2", fontSize: 12 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          tick={{ fill: "#9baaa2", fontSize: 11 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#17211c",
                            border: "1px solid #34443b",
                            borderRadius: 10,
                          }}
                        />
                        <Bar dataKey="samples" fill="#67d8a7" radius={[5, 5, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="grid h-full place-items-center text-sm text-[#718278]">
                      Dataset data will appear when the API is connected.
                    </div>
                  )}
                </div>
              </Panel>
            </section>
          </>
        )}
        {page === "Live Detection" && (
          <Panel>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h3 className="font-semibold">Live activity detection</h3>
                <p className="mt-2 text-xs text-[#93a49a]">
                  The Python backend controls the webcam and runs pose inference.
                </p>
              </div>
              <span className="rounded-full border border-[#294b37] px-3 py-1 text-xs text-emerald-200">
                {cameraRunning ? "Camera running" : "Camera stopped"}
              </span>
            </div>
            <div className="relative mt-5 flex aspect-video min-h-[480px] max-h-[78vh] w-full items-center justify-center overflow-hidden rounded-xl border border-[#34473a] bg-black lg:min-h-[600px]">
              {cameraRunning ? (
                <img
                  src={`${API}/api/video`}
                  alt="Live webcam with pose detection"
                  className="absolute inset-0 h-full w-full object-contain"
                />
              ) : (
                <div className="p-6 text-center">
                  <Camera size={36} className="mx-auto mb-3 text-emerald-300" />
                  <p className="text-sm">Start detection to activate the webcam.</p>
                </div>
              )}
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                disabled={!online || busy }
                onClick={() => controlDetection("start")}
                className="flex items-center gap-2 rounded-lg bg-emerald-300 px-4 py-2 text-xs font-semibold text-[#0d2116] disabled:opacity-40"
              >
                <Play size={15} /> Start detection
              </button>
              <button
                disabled={!online || busy || !cameraRunning}
                onClick={() => controlDetection("stop")}
                className="flex items-center gap-2 rounded-lg border border-[#25352b] px-4 py-2 text-xs disabled:opacity-40"
              >
                <Square size={14} /> Stop
              </button>
            </div>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <Detail label="Prediction" value={prediction?.activity || "Waiting for prediction"} />
              <Detail label="Confidence" value={confidence} />
            </div>
          </Panel>
        )}
        {page === "Dataset" && (
          <Panel>
            <h3 className="font-semibold">Dataset overview</h3>
            <p className="mt-2 text-xs text-[#93a49a]">Recorded pose samples from your CSV file.</p>
            <div className="mt-4">
              <Detail label="Total samples" value={(stats.total_samples ?? 0).toLocaleString()} />
              <Detail label="Recording sessions" value={(stats.total_sessions ?? 0).toLocaleString()} />
              {activities.map((item) => (
                <Detail
                  key={item.activity}
                  label={`${item.activity} samples`}
                  value={item.samples.toLocaleString()}
                />
              ))}
            </div>
            {stats.error && <p className="mt-4 text-xs text-amber-300">{stats.error}</p>}
          </Panel>
        )}
        {page === "Model Metrics" && (
          <Panel>
            <h3 className="font-semibold">Model evaluation</h3>
            <p className="mt-2 text-xs text-[#93a49a]">
              Metrics loaded from reports/metrics.txt.
            </p>
            <div className="mt-4">
              <Detail label="Test accuracy" value={accuracy} />
              <Detail label="Evaluation method" value="Session-based (as recorded in report)" />
            </div>
            <pre className="mt-5 max-h-[600px] overflow-auto whitespace-pre-wrap break-words rounded-lg bg-[#0b120e] p-4 text-xs leading-5 text-emerald-200">
              {metrics?.report ||
                "No evaluation report found yet. Run your training script to generate reports/metrics.txt."}
            </pre>
          </Panel>
        )}
        {page === "History" && (
          <Panel>
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="font-semibold">Prediction history</h3>
                <p className="mt-2 text-xs text-[#93a49a]">
                  Recent predictions from the running Python process.
                </p>
              </div>
              <History size={19} className="text-[#93a49a]" />
            </div>
            {history.length === 0 ? (
              <div className="flex items-center gap-3 py-10 text-sm text-[#93a49a]">
                <History size={22} />
                No prediction history yet. Start detection and the predictions will appear here.
              </div>
            ) : (
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[420px] text-left text-xs">
                  <thead className="text-[#93a49a]">
                    <tr className="border-b border-[#25352b]">
                      <th className="py-3 pr-4 font-medium">Time</th>
                      <th className="py-3 pr-4 font-medium">Activity</th>
                      <th className="py-3 text-right font-medium">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...history].reverse().map((item, index) => (
                      <tr key={`${item.timestamp ?? item.time ?? "entry"}-${index}`} className="border-b border-[#25352b]/70">
                        <td className="py-3 pr-4 text-[#93a49a]">
                          {item.time ||
                            (item.timestamp
                              ? new Date(item.timestamp * 1000).toLocaleTimeString()
                              : "—")}
                        </td>
                        <td className="py-3 pr-4 font-medium">{item.activity || "—"}</td>
                        <td className="py-3 text-right text-emerald-200">
                          {typeof item.confidence === "number"
                            ? `${(item.confidence * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        )}
        <footer className="mt-7 flex flex-wrap justify-between gap-3 border-t border-[#25352b] pt-5 text-[10px] text-[#718278]">
          <span>HAR Studio · Human Activity Recognition</span>
          <span>Local Python API · {online ? "Connected" : "Disconnected"}</span>
        </footer>
      </main>
    </div>
  );
}

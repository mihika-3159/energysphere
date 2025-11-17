import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  Legend,
} from "recharts";

const API_URL = import.meta.env.VITE_API_URL;

// ---------- small helpers ----------
function formatNumber(n) {
  return n.toLocaleString(undefined, {
    maximumFractionDigits: 1,
  });
}

// simple outage risk based on peak vs average
function computeOutageRisk(forecast) {
  if (!forecast || forecast.length === 0) return { level: "Unknown", score: 0 };
  const maxVal = Math.max(...forecast);
  const avgVal = forecast.reduce((a, b) => a + b, 0) / forecast.length;
  const ratio = maxVal / (avgVal || 1);

  if (ratio < 1.1) return { level: "Low", score: 25 };
  if (ratio < 1.3) return { level: "Medium", score: 55 };
  return { level: "High", score: 85 };
}

// rough CO2 savings vs fossil baseline
function computeCO2Impact(forecast) {
  if (!forecast || forecast.length === 0) {
    return { savings: 0, baseline: 0, pct: 0 };
  }
  const totalDemand = forecast.reduce((a, b) => a + b, 0);
  const baselineEmissionFactor = 0.7; // tCO2/MWh for fossil heavy mix
  const greenEmissionFactor = 0.25;   // tCO2/MWh for clean mix

  const baseline = totalDemand * baselineEmissionFactor;
  const green = totalDemand * greenEmissionFactor;
  const savings = baseline - green;
  const pct = baseline ? (savings / baseline) * 100 : 0;

  return { savings, baseline, pct };
}

// simple fake “projects” for investment optimizer (Phase 5)
const PROJECTS = [
  {
    id: "solar-city-roof",
    name: "Urban Rooftop Solar Initiative",
    type: "Solar PV",
    minBudget: 100_000,
    risk: "Low",
    roi: 0.10,
    co2: 4200, // tCO2/year avoided
  },
  {
    id: "coastal-wind",
    name: "Coastal Wind Farm Expansion",
    type: "Onshore Wind",
    minBudget: 250_000,
    risk: "Medium",
    roi: 0.15,
    co2: 9800,
  },
  {
    id: "community-microgrid",
    name: "Community Microgrid for Resilience",
    type: "Solar + Storage",
    minBudget: 75_000,
    risk: "Medium",
    roi: 0.12,
    co2: 3100,
  },
  {
    id: "battery-hub",
    name: "Battery Storage Hub",
    type: "Grid Storage",
    minBudget: 150_000,
    risk: "High",
    roi: 0.18,
    co2: 5600,
  },
];

function filterProjects(budget, risk) {
  return PROJECTS.filter((p) => {
    if (budget < p.minBudget) return false;
    if (risk === "Low" && p.risk === "High") return false;
    if (risk === "Medium" && p.risk === "High" && budget < 300_000) return false;
    return true;
  });
}

// ---------- main dashboard ----------
export default function Dashboard() {
  const [forecast, setForecast] = useState([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [investmentBudget, setInvestmentBudget] = useState(200_000);
  const [riskLevel, setRiskLevel] = useState("Medium");
  const [userShiftPct, setUserShiftPct] = useState(20); // % of own usage shifting off-peak

  useEffect(() => {
    async function fetchForecast() {
      try {
        setLoading(true);
        const res = await axios.get(`${API_URL}/predict`);
        const data = res.data;

        const raw = Array.isArray(data.forecast)
          ? data.forecast
          : [data.forecast];

        // optional scaling so it looks like MW range
        const scaled = raw.map((v) => v * 150 + 200); // from ~1.6 → ~440 MW
        setForecast(scaled);
        setStatus(data.status || "Model online");
        setError("");
      } catch (err) {
        console.error("API error:", err);
        setError("Could not load forecast. Please try again later.");
      } finally {
        setLoading(false);
      }
    }

    fetchForecast();
  }, []);

  const chartData = useMemo(() => {
    return forecast.map((val, idx) => ({
      hour: idx,
      demand: Number(val.toFixed(1)),
    }));
  }, [forecast]);

  const stats = useMemo(() => {
    if (!forecast.length) return null;
    const max = Math.max(...forecast);
    const min = Math.min(...forecast);
    const avg = forecast.reduce((a, b) => a + b, 0) / forecast.length;
    const peakHour = forecast.indexOf(max);
    return { max, min, avg, peakHour };
  }, [forecast]);

  const outageRisk = useMemo(
    () => computeOutageRisk(forecast),
    [forecast]
  );

  const co2 = useMemo(
    () => computeCO2Impact(forecast),
    [forecast]
  );

  const suggestedProjects = useMemo(
    () => filterProjects(investmentBudget, riskLevel),
    [investmentBudget, riskLevel]
  );

  const citizenSavings = useMemo(() => {
    if (!forecast.length) return { kwh: 0, co2: 0 };
    const avgDemand = forecast.reduce((a, b) => a + b, 0) / forecast.length;
    // assume user uses 20 kWh/day, shifting % of that from peak to off-peak
    const baseUse = 20; // kWh/day
    const shiftedKwh = (baseUse * userShiftPct) / 100;
    const emissionFactor = 0.4; // tCO2/MWh
    const co2Saved = (shiftedKwh / 1000) * emissionFactor;
    return { kwh: shiftedKwh, co2: co2Saved };
  }, [forecast, userShiftPct]);

  return (
    <div className="es-root">
      {/* HEADER */}
      <header className="es-header">
        <div>
          <h1>EnergySphere</h1>
          <p className="es-subtitle">
            AI-powered energy demand forecasting & investment intelligence
          </p>
        </div>
        <div className="es-header-badges">
          <span className="es-badge es-badge-green">
            {status || "Model online"}
          </span>
          <span className="es-badge es-badge-blue">
            Backend: {API_URL ? "Connected" : "Offline"}
          </span>
        </div>
      </header>

      {/* MAIN GRID */}
      <main className="es-grid">
        {/* LEFT: FORECAST + STATS */}
        <section className="es-card es-card-wide">
          <div className="es-card-header">
            <div>
              <h2>24-hour Demand Forecast</h2>
              <p>Transformer-based prediction of short-term grid load</p>
            </div>
            {stats && (
              <div className="es-stats-row">
                <div className="es-stat">
                  <span>Peak</span>
                  <strong>{formatNumber(stats.max)} MW</strong>
                </div>
                <div className="es-stat">
                  <span>Average</span>
                  <strong>{formatNumber(stats.avg)} MW</strong>
                </div>
                <div className="es-stat">
                  <span>Peak Hour</span>
                  <strong>{stats.peakHour}:00</strong>
                </div>
              </div>
            )}
          </div>

          <div className="es-chart-wrapper">
            {loading ? (
              <div className="es-placeholder">Loading AI forecast…</div>
            ) : error ? (
              <div className="es-error">{error}</div>
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorDemand" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4ade80" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" tickFormatter={(h) => `${h}:00`} />
                  <YAxis />
                  <Tooltip
                    formatter={(v) => [`${formatNumber(v)} MW`, "Demand"]}
                    labelFormatter={(h) => `Hour ${h}:00`}
                  />
                  <Area
                    type="monotone"
                    dataKey="demand"
                    stroke="#16a34a"
                    fillOpacity={1}
                    fill="url(#colorDemand)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </section>

        {/* RIGHT COLUMN: RISK + CO2 + CITIZEN */}
        <section className="es-column">
          {/* GRID RISK */}
          <div className="es-card">
            <h3>Grid Stability & Outage Risk</h3>
            <p className="es-muted">
              Derived from peak vs average load and short-term volatility.
            </p>
            <div className="es-risk-row">
              <div className={`es-risk-tag es-risk-${outageRisk.level.toLowerCase()}`}>
                {outageRisk.level} risk
              </div>
              <div className="es-risk-bar">
                <div
                  className="es-risk-bar-fill"
                  style={{ width: `${outageRisk.score}%` }}
                />
              </div>
            </div>
            <ul className="es-list">
              {outageRisk.level === "High" && (
                <>
                  <li>Consider activating demand response programs.</li>
                  <li>Delay non-critical industrial loads during peak hours.</li>
                </>
              )}
              {outageRisk.level === "Medium" && (
                <>
                  <li>Encourage voluntary off-peak usage for citizens.</li>
                  <li>Prepare spinning reserves and storage dispatch.</li>
                </>
              )}
              {outageRisk.level === "Low" && (
                <>
                  <li>Grid conditions are healthy.</li>
                  <li>Ideal time to charge storage and EV fleets.</li>
                </>
              )}
            </ul>
          </div>

          {/* CO2 IMPACT */}
          <div className="es-card">
            <h3>CO₂ Impact of Current Mix</h3>
            <p className="es-muted">
              Comparing a fossil-heavy baseline vs an AI-optimized clean mix.
            </p>
            <div className="es-co2-row">
              <div>
                <span>Annualised savings</span>
                <strong>{formatNumber(co2.savings)} tCO₂</strong>
              </div>
              <div>
                <span>Cleaner than baseline</span>
                <strong>{co2.pct.toFixed(1)}%</strong>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart
                data={[
                  { name: "Baseline", emissions: co2.baseline },
                  { name: "With EnergySphere", emissions: co2.baseline - co2.savings },
                ]}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(v) => [`${formatNumber(v)} tCO₂`, "Emissions"]} />
                <Legend />
                <Bar dataKey="emissions" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* CITIZEN OFF-PEAK */}
          <div className="es-card">
            <h3>Citizen Off-peak Participation</h3>
            <p className="es-muted">
              Model how individual behaviour scales to system-wide impact.
            </p>
            <label className="es-slider-label">
              Shifted household usage:{" "}
              <strong>{userShiftPct}%</strong>
            </label>
            <input
              type="range"
              min={0}
              max={50}
              value={userShiftPct}
              onChange={(e) => setUserShiftPct(Number(e.target.value))}
            />
            <div className="es-co2-row">
              <div>
                <span>Daily usage shifted</span>
                <strong>{citizenSavings.kwh.toFixed(1)} kWh</strong>
              </div>
              <div>
                <span>CO₂ saved per day</span>
                <strong>{citizenSavings.co2.toFixed(3)} tCO₂</strong>
              </div>
            </div>
            <p className="es-muted-small">
              At scale, if 100,000 households opt in, this scenario saves{" "}
              <strong>
                {(citizenSavings.co2 * 100_000).toFixed(0)} tCO₂
              </strong>{" "}
              per day.
            </p>
          </div>
        </section>

        {/* FULL-WIDTH INVESTMENT PANEL */}
        <section className="es-card es-card-wide">
          <div className="es-card-header">
            <div>
              <h2>AI-Guided Investment Opportunities</h2>
              <p>
                Explore high-impact renewable projects based on your budget and risk appetite.
              </p>
            </div>
          </div>

          <div className="es-investment-grid">
            <div className="es-investment-controls">
              <label>
                Total budget (USD)
                <input
                  type="number"
                  value={investmentBudget}
                  onChange={(e) =>
                    setInvestmentBudget(
                      Math.max(10_000, Number(e.target.value) || 0)
                    )
                  }
                />
              </label>

              <label>
                Risk preference
                <select
                  value={riskLevel}
                  onChange={(e) => setRiskLevel(e.target.value)}
                >
                  <option value="Low">Low (public funds, city bonds)</option>
                  <option value="Medium">Medium (utilities, infra funds)</option>
                  <option value="High">High (VC / private)</option>
                </select>
              </label>

              <p className="es-muted-small">
                These are model-derived recommendations using heuristic risk/ROI scoring;
                in a full deployment, they would be backed by live market & grid data.
              </p>
            </div>

            <div className="es-investment-list">
              {suggestedProjects.length === 0 ? (
                <div className="es-placeholder">
                  Increase your budget or risk tolerance to see investable projects.
                </div>
              ) : (
                suggestedProjects.map((p) => (
                  <div className="es-invest-card" key={p.id}>
                    <h4>{p.name}</h4>
                    <p className="es-pill">{p.type}</p>
                    <div className="es-invest-meta">
                      <span>Min. ticket: ${p.minBudget.toLocaleString()}</span>
                      <span>Risk: {p.risk}</span>
                      <span>Est. ROI: {(p.roi * 100).toFixed(1)}% / yr</span>
                    </div>
                    <p className="es-muted-small">
                      CO₂ impact:{" "}
                      <strong>{p.co2.toLocaleString()} tCO₂/yr avoided</strong>
                    </p>
                    <button className="es-btn-outline">
                      View impact breakdown
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

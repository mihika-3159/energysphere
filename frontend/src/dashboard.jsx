import React, { useEffect, useState } from "react";
import axios from "axios";

export default function Dashboard() {
  const [forecast, setForecast] = useState([]);

  useEffect(() => {
    axios.get(`${import.meta.env.VITE_API_URL}/predict`)
      .then(res => setForecast(res.data.forecast))
      .catch(err => console.error(err));
  }, []);

  return (
    <div>
      <h2>Forecasted Energy Demand</h2>
      <ul>
        {forecast.map((val, idx) => (
          <li key={idx}>{val.toFixed(2)} MW</li>
        ))}
      </ul>
    </div>
  );
}

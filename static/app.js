document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const presetSelect = document.getElementById("route-preset");
  const distanceSlider = document.getElementById("distance-slider");
  const distanceInput = document.getElementById("route-distance");
  const distanceReadout = document.getElementById("distance-readout");
  const timeInput = document.getElementById("time-of-day");
  const trafficInput = document.getElementById("traffic-category");
  const prevTimeInput = document.getElementById("prev-time");
  const errorBanner = document.getElementById("form-error");
  const historyBody = document.getElementById("history-body");
  const clearHistoryBtn = document.getElementById("clear-history");

  const historyItems = [];

  function syncDistance(value) {
    const num = Number(value);
    if (!Number.isNaN(num)) {
      distanceInput.value = num;
      distanceSlider.value = Math.min(12.0, Math.max(0.5, num));
      distanceReadout.textContent = `${num.toFixed(1)} km`;
    }
  }

  distanceSlider.addEventListener("input", (e) => syncDistance(e.target.value));
  distanceInput.addEventListener("input", (e) => syncDistance(e.target.value));

  presetSelect.addEventListener("change", (e) => {
    const selected = e.target.selectedOptions[0];
    if (!selected || !selected.value) return;
    syncDistance(selected.dataset.distance);
    prevTimeInput.value = selected.dataset.prev;
    runPrediction();
  });

  document.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      timeInput.value = btn.dataset.time;
    });
  });

  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      trafficInput.value = btn.dataset.traffic;
    });
  });

  clearHistoryBtn.addEventListener("click", () => {
    historyItems.length = 0;
    renderHistory();
  });

  function renderHistory() {
    if (historyItems.length === 0) {
      historyBody.innerHTML =
        '<tr class="empty-row"><td colspan="6">No estimates run yet in this session.</td></tr>';
      return;
    }
    historyBody.innerHTML = historyItems
      .map(
        (item) => `
      <tr>
        <td>${item.departure_time}</td>
        <td>${item.route_distance_km.toFixed(1)} km</td>
        <td>${item.traffic_category}</td>
        <td>${item.previous_travel_time_min.toFixed(1)} min</td>
        <td><strong>${item.estimated_minutes.toFixed(1)} min</strong></td>
        <td>${item.estimated_arrival_clock}</td>
      </tr>`
      )
      .join("");
  }

  function renderPrediction(data) {
    document.getElementById("eta-minutes").textContent = data.estimated_minutes.toFixed(1);
    document.getElementById("eta-clock").textContent = data.estimated_arrival_clock;
    document.getElementById("eta-range").textContent =
      `${data.lower_bound_minutes.toFixed(1)} - ${data.upper_bound_minutes.toFixed(1)} min`;

    const trafficBadgeClass = `badge-${data.traffic_category.toLowerCase()}`;
    const peakBadge = data.is_peak_hour
      ? '<span class="badge badge-peak">Campus Peak Hour</span>'
      : '<span class="badge badge-neutral">Off-Peak Window</span>';

    document.getElementById("eta-badges").innerHTML = `
      <span class="badge ${trafficBadgeClass}">${data.traffic_category} Traffic</span>
      ${peakBadge}
    `;

    const list = document.getElementById("explanation-list");
    list.innerHTML = data.explanation.map((line) => `<li>${line}</li>`).join("");

    historyItems.unshift(data);
    if (historyItems.length > 6) historyItems.pop();
    renderHistory();
  }

  async function loadMetadata() {
    try {
      const res = await fetch("/api/metadata");
      if (!res.ok) return;
      const meta = await res.json();
      const tm = meta.test_metrics || {};
      document.getElementById("header-model-pill").textContent = meta.best_model || "Regression Model";
      document.getElementById("stat-model").textContent = meta.best_model || "-";
      document.getElementById("stat-mae").textContent = tm.mae != null ? `${tm.mae.toFixed(2)} min` : "-";
      document.getElementById("stat-rmse").textContent = tm.rmse != null ? `${tm.rmse.toFixed(2)} min` : "-";
      document.getElementById("stat-r2").textContent = tm.r2 != null ? tm.r2.toFixed(3) : "-";

      const bars = document.getElementById("importance-bars");
      const items = (meta.feature_importance || []).slice(0, 5);
      const maxImp = Math.max(...items.map((x) => x.importance), 0.01);
      bars.innerHTML = items
        .map((item) => {
          const pct = Math.round((item.importance / maxImp) * 100);
          return `
          <div class="bar-row">
            <span>${item.feature}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
            <span>${(item.importance * 100).toFixed(0)}%</span>
          </div>`;
        })
        .join("");
    } catch {
      document.getElementById("header-model-pill").textContent = "Model Ready";
    }
  }

  async function runPrediction() {
    errorBanner.hidden = true;
    const payload = {
      route_distance_km: Number(distanceInput.value),
      time_of_day: timeInput.value,
      traffic_category: trafficInput.value,
      previous_travel_time_min: Number(prevTimeInput.value),
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        errorBanner.textContent = data.error || "Invalid input values.";
        errorBanner.hidden = false;
        return;
      }
      renderPrediction(data);
    } catch {
      errorBanner.textContent = "Could not reach prediction server.";
      errorBanner.hidden = false;
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    runPrediction();
  });

  loadMetadata();
  runPrediction();
});

// MediTrack front desk UI. No framework: small DOM helpers, text set via textContent (no innerHTML of data).
"use strict";

const $ = (sel) => document.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
}

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
    } catch { /* keep status text */ }
    throw new Error(detail);
  }
  return res.json();
}

const pct = (p) => `${Math.round(p * 100)}%`;
const todayIso = () => new Date().toLocaleDateString("en-CA"); // YYYY-MM-DD in local time
const prettyDate = (iso) =>
  new Date(`${iso}T00:00`).toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long" });
const hhmm = (iso) => iso.slice(11, 16);
let baseRate = 0.16;

// ---------------------------------------------------------------- tabs
const tabs = [...document.querySelectorAll('[role="tab"]')];
const loaders = { "view-appointments": loadAppointments, "view-dashboard": loadDashboard };

function selectTab(tab) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute("aria-selected", on);
    t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
  }
  const view = tab.getAttribute("aria-controls");
  history.replaceState(null, "", `#${view.replace("view-", "")}`);
  loaders[view]?.();
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => selectTab(t));
  t.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!step) return;
    const next = tabs[(i + step + tabs.length) % tabs.length];
    next.focus();
    selectTab(next);
  });
});

// ---------------------------------------------------------------- shared risk meter
function riskMeter(p, band) {
  return el("div", { class: "risk" },
    el("div", { class: "risk-head" },
      el("span", { class: `badge ${band}`, text: `${band[0].toUpperCase()}${band.slice(1)} risk` }),
      el("span", { class: "risk-pct", text: pct(p) })),
    el("div", { class: "meter", role: "img", "aria-label": `${pct(p)} chance of no-show; clinic average ${pct(baseRate)}` },
      el("div", { class: `meter-fill ${band}`, style: `width:${Math.max(2, p * 100)}%` }),
      el("div", { class: "meter-avg", style: `left:calc(${baseRate * 100}% - 1px)`, title: "Clinic average" })));
}

// ---------------------------------------------------------------- assistant
const log = $("#chat-log");
const form = $("#chat-form");
const input = $("#chat-input");
const send = $("#chat-send");

function scrollToEnd() { log.scrollTop = log.scrollHeight; }

const SOURCE_NAMES = { llm: "Gemini", fallback: "Offline parser", cache: "Cache" };
function sourceLabel(r) {
  if (r.cached) return "Cached answer";
  if (r.source === "cache") return "Cached question, fresh data";
  return SOURCE_NAMES[r.source] ?? r.source;
}

function renderSlots(groups) {
  return el("div", { class: "slot-groups" }, groups.map((g) => {
    const limit = 12;
    const chips = el("div", { class: "chips" }, g.slots.slice(0, limit).map((s) => el("span", { class: "chip", text: s })));
    if (g.slots.length > limit) {
      const more = el("button", {
        class: "more", type: "button", text: `+${g.slots.length - limit} more`,
        onclick: () => { more.replaceWith(...g.slots.slice(limit).map((s) => el("span", { class: "chip", text: s }))); },
      });
      chips.append(more);
    }
    return el("div", { class: "slot-group" },
      el("p", {}, g.doctor, el("span", { text: ` · ${g.department}` })), chips);
  }));
}

function renderRisks(risks) {
  return el("ol", { class: "watchlist", style: "margin-top:10px" }, risks.slice(0, 8).map((r) =>
    el("li", {},
      el("time", { datetime: r.start_time, text: hhmm(r.start_time) }),
      el("span", { text: `${r.patient_name}, ${r.doctor}` }),
      el("span", { class: "pct", text: pct(r.no_show_probability) }))));
}

function addBotMessage(r) {
  const structured = r.results?.length || r.risks?.length;
  const msg = el("div", { class: "msg bot" }, el("p", { class: "text", text: structured && r.headline ? r.headline : r.reply }));
  if (r.results?.length) msg.append(renderSlots(r.results));
  if (r.risks?.length) msg.append(renderRisks(r.risks));
  msg.append(el("p", { class: "meta" }, el("b", { text: sourceLabel(r) }), ` · ${r.latency_ms < 1000 ? `${Math.round(r.latency_ms)} ms` : `${(r.latency_ms / 1000).toFixed(1)} s`}`));
  log.append(msg);
}

async function ask(message) {
  $("#chat-intro")?.querySelector(".suggestions")?.remove();
  log.append(el("div", { class: "msg user", text: message }));
  const typing = el("p", { class: "msg bot typing", text: "Checking the schedule…" });
  log.append(typing);
  scrollToEnd();
  send.disabled = true;
  try {
    const r = await api("/api/chat", { method: "POST", body: JSON.stringify({ message }) });
    typing.remove();
    addBotMessage(r);
  } catch (err) {
    typing.remove();
    log.append(el("div", { class: "msg bot error" }, el("p", { class: "text", text: `Couldn't get an answer: ${err.message}` })));
  } finally {
    send.disabled = false;
    scrollToEnd();
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message || send.disabled) return;
  input.value = "";
  ask(message);
});
$("#suggestions").addEventListener("click", (e) => {
  if (e.target.matches("button")) ask(e.target.textContent);
});

async function loadGlance() {
  const day = todayIso();
  $("#glance-date").textContent = prettyDate(day);
  try {
    const appts = await api(`/api/appointments/upcoming?date=${day}`);
    const expected = appts.reduce((s, a) => s + a.no_show_probability, 0);
    $("#glance-booked").textContent = appts.length;
    $("#glance-expected").textContent = expected.toFixed(1);
    const watch = appts.filter((a) => a.risk === "high").sort((a, b) => b.no_show_probability - a.no_show_probability);
    const list = $("#glance-watch");
    list.replaceChildren(...(watch.length
      ? watch.slice(0, 6).map((a) => el("li", {},
          el("time", { datetime: a.start_time, text: hhmm(a.start_time) }),
          el("span", { text: a.patient_name }),
          el("span", { class: "pct", text: pct(a.no_show_probability) })))
      : [el("li", { class: "empty", text: "No high-risk appointments today." })]));
  } catch (err) {
    $("#glance-watch").replaceChildren(el("li", { class: "empty", text: `Couldn't load today: ${err.message}` }));
  }
}

// ---------------------------------------------------------------- appointments
const dateInput = $("#appt-date");
dateInput.value = todayIso();
dateInput.addEventListener("change", loadAppointments);
$("#risk-filter").addEventListener("change", loadAppointments);

async function loadAppointments() {
  const list = $("#appt-list");
  const risk = document.querySelector('input[name="risk"]:checked').value;
  const params = new URLSearchParams({ date: dateInput.value || todayIso() });
  if (risk) params.set("risk", risk);
  list.replaceChildren(el("li", { class: "empty", text: "Loading…" }));
  try {
    const appts = await api(`/api/appointments/upcoming?${params}`);
    if (!appts.length) {
      list.replaceChildren(el("li", { class: "empty", text: risk
        ? `No ${risk}-risk appointments on this day. Try another day or risk level.`
        : "No upcoming appointments on this day. Pick another date." }));
      return;
    }
    list.replaceChildren(...appts.map((a) => el("li", { class: `appt ${a.risk}` },
      el("time", { datetime: a.start_time, text: hhmm(a.start_time) }),
      el("div", {},
        el("p", { class: "who", text: a.patient_name }),
        el("p", { class: "doc", text: `${a.doctor} · ${a.department}` }),
        el("p", { class: "sms", text: a.sms_reminder ? "SMS reminder sent" : "No SMS reminder" })),
      el("div", { class: "reasons-cell" },
        el("p", { class: "reasons", text: a.reasons.length ? `Why: ${a.reasons.join("; ")}` : "No strong risk factors" })),
      riskMeter(a.no_show_probability, a.risk))));
  } catch (err) {
    list.replaceChildren(el("li", { class: "empty", text: `Couldn't load appointments: ${err.message}` }));
  }
}

// ---------------------------------------------------------------- dashboard
const charts = {};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function baseOptions({ percent = false, horizontal = false, legend = false } = {}) {
  const valueAxis = {
    beginAtZero: true,
    grid: { color: "#e6ecea" },
    border: { display: false },
    ticks: { color: css("--muted"), callback: percent ? (v) => `${Math.round(v * 100)}%` : undefined },
  };
  const catAxis = { grid: { display: false }, border: { color: css("--line") }, ticks: { color: css("--muted") } };
  return {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: horizontal ? "y" : "x",
    interaction: { mode: "index", intersect: false },
    scales: horizontal ? { x: valueAxis, y: catAxis } : { x: catAxis, y: valueAxis },
    plugins: {
      legend: { display: legend, position: "top", align: "start", labels: { color: css("--ink"), boxWidth: 12, boxHeight: 12 } },
      tooltip: {
        callbacks: percent ? { label: (c) => ` ${c.dataset.label}: ${pct(c.parsed[horizontal ? "x" : "y"])}` } : {},
      },
    },
  };
}

function drawChart(id, config) {
  charts[id]?.destroy();
  charts[id] = new Chart(document.getElementById(id), config);
}

const bar = (label, data, color, extra = {}) => ({
  label, data, backgroundColor: color, borderRadius: 4, borderSkipped: "start", maxBarThickness: 36, ...extra,
});

function tile(value, label, alert = false) {
  return el("div", { class: `tile${alert ? " alert" : ""}` }, el("p", { class: "value", text: value }), el("p", { class: "label", text: label }));
}

async function loadDashboard() {
  let s;
  try {
    s = await api("/api/stats");
  } catch (err) {
    $("#tiles").replaceChildren(el("p", { class: "empty", text: `Couldn't load statistics: ${err.message}` }));
    return;
  }
  const t = s.totals;
  baseRate = t.no_show_rate;
  $("#base-rate").textContent = pct(baseRate);
  $("#tiles").replaceChildren(
    tile(t.appointments.toLocaleString(), `Appointments since ${prettyDate(t.period[0]).split(" ").slice(1).join(" ")}`),
    tile(t.no_shows.toLocaleString(), "No-shows"),
    tile(pct(t.no_show_rate), "No-show rate"),
    tile(t.upcoming_booked.toLocaleString(), "Booked, next 7 days"),
    tile(t.upcoming_high_risk.toLocaleString(), "High risk, next 7 days", t.upcoming_high_risk > 0));

  const s1 = css("--series-1"), s2 = css("--series-2");
  const monthName = (m) => new Date(`${m}-01T00:00`).toLocaleDateString("en-GB", { month: "short", year: "2-digit" });

  drawChart("chart-monthly", {
    type: "bar",
    data: {
      labels: s.monthly.map((m) => monthName(m.month)),
      datasets: [
        bar("Attended", s.monthly.map((m) => m.appointments - m.no_shows), s1, { stack: "a" }),
        bar("No-show", s.monthly.map((m) => m.no_shows), s2, { stack: "a" }),
      ],
    },
    options: { ...baseOptions({ legend: true }), datasets: { bar: { borderWidth: { top: 2 }, borderColor: css("--surface") } } },
  });
  drawChart("chart-lead", {
    type: "bar",
    data: { labels: s.by_lead_time.map((d) => d.bucket), datasets: [bar("No-show rate", s.by_lead_time.map((d) => d.no_show_rate), s1)] },
    options: baseOptions({ percent: true }),
  });
  drawChart("chart-weekday", {
    type: "bar",
    data: { labels: s.by_weekday.map((d) => d.day), datasets: [bar("No-show rate", s.by_weekday.map((d) => d.no_show_rate), s1)] },
    options: baseOptions({ percent: true }),
  });
  drawChart("chart-dept", {
    type: "bar",
    data: { labels: s.by_department.map((d) => d.department), datasets: [bar("No-show rate", s.by_department.map((d) => d.no_show_rate), s1)] },
    options: baseOptions({ percent: true, horizontal: true }),
  });
  const dayLabel = (iso) => new Date(`${iso}T00:00`).toLocaleDateString("en-GB", { weekday: "short", day: "numeric" });
  drawChart("chart-upcoming", {
    type: "bar",
    data: {
      labels: s.upcoming.map((d) => dayLabel(d.date)),
      datasets: [
        bar("Booked", s.upcoming.map((d) => d.booked), s1),
        bar("Expected no-shows", s.upcoming.map((d) => d.expected_no_shows), s2),
      ],
    },
    options: baseOptions({ legend: true }),
  });

  const m = s.model;
  const modelName = { logistic_regression: "Logistic regression", gradient_boosting: "Gradient boosting" }[m.name] ?? m.name;
  $("#model-card").replaceChildren(
    el("dl", { class: "model-stats" },
      el("div", {}, el("dt", { text: "Model" }), el("dd", { class: "name", text: modelName })),
      el("div", {}, el("dt", { text: "ROC-AUC" }), el("dd", { text: m.roc_auc.toFixed(2) })),
      el("div", {}, el("dt", { text: "Brier (lower is better)" }), el("dd", { text: m.brier.toFixed(3) }))),
    el("p", { class: "model-note", text:
      `Scored on the most recent 20% of history. Guessing the average rate for everyone scores a Brier of ${m.baseline_brier.toFixed(3)}.` }));

  const c = s.chat;
  const hitRate = c.queries ? pct(c.cache_hits / c.queries) : "–";
  $("#usage").replaceChildren(
    ...[["Questions", c.queries], ["Cache hit rate", hitRate], ["Avg response", c.avg_latency_ms == null ? "–" : `${Math.round(c.avg_latency_ms)} ms`],
      ["Answered by Gemini", c.llm], ["Offline parser", c.fallback], ["Errors", c.errors]]
      .map(([k, v]) => el("div", {}, el("dt", { text: k }), el("dd", { text: String(v) }))));

  try {
    const logs = await api("/api/chat/logs?limit=15");
    $("#log-rows").replaceChildren(...(logs.length ? logs.map((l) => el("tr", {},
      el("td", {}, el("time", { datetime: l.timestamp, text: new Date(l.timestamp).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit" }) })),
      el("td", { class: "q", text: l.query }),
      el("td", { text: l.intent ?? "–" }),
      el("td", { text: (SOURCE_NAMES[l.source] ?? l.source) + (l.error ? " (after LLM error)" : "") }),
      el("td", { class: "num", text: `${Math.round(l.latency_ms)} ms` })))
      : [el("tr", {}, el("td", { colspan: 5, class: "empty", text: "No questions yet. Ask the assistant something." }))]));
  } catch (err) {
    $("#log-rows").replaceChildren(el("tr", {}, el("td", { colspan: 5, class: "empty", text: `Couldn't load the log: ${err.message}` })));
  }
}

// ---------------------------------------------------------------- boot
(async function boot() {
  try {
    const h = await api("/api/health");
    const engine = $("#engine");
    engine.dataset.mode = h.llm_enabled ? "llm" : "fallback";
    engine.textContent = h.llm_enabled ? "Gemini connected" : "Offline parser (no API key)";
  } catch {
    $("#engine").textContent = "Server unreachable";
  }
  api("/api/stats").then((s) => { baseRate = s.totals.no_show_rate; $("#base-rate").textContent = pct(baseRate); }).catch(() => {});
  loadGlance();
  const initial = tabs.find((t) => t.getAttribute("aria-controls") === `view-${location.hash.slice(1)}`);
  if (initial) selectTab(initial);
})();

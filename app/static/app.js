const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const headers = options.body instanceof FormData ? (options.headers || {}) : { "Content-Type": "application/json", ...(options.headers || {}) };
  const response = await fetch(path, { ...options, headers });
  const body = await response.json();
  if (!response.ok) {
    const detail = body.details?.errors?.[0];
    const suffix = detail ? `（第 ${detail.row || "?"} 行 ${detail.field}: ${detail.reason}）` : "";
    throw new Error(`${body.message || body.detail || `HTTP ${response.status}`}${suffix}`);
  }
  return body;
}

let currentSeries = [];
let currentSymbols = [];

async function guarded(button, action) {
  if (button.disabled) return;
  button.disabled = true;
  try { await action(); } finally { button.disabled = false; }
}

function pct(value) { return `${(value * 100).toFixed(2)}%`; }

function parseSymbols(value) {
  const symbols = value.split(/[,，\s]+/).map(item => item.trim().toUpperCase()).filter(Boolean);
  return symbols.length ? [...new Set(symbols)] : null;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim());
}

function isTableDivider(line) {
  const cells = tableCells(line);
  return cells.length > 0 && cells.every(cell => /^:?-{3,}:?$/.test(cell));
}

function renderMarkdown(markdown) {
  const lines = String(markdown).replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) { index += 1; continue; }

    const heading = line.match(/^(#{1,6})\s+(.+?)(?:\s+#+)?$/);
    if (heading) {
      const level = Math.min(heading[1].length + 2, 6);
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (line.includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      const headers = tableCells(line);
      const rows = [];
      index += 2;
      while (index < lines.length && lines[index].trim().includes("|")) {
        rows.push(tableCells(lines[index]));
        index += 1;
      }
      html.push(`<div class="markdown-table-wrap"><table><thead><tr>${headers.map(cell => `<th>${renderInlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${headers.map((_, cellIndex) => `<td>${renderInlineMarkdown(row[cellIndex] || "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*]\s+/, ""));
        index += 1;
      }
      html.push(`<ul>${items.map(item => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    const paragraph = [line];
    index += 1;
    while (index < lines.length) {
      const next = lines[index].trim();
      if (!next || /^#{1,6}\s+/.test(next) || /^[-*]\s+/.test(next)) break;
      if (next.includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1])) break;
      paragraph.push(next);
      index += 1;
    }
    html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
  }

  return html.join("");
}

function updateMetrics(metrics) {
  const values = [
    pct(metrics.total_return), pct(metrics.annual_return), pct(metrics.annual_volatility),
    metrics.sharpe_ratio.toFixed(2), pct(metrics.max_drawdown), metrics.calmar_ratio.toFixed(2),
    pct(metrics.daily_var_95), pct(metrics.daily_cvar_95), pct(metrics.win_rate),
    pct(metrics.average_daily_turnover),
  ];
  document.querySelectorAll("#metrics strong").forEach((node, index) => node.textContent = values[index]);
}

function drawEquity(series) {
  const canvas = $("#equity-chart");
  const context = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio; canvas.height = height * ratio; context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  if (!series.length) return;
  const values = series.map(item => item.equity);
  const min = Math.min(...values), max = Math.max(...values), spread = Math.max(max - min, .01);
  context.strokeStyle = "#7be0b7"; context.lineWidth = 2; context.beginPath();
  values.forEach((value, index) => {
    const x = 12 + index / Math.max(values.length - 1, 1) * (width - 24);
    const y = height - 18 - (value - min) / spread * (height - 36);
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.stroke();
  $("#equity-caption").innerHTML = `<span>${escapeHtml(series[0].date)} · ${min.toFixed(3)}</span><span>${escapeHtml(series.at(-1).date)} · ${max.toFixed(3)}</span>`;
}

function drawDrawdown(series) {
  const canvas = $("#drawdown-chart");
  const context = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio; canvas.height = height * ratio; context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  if (!series.length) return;
  const values = series.map(item => item.drawdown);
  const min = Math.min(...values), spread = Math.max(Math.abs(min), .01);
  context.strokeStyle = "#ff7c7c"; context.fillStyle = "rgba(255,124,124,.12)"; context.lineWidth = 2; context.beginPath();
  values.forEach((value, index) => {
    const x = 12 + index / Math.max(values.length - 1, 1) * (width - 24);
    const y = 12 + Math.abs(value) / spread * (height - 24);
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.stroke(); context.lineTo(width - 12, 12); context.lineTo(12, 12); context.closePath(); context.fill();
  $("#drawdown-caption").innerHTML = `<span>最大回撤 ${pct(min)}</span><span>0%</span>`;
}

async function refreshHealth() {
  const health = await api("/health");
  const datasetName = health.dataset.name || "UNKNOWN DATASET";
  $("#health").textContent = `${health.status.toUpperCase()} · ${datasetName} · ${health.symbols} ASSETS`;
  $("#health").classList.add("online");
  currentSymbols = await api("/api/market/symbols");
  const quickSymbol = currentSymbols[0];
  $("#price-quick").disabled = !quickSymbol;
  $("#price-quick").dataset.query = quickSymbol ? `查看 ${quickSymbol} 最近30日行情` : "";
  return health;
}

async function loadDatasets() {
  const [datasets, health] = await Promise.all([api("/api/datasets"), api("/health")]);
  const current = health.dataset;
  const quality = current.quality ? ` · 缺失交易日股票 ${Object.keys(current.quality.missing_trading_days_by_symbol || {}).length} 只` : "";
  $("#dataset-status").innerHTML = `<strong>${escapeHtml(current.name)}</strong><div>${escapeHtml(`${current.market || "UNKNOWN"} · ${current.is_demo ? "演示数据" : `${current.start_date} 至 ${current.end_date}`} · ${health.symbols} 只资产${quality}`)}</div>`;
  $("#dataset-list").innerHTML = datasets.map(item => `<div class="dataset-item"><div><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(`#${item.id} · ${item.market} · ${item.adjustment} · ${item.start_date} 至 ${item.end_date} · ${item.symbol_count} 只 · ${item.row_count} 行`)}</small></div>${item.active ? '<span class="active-badge">当前数据集</span>' : `<button data-activate-dataset="${item.id}">启用</button>`}</div>`).join("");
  document.querySelectorAll("[data-activate-dataset]").forEach(button => button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api(`/api/datasets/${button.dataset.activateDataset}/activate`, { method: "POST" });
      await Promise.all([loadDatasets(), refreshHealth()]);
      $("#factor-symbols").value = ""; $("#backtest-symbols").value = "";
    } catch (error) { $("#dataset-status").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`; }
  }));
}

async function importDataset() {
  const file = $("#dataset-file").files[0];
  if (!file) { $("#dataset-status").innerHTML = '<span class="error">请先选择 CSV 文件。</span>'; return; }
  const form = new FormData();
  form.append("file", file);
  form.append("name", $("#dataset-name").value);
  form.append("market_name", $("#dataset-market").value);
  form.append("adjustment", $("#dataset-adjustment").value);
  form.append("source", $("#dataset-source").value);
  form.append("activate", "true");
  $("#dataset-status").textContent = "正在校验并导入行情数据...";
  try {
    const dataset = await api("/api/datasets/import", { method: "POST", body: form });
    const missing = Object.keys(dataset.quality.missing_trading_days_by_symbol || {}).length;
    $("#dataset-status").innerHTML = `<strong>导入成功：${escapeHtml(dataset.name)}</strong><div>${escapeHtml(`${dataset.start_date} 至 ${dataset.end_date} · ${dataset.symbol_count} 只 · ${dataset.row_count} 行 · 重复 0 行 · 缺失交易日股票 ${missing} 只`)}</div>`;
    await Promise.all([loadDatasets(), refreshHealth()]);
  } catch (error) { $("#dataset-status").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`; }
}

async function runFactor() {
  const node = $("#factor-result"); node.textContent = "计算中...";
  const payload = {
    factor: $("#factor-analysis").value,
    lookback: +$("#factor-lookback").value,
    symbols: parseSymbols($("#factor-symbols").value),
    start_date: $("#factor-start").value || null,
    end_date: $("#factor-end").value || null,
  };
  try {
    const result = await api("/api/factors/analyze", { method: "POST", body: JSON.stringify(payload) });
    const header = "| 排名 | 股票 | 日期 | 原始值 | 排序分数 | 方向 |\n|---:|---|---|---:|---:|---|\n";
    const rows = result.ranking.map(item => `| ${item.rank} | ${item.symbol} | ${item.date} | ${item.raw_value} | ${item.score} | ${item.direction} |`).join("\n");
    node.innerHTML = `<div class="markdown-body">${renderMarkdown(`### ${result.factor} 因子排名\n截至 ${result.as_of_date} · 有效 ${result.effective_symbol_count}/${result.requested_symbol_count}（覆盖率 ${pct(result.coverage_ratio)}）\n${header}${rows}`)}</div>`;
  } catch (error) { node.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`; }
}

async function runBacktest() {
  const payload = { factor: $("#factor").value, lookback: +$("#lookback").value, top_k: +$("#top-k").value, rebalance_days: +$("#rebalance").value, transaction_cost_bps: +$("#cost").value, symbols: parseSymbols($("#backtest-symbols").value), start_date: $("#backtest-start").value || null, end_date: $("#backtest-end").value || null };
  $("#backtest-detail").textContent = "计算中...";
  try {
    const result = await api("/api/backtests", { method: "POST", body: JSON.stringify(payload) });
    renderBacktest(result);
    await loadBacktestHistory();
  } catch (error) { $("#backtest-detail").textContent = error.message; }
}

function renderBacktest(result) {
  currentSeries = result.series || [];
  updateMetrics(result.metrics); drawEquity(currentSeries); drawDrawdown(currentSeries);
  const latest = currentSeries.at(-1) || { weights: {}, turnover: 0 };
  const holdings = Object.entries(latest.weights || {}).map(([symbol, weight]) => `${symbol} ${pct(weight)}`).join("、") || "空仓";
  const p = result.provenance || {};
  $("#backtest-detail").innerHTML = `<strong>回测 #${result.id} · ${escapeHtml(p.dataset_name || "未知数据集")}</strong>
    <div>${escapeHtml(`${p.start_date || "-"} 至 ${p.end_date || "-"} · 首次持仓 ${result.methodology?.first_holding_date || "-"} · 有效 ${result.methodology?.effective_trading_days || currentSeries.length} 日`)}</div>
    <div>当前持仓：${escapeHtml(holdings)}；最近换手：${pct(latest.turnover || 0)}</div>
    <div>口径：${escapeHtml(result.methodology?.weighting || "")}；${escapeHtml(result.methodology?.cost_model || "")}</div>`;
}

async function loadBacktestHistory() {
  const history = await api("/api/backtests?limit=5");
  $("#backtest-history").innerHTML = history.map(item => `<button class="history-item" data-backtest-id="${item.id}">#${item.id} · ${escapeHtml(item.strategy)} · ${pct(item.metrics.total_return)} · ${escapeHtml(item.created_at)}</button>`).join("");
  document.querySelectorAll("[data-backtest-id]").forEach(button => button.addEventListener("click", () => guarded(button, async () => {
    renderBacktest(await api(`/api/backtests/${button.dataset.backtestId}`));
  })));
  return history;
}

async function runAgent() {
  const node = $("#agent-result"); node.textContent = "正在解析需求并调用量化工具...";
  try {
    const result = await api("/api/agent/run", { method: "POST", body: JSON.stringify({ query: $("#agent-query").value }) });
    const trace = `任务 #${result.task_id} · ${result.plan.map(item => item.tool).join(" → ")}`;
    node.innerHTML = `<div class="markdown-body">${renderMarkdown(result.answer)}</div><div class="citation">${escapeHtml(trace)}</div>`;
    const backtest = result.steps.find(step => step.tool === "run_backtest");
    if (backtest) { updateMetrics(backtest.output.metrics); drawEquity(backtest.output.series); drawDrawdown(backtest.output.series); }
  } catch (error) { node.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`; }
}

$("#import-dataset").addEventListener("click", event => guarded(event.currentTarget, importDataset));
$("#run-factor").addEventListener("click", event => guarded(event.currentTarget, runFactor));
$("#run-backtest").addEventListener("click", event => guarded(event.currentTarget, runBacktest));
$("#run-agent").addEventListener("click", event => guarded(event.currentTarget, runAgent));
document.querySelectorAll("[data-query]").forEach(button => button.addEventListener("click", () => { $("#agent-query").value = button.dataset.query; guarded($("#run-agent"), runAgent); }));
$("#price-quick").addEventListener("click", event => {
  $("#agent-query").value = event.currentTarget.dataset.query;
  guarded($("#run-agent"), runAgent);
});

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { drawEquity(currentSeries); drawDrawdown(currentSeries); }, 120);
});

(async () => {
  try {
    await refreshHealth();
    await loadDatasets();
    const history = await loadBacktestHistory();
    if (history.length) renderBacktest(await api(`/api/backtests/${history[0].id}`));
  } catch (error) { $("#health").textContent = `连接失败：${error.message}`; }
})();

const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

function pct(value) { return `${(value * 100).toFixed(2)}%`; }

function updateMetrics(metrics) {
  const values = [pct(metrics.total_return), pct(metrics.annual_return), metrics.sharpe_ratio.toFixed(2), pct(metrics.max_drawdown)];
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
    const x = 12 + index / (values.length - 1) * (width - 24);
    const y = height - 18 - (value - min) / spread * (height - 36);
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.stroke();
}

async function runBacktest() {
  const payload = { factor: $("#factor").value, lookback: +$("#lookback").value, top_k: +$("#top-k").value, rebalance_days: +$("#rebalance").value, transaction_cost_bps: +$("#cost").value };
  $("#backtest-detail").textContent = "计算中...";
  try {
    const result = await api("/api/backtests", { method: "POST", body: JSON.stringify(payload) });
    updateMetrics(result.metrics); drawEquity(result.series);
    $("#backtest-detail").textContent = JSON.stringify({ id: result.id, strategy: result.strategy, parameters: result.parameters, metrics: result.metrics, methodology: result.methodology }, null, 2);
  } catch (error) { $("#backtest-detail").textContent = error.message; }
}

async function runAgent() {
  const node = $("#agent-result"); node.textContent = "Agent 正在拆解任务并调用工具...";
  try {
    const result = await api("/api/agent/run", { method: "POST", body: JSON.stringify({ query: $("#agent-query").value }) });
    node.innerHTML = `<strong>${result.answer}</strong><div class="citation">任务 #${result.task_id} · ${result.plan.map(item => item.tool).join(" → ")}</div>`;
    const backtest = result.steps.find(step => step.tool === "run_backtest");
    if (backtest) { updateMetrics(backtest.output.metrics); drawEquity(backtest.output.series); }
  } catch (error) { node.innerHTML = `<span class="error">${error.message}</span>`; }
}

async function searchRag() {
  const node = $("#rag-result"); node.textContent = "检索中...";
  try {
    const result = await api("/api/rag/search", { method: "POST", body: JSON.stringify({ query: $("#rag-query").value, top_k: 4 }) });
    node.innerHTML = `<strong>${result.answer}</strong>${result.citations.map(item => `<div class="citation">[${item.rank}] ${item.document} · chunk ${item.chunk_index} · score ${item.score}</div>`).join("")}`;
  } catch (error) { node.innerHTML = `<span class="error">${error.message}</span>`; }
}

async function loadDocuments() {
  const documents = await api("/api/documents");
  $("#documents").innerHTML = documents.map(item => `<span>${item.title} · ${item.chunks} chunks</span>`).join("");
}

$("#document-file").addEventListener("change", async (event) => {
  const file = event.target.files[0]; if (!file) return;
  const content = await file.text();
  await api("/api/documents", { method: "POST", body: JSON.stringify({ title: file.name, source: "upload", content }) });
  await loadDocuments();
});
$("#upload-document").addEventListener("click", () => $("#document-file").click());
$("#run-backtest").addEventListener("click", runBacktest);
$("#run-agent").addEventListener("click", runAgent);
$("#search-rag").addEventListener("click", searchRag);
document.querySelectorAll("[data-query]").forEach(button => button.addEventListener("click", () => { $("#agent-query").value = button.dataset.query; runAgent(); }));

(async () => {
  try {
    const health = await api("/health");
    $("#health").textContent = `${health.status.toUpperCase()} · ${health.vector_backend.toUpperCase()} · ${health.symbols} ASSETS`;
    $("#health").classList.add("online");
    await loadDocuments(); await runBacktest();
  } catch (error) { $("#health").textContent = `连接失败：${error.message}`; }
})();

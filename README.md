# QuantAgent 量化研究智能体

QuantAgent 是一个可以离线运行的量化研究工作台。它把自然语言请求拆分为行情查询、因子计算、策略回测、风险评估和知识库检索工具，并将研究参数、任务轨迹和回测结果保存到 SQLite。金融数值由 Pandas/NumPy 确定性计算，Agent 只负责规划与工具路由。

## 功能

- 多资产 OHLCV 行情查询，内置固定随机种子的可复现实验数据；
- 动量、短期反转、低波动、均线偏离和成交量 Z-Score 因子；
- 横截面 Top-K 回测，信号滞后一日、固定周期调仓并按换手率扣除交易成本；
- 累计/年化收益、波动率、夏普、最大回撤、Calmar、VaR、CVaR 和胜率；
- 文档切分、Embedding、FAISS 检索与来源引用；
- Tool Calling 风格的 Agent 规划，任务及结果写入 SQLite；
- FastAPI/Swagger 接口和原生 Web 研究面板；
- 单元、RAG、API 与端到端工作流测试。

## 快速启动

需要 Python 3.11 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --reload
```

也可以直接运行（Windows 推荐使用批处理文件，不受 PowerShell 执行策略影响）：

```powershell
.\run.bat
```

或运行 PowerShell 脚本：

```powershell
.\run.ps1
```

如果系统拦截 PowerShell 脚本，可以仅为当前进程临时放行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

打开：

- 研究面板：<http://127.0.0.1:8000>
- Swagger：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query":"回测20日动量因子并给出夏普比率和最大回撤"}'
```

```bash
curl -X POST http://127.0.0.1:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{"title":"研究笔记","source":"manual","content":"动量因子使用过去收益排序。"}'
```

## 测试

```powershell
.\.venv\Scripts\python -m pytest
```

测试覆盖：数据可复现、因子排名、交易成本、未来数据隔离、风险指标、RAG 引用、API 参数校验、回测持久化和 Agent 完整流程。

## 目录

```text
app/
  main.py              FastAPI 路由与依赖装配
  agent.py             任务规划、参数提取与工具调用
  tools.py             工具注册表及 JSON Schema
  db.py                SQLite 事务与任务/回测持久化
  quant/                行情、因子、回测和风险指标
  rag/                  文档切分、Embedding 与 FAISS 索引
  static/               单页研究面板
tests/                  单元与端到端测试
docs/                   架构和面试追问说明
```

详细设计参见 [Architecture](docs/ARCHITECTURE.md)，项目追问与局限参见 [Interview Guide](docs/INTERVIEW_GUIDE.md)。

## 数据声明

仓库默认数据为固定随机种子生成的模拟行情，仅用于软件功能、因子时序和回测流程演示，不代表真实证券，也不构成投资建议。可以使用 `scripts/export_demo_data.py` 导出 CSV，并通过 `MarketDataService.replace_from_csv()` 接入具有相同字段的真实数据。

## License

MIT

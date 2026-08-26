# QuantAgent 量化研究智能体

QuantAgent 是一个可复现的量化研究工作台，支持通过自然语言理解研究需求并调用受控工具。行情、因子、回测与风险数值由 Pandas/NumPy 确定性计算；真实行情标准化为 Parquet，数据版本、研究参数、工具轨迹和回测结果保存到 SQLite。

## 功能

- Web 上传真实 OHLCV CSV，执行字段、重复行、缺失值和价格关系校验；
- 原始文件与标准化 Parquet 分离保存，以 SHA-256 标识不可变数据版本；
- 数据集切换、重启恢复、动态股票代码和自定义股票池；
- 多资产 OHLCV 行情查询，并保留固定随机种子的可复现实验数据；
- 动量、短期反转、低波动、均线偏离和成交量 Z-Score 因子；
- 横截面 Top-K 回测，信号滞后一日、固定周期调仓并按换手率扣除交易成本；
- 累计/年化收益、波动率、夏普、最大回撤、Calmar、VaR、CVaR 和胜率；
- 可选的自然语言参数提取、多轮工具调用与研究结果解释；
- 严格 JSON Schema、Pydantic 二次校验、工具白名单和最大调用轮数；
- FastAPI/Swagger 接口和原生 Web 研究面板；
- 单元、数据导入、API 与端到端工作流测试。

## 快速启动

需要 Python 3.11 或更高版本。

项目未配置模型密钥时，行情导入、因子分析、策略回测和风险概览仍可直接使用。如需启用自然语言研究任务，复制环境变量示例并填写自己的 API Key：

```powershell
Copy-Item .env.example .env
```

```env
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MAX_TOOL_ROUNDS=6
QUANTAGENT_DATASET_PATH=data/datasets
```

`.env` 已加入 `.gitignore`，不要将真实密钥提交到 GitHub。

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

## 导入真实行情

在研究面板的“行情数据”区域上传 UTF-8 CSV。文件采用长表结构，同一行表示一只股票在一个交易日的行情：

```csv
date,symbol,open,high,low,close,volume
2024-01-02,000001.SZ,9.28,9.42,9.21,9.36,85213600
2024-01-03,000001.SZ,9.35,9.51,9.30,9.47,76325100
2024-01-02,600519.SH,1680.00,1695.00,1672.10,1688.50,2813500
```

导入成功后，该数据集会自动成为当前研究数据。页面、因子接口、回测引擎和 Agent 会动态使用 CSV 中的股票代码。原始文件和 Parquet 位于 `data/datasets/`，该目录已被 Git 忽略；数据集元数据和当前选择保存在 SQLite。

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/api/datasets/import \
  -F "file=@market.csv" -F "name=A股日线" \
  -F "market_name=CN" -F "adjustment=qfq"
```

```bash
curl -X POST http://127.0.0.1:8000/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query":"回测20日动量因子并给出夏普比率和最大回撤"}'
```

## 测试

```powershell
.\.venv\Scripts\python -m pytest
```

测试覆盖：数据可复现、CSV 校验、Parquet 持久化、重启恢复、动态股票池、因子排名、交易成本、未来数据隔离、风险指标、API 参数校验、回测数据血缘，以及使用模拟 DeepSeek Responses 客户端验证的多轮 Function Calling。测试不会消耗真实 API 额度。

## 目录

```text
app/
  main.py              FastAPI 路由与依赖装配
  datasets.py          CSV 导入、数据质量、Parquet 与数据集切换
  agent.py             DeepSeek Responses API 多轮工具调用编排
  tools.py             工具白名单、严格 JSON Schema 与参数校验
  db.py                SQLite 事务与任务/回测持久化
  quant/                行情、因子、回测和风险指标
  static/               单页研究面板
tests/                  单元与端到端测试
docs/                   架构和面试追问说明
```

产品需求参见 [Requirements](docs/REQUIREMENTS.md)，详细设计参见 [Architecture](docs/ARCHITECTURE.md)，项目追问与局限参见 [Interview Guide](docs/INTERVIEW_GUIDE.md)。

## 数据声明

未导入数据时，仓库使用固定随机种子生成的模拟行情，仅用于软件功能、因子时序和回测流程演示，不代表真实证券，也不构成投资建议。真实行情必须具有一致的复权口径；每次回测都会记录数据集 ID、内容哈希、实际股票、日期范围和程序版本。DeepSeek Agent 仅能调用预定义研究工具，不能执行任意代码或交易指令。

## License

MIT

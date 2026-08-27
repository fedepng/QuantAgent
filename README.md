# QuantAgent 量化研究智能体

QuantAgent 是一个可复现的量化研究工作台，支持通过自然语言理解研究需求并调用受控工具。行情、因子、回测与风险数值由 Pandas/NumPy 确定性计算；真实行情标准化为 Parquet，数据版本、研究参数、工具轨迹和回测结果保存到 SQLite。

## 已实现

- Web 上传 UTF-8/UTF-8 BOM OHLCV CSV，执行自然日、股票代码、有限数、重复行和价格关系校验，并返回结构化错误；
- 原始文件与标准化 Parquet 分离保存，分别记录原始字节和标准化数据 SHA-256；
- 数据集切换、重启恢复、动态股票代码和自定义股票池；
- 多资产 OHLCV 行情查询，并保留固定随机种子的可复现实验数据；
- 动量、反转、低波动、均线偏离和成交量 Z-Score 因子，以及固定覆盖率的同日横截面排名；
- 横截面 Top-K 回测，使用起点前数据预热、信号滞后一日、漂移权重和显式建仓成本；
- 累计/年化收益、波动率、夏普、最大回撤、Calmar、VaR、CVaR 和胜率；
- 可选的自然语言参数提取、多轮工具调用与研究结果解释，逐次记录工具耗时、状态、摘要和错误；
- 严格 JSON Schema、Pydantic 二次校验、工具白名单和最大调用轮数；
- FastAPI/Swagger 接口和原生 Web 研究面板；
- 统一 API 错误码、请求 ID、数据损坏降级状态和可迁移的相对存储路径；
- 单元、数据导入、API、Agent 与端到端回归测试。

## 有限实现

- 回测是日频 close-to-close 研究模型，使用线性成本，期末不假设强制平仓；
- 持仓资产缺少收益时回测会明确失败，尚未模拟停牌撮合、涨跌停、滑点与冲击成本；
- “真实数据”指用户上传的 CSV，系统不会替用户验证来源、复权质量或投资适用性；
- SQLite 和原生 Web 面向本地单用户研究，不是生产级交易系统。

## 未来候选

外部行情源、命名股票池、增量导入、数据删除/导出、基准对比、参数寻优、
机器学习、报告导出、实盘交易和多用户云部署均未实现。

## 快速启动

环境要求：Windows、Git、Python 3.11 或更高版本。

### 第一次运行

打开 PowerShell，下载项目：

```powershell
cd $HOME\Desktop
git clone https://github.com/fedepng/QuantAgent.git
cd QuantAgent
```

创建配置文件并打开：

```powershell
Copy-Item .env.example .env
notepad .env
```

填写自己的 API Key：

```env
LLM_API_KEY=你的API密钥
```

保存 `.env` 后启动：

```powershell
.\run.bat
```

`run.bat` 会在首次运行时自动创建 `.venv`、安装依赖并启动服务。浏览器访问 <http://127.0.0.1:8000>。

### 手动启动

不使用启动脚本时，依次执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

### 后续启动

以后进入项目目录运行：

```powershell
cd $HOME\Desktop\QuantAgent
.\run.bat
```

常用地址：

- 研究面板：<http://127.0.0.1:8000>
- Swagger：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

## 模型 API 配置

行情导入、因子分析、策略回测和风险概览不依赖模型。自然语言研究功能需要在 `.env` 中配置模型 API；`.env` 已加入 `.gitignore`，不要将真实密钥提交到 GitHub。

项目提供两种工具调用协议：

- `responses`：适用于支持 Responses API 的平台；
- `chat_completions`：适用于支持 Chat Completions Function Calling 的兼容平台。

默认示例：

```env
LLM_PROVIDER=deepseek
LLM_PROTOCOL=responses
LLM_API_KEY=sk-your-api-key
LLM_MODEL=deepseek-v4-flash
LLM_BASE_URL=https://api.deepseek.com
LLM_MAX_TOOL_ROUNDS=6
```

OpenAI Responses API 示例：

```env
LLM_PROVIDER=openai
LLM_PROTOCOL=responses
LLM_API_KEY=sk-your-openai-api-key
LLM_MODEL=your-openai-model
LLM_BASE_URL=https://api.openai.com/v1
```

OpenAI 兼容 Chat Completions 平台示例：

```env
LLM_PROVIDER=custom
LLM_PROTOCOL=chat_completions
LLM_API_KEY=your-provider-api-key
LLM_MODEL=your-model-name
LLM_BASE_URL=https://your-provider.example/v1
```

切换平台时必须确认目标接口支持 Function Calling。旧版 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MAX_TOOL_ROUNDS` 仍可读取，但新配置应使用 `LLM_*`。

## 导入真实行情

在研究面板的“行情数据”区域上传 UTF-8 或 UTF-8 BOM CSV。文件采用长表结构，同一行表示一只股票在一个交易日的行情：

```csv
date,symbol,open,high,low,close,volume
2024-01-02,000001.SZ,9.28,9.42,9.21,9.36,85213600
2024-01-03,000001.SZ,9.35,9.51,9.30,9.47,76325100
2024-01-02,600519.SH,1680.00,1695.00,1672.10,1688.50,2813500
```

列名会 trim 后转小写，股票代码会 trim 后转大写，日期归一为无时区自然日；
OHLCV 必须为有限数，OHLC 大于 0，volume 允许非负小数。导入成功后，该数据集
会自动成为当前研究数据。原始文件和 Parquet 位于 `data/datasets/`，数据集元数据
和当前选择保存在 SQLite；启用和重启恢复都会校验标准化数据哈希。

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

测试覆盖：首日回撤与风险边界、手算权重漂移和建仓成本、预热期、同日因子截面、
CSV 时间重复/BOM/非有限数、Parquet 迁移与损坏恢复、统一错误契约、Agent 成功和
失败轨迹，以及两种模型协议。测试不会消耗真实 API 额度。

## 目录

```text
app/
  main.py              FastAPI 路由与依赖装配
  datasets.py          CSV 导入、数据质量、Parquet 与数据集切换
  agent.py             多模型协议适配与多轮工具调用编排
  tools.py             工具白名单、严格 JSON Schema 与参数校验
  db.py                SQLite 事务与任务/回测持久化
  quant/                行情、因子、回测和风险指标
  static/               单页研究面板
tests/                  单元与端到端测试
docs/                   架构和面试追问说明
```

当前能力边界参见 [Requirements](docs/REQUIREMENTS.md)，计算公式参见
[Methodology](docs/METHODOLOGY.md)，详细设计参见
[Architecture](docs/ARCHITECTURE.md)，项目追问与局限参见
[Interview Guide](docs/INTERVIEW_GUIDE.md)。

## 数据声明

未导入数据时，仓库使用固定随机种子生成的模拟行情，仅用于软件功能、因子时序和回测流程演示，不代表真实证券，也不构成投资建议。真实行情必须具有一致的复权口径；每次回测都会记录数据集 ID、原始/标准化哈希、实际股票、日期范围、Git commit 和方法版本。模型仅能调用预定义研究工具，不能执行任意代码或交易指令。

## License

MIT

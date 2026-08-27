# Architecture

## Request flow

```text
Browser / API client
        |
        v
FastAPI validation layer
        |
        +------> DatasetService --> Raw CSV + Parquet
        |               |
        |               +--------> SQLite dataset metadata
        |
        v
LLM protocol adapter --> SQLite task audit
        |
        v
ToolRegistry
  |          |            |
Market    Factor      Backtest/Risk
data      engine       engine
  |          |            |
  +----------+------------+
             |
     SQLite + Parquet
```

The configured model interprets natural language and emits custom function calls. `ResponsesAdapter` handles Responses API items, while `ChatCompletionsAdapter` handles assistant `tool_calls` and tool messages. Both normalize calls before Pydantic validation and dispatch through the same allow-listed `ToolRegistry`. The model never calculates financial numbers from prose: market data, factors, backtest series, and risk metrics come from deterministic Pandas/NumPy code.

## Agent loop

1. Send the user request, instructions, and function schemas through the configured protocol adapter.
2. Validate each returned function name and JSON argument object.
3. Create an independent tool-call record, execute the local tool, then save duration, status, compact summary, error and optional backtest id.
4. Return a compact tool result using the protocol's required message format and original call id.
5. Repeat until the model returns final text or the configured tool-round limit is reached.

Provider, protocol, model, base URL and credential are read from `LLM_*` environment variables. Legacy `DEEPSEEK_*` variables remain a fallback for existing installations. Credentials are never stored in SQLite, and the current dataset and dynamic symbol list are included in the research context.

## Dataset lifecycle

1. Receive a CSV upload with a 50 MB limit.
2. Validate required columns, numeric values, uniqueness, missing values, volume and OHLC relationships.
3. Calculate a SHA-256 hash and reject duplicate files.
4. Save the immutable original CSV and normalized Parquet under a generated directory.
5. Commit dataset metadata and quality statistics to SQLite.
6. Activate the dataset and replace the in-memory market frame.
7. Restore the active dataset from SQLite and Parquet after restart.

The database stores a relative Parquet key. Activation and restart recovery verify readability and the normalized-data hash; failures leave the service explicitly degraded. Backtest provenance includes raw and normalized hashes, dataset metadata, actual symbols, effective range, Git commit, and factor/risk/backtest methodology versions. API responses never expose the internal storage path.

## Backtest timing

1. Read pre-start observations to warm up the requested factor window.
2. Shift the complete signal matrix by one research day.
3. Start the rebalance clock on the first day that can form a Top-K portfolio.
4. Before later rebalances, drift prior weights by realized asset returns.
5. Compare the target weights with drifted weights and deduct `turnover * transaction_cost_bps / 10000`.
6. Fail explicitly if a held asset lacks its close-to-close return.

Initial entry cost is charged once and final liquidation is not assumed. This ordering prevents the strategy from using the current closing price before earning the current close-to-close return.

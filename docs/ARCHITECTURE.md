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
3. Execute the corresponding local tool and persist its full result.
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

Backtest provenance includes the dataset id and hash, actual symbol list, effective date range and application version. API responses never expose the internal storage path.

## Backtest timing

1. Calculate factors from close prices through day `t`.
2. Shift the complete signal matrix by one trading day.
3. On a rebalance day, select the top `k` symbols from the lagged signal.
4. Apply those weights to the current day's asset returns.
5. Deduct `turnover * transaction_cost_bps / 10000`.

This ordering prevents the strategy from using the current closing price before earning the current close-to-close return.

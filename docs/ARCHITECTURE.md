# Architecture

## Request flow

```text
Browser / API client
        |
        v
FastAPI validation layer
        |
        v
ResearchAgent planner -----> SQLite task audit
        |
        v
ToolRegistry
  |          |            |              |
Market    Factor      Backtest/Risk     RAG
data      engine       engine           service
  |          |            |              |
  +----------+------------+-------+------+
                                 |
                         SQLite + FAISS
```

The agent never calculates financial numbers from prose. It creates a typed tool plan and delegates every numeric result to deterministic Pandas/NumPy code. Tool inputs and results are stored with the task so that a run can be inspected later.

## Backtest timing

1. Calculate factors from close prices through day `t`.
2. Shift the complete signal matrix by one trading day.
3. On a rebalance day, select the top `k` symbols from the lagged signal.
4. Apply those weights to the current day's asset returns.
5. Deduct `turnover * transaction_cost_bps / 10000`.

This ordering prevents the strategy from using the current closing price before earning the current close-to-close return.

## RAG pipeline

1. Normalize and split documents into overlapping chunks.
2. Encode English tokens and Chinese character n-grams into normalized vectors.
3. Use FAISS `IndexFlatIP` for cosine-equivalent retrieval. If FAISS is unavailable, use a NumPy matrix product with identical scoring semantics.
4. Return document id, chunk id, rank, similarity score, and excerpt for every citation.

The default hashing embedding is intentionally local and deterministic. A neural embedding provider can replace it behind the same `encode()` interface.


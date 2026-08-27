from __future__ import annotations

from typing import Any


class QuantAgentError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 422,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


class DataValidationError(QuantAgentError):
    def __init__(self, errors: list[dict[str, Any]], total_errors: int | None = None) -> None:
        super().__init__(
            "INVALID_MARKET_DATA",
            "CSV data failed validation",
            details={
                "errors": errors[:20],
                "total_errors": total_errors if total_errors is not None else len(errors),
                "returned_errors": min(len(errors), 20),
            },
        )

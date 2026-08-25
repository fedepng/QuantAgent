from pathlib import Path

from app.db import Database
from app.rag.service import RagService, split_text


def test_split_text_uses_overlap_without_empty_chunks() -> None:
    chunks = split_text("动量策略需要避免未来函数。" * 80, chunk_size=120, overlap=20)
    assert len(chunks) > 2
    assert all(chunks)


def test_rag_returns_source_citation(tmp_path: Path) -> None:
    database = Database(tmp_path / "rag.db")
    database.initialize()
    rag = RagService(database, dimension=128)
    document = rag.add_document(
        "回测规范",
        "为了避免未来函数，因子必须滞后一个交易日再形成持仓。交易成本按换手率扣除。",
        "test",
    )
    result = rag.search("回测如何避免未来函数", top_k=2)
    assert result["citations"]
    assert result["citations"][0]["document_id"] == document["id"]
    assert "未来函数" in result["citations"][0]["excerpt"]


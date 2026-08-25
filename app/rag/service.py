from __future__ import annotations

from dataclasses import dataclass

from app.db import Database, utc_now

from .embeddings import HashingEmbedding
from .index import VectorIndex


@dataclass(frozen=True)
class Chunk:
    id: int
    document_id: int
    document_title: str
    chunk_index: int
    content: str


def split_text(text: str, chunk_size: int = 520, overlap: int = 80) -> list[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            break_at = max(normalized.rfind("。", start, end), normalized.rfind("\n", start, end))
            if break_at > start + chunk_size // 2:
                end = break_at + 1
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


class RagService:
    def __init__(self, database: Database, dimension: int = 384) -> None:
        self.database = database
        self.embedding = HashingEmbedding(dimension)
        self.index = VectorIndex(dimension)
        self._chunks: list[Chunk] = []
        self.rebuild()

    def rebuild(self) -> None:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.document_id, d.title AS document_title,
                       c.chunk_index, c.content
                FROM chunks c JOIN documents d ON d.id = c.document_id
                ORDER BY c.id
                """
            ).fetchall()
        self._chunks = [Chunk(**dict(row)) for row in rows]
        matrix = self.embedding.encode([chunk.content for chunk in self._chunks])
        self.index.rebuild(matrix)

    def add_document(self, title: str, content: str, source: str = "manual") -> dict[str, object]:
        title = title.strip()
        if not title:
            raise ValueError("document title cannot be empty")
        chunks = split_text(content)
        if not chunks:
            raise ValueError("document content cannot be empty")
        with self.database.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO documents(title, source, content, created_at) VALUES (?, ?, ?, ?)",
                (title, source, content, utc_now()),
            )
            document_id = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO chunks(document_id, chunk_index, content) VALUES (?, ?, ?)",
                [(document_id, index, chunk) for index, chunk in enumerate(chunks)],
            )
        self.rebuild()
        return {"id": document_id, "title": title, "source": source, "chunks": len(chunks)}

    def list_documents(self) -> list[dict[str, object]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT d.id, d.title, d.source, d.created_at, COUNT(c.id) AS chunks
                FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id ORDER BY d.id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, document_id: int) -> bool:
        with self.database.connect() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE id=?", (document_id,))
        self.rebuild()
        return cursor.rowcount > 0

    def search(self, query: str, top_k: int = 4) -> dict[str, object]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        vector = self.embedding.encode([query])[0]
        matches = self.index.search(vector, top_k)
        citations = []
        for rank, (index, score) in enumerate(matches, 1):
            chunk = self._chunks[index]
            citations.append(
                {
                    "rank": rank,
                    "document_id": chunk.document_id,
                    "document": chunk.document_title,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "score": round(score, 6),
                    "excerpt": chunk.content[:280],
                }
            )
        if citations:
            answer = "根据知识库检索结果：" + "；".join(
                f"[{item['rank']}] {item['excerpt']}" for item in citations[:2]
            )
        else:
            answer = "知识库暂时为空，请先上传研报或研究笔记。"
        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "index_backend": self.index.backend,
        }


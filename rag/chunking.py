"""법령 코퍼스를 청크로 자른다.

두 가지 전략을 비교한다.

- ``fixed``   : 조문 구조를 무시한 고정 길이 슬라이딩 윈도우 (Naive RAG의 통상적 방식)
- ``article`` : 항(項) 단위 — 단서 조항은 본문과 같은 청크에 유지

고정 길이 청킹이 단서 조항을 본문에서 떼어놓는 현상을 수치로 드러내는 것이
이 모듈의 핵심 목적이다. ``units_full`` / ``units_partial`` 구분이 그 측정에 쓰인다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PROVISO_SUFFIX = "-proviso"


@dataclass
class Unit:
    """조문의 최소 단위 (항·호·단서)."""

    unit_id: str
    doc_id: str
    label: str
    text: str
    # 문서 전체 텍스트 안에서 이 단위가 차지하는 [start, end) 구간
    start: int = 0
    end: int = 0

    @property
    def is_proviso(self) -> bool:
        return self.unit_id.endswith(PROVISO_SUFFIX)

    @property
    def parent_id(self) -> str:
        """단서라면 본문 단위의 id, 아니면 자기 자신."""
        if self.is_proviso:
            return self.unit_id[: -len(PROVISO_SUFFIX)]
        return self.unit_id

    @property
    def rendered(self) -> str:
        return f"{self.label} {self.text}".strip() if self.label else self.text


@dataclass
class Chunk:
    chunk_id: str
    text: str
    # 이 청크 안에 '온전히' 담긴 단위들
    units_full: list[str] = field(default_factory=list)
    # 경계에 걸려 '일부만' 담긴 단위들
    units_partial: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)


def doc_header(doc: dict) -> str:
    """검색 단서가 되도록 각 문서 앞에 붙이는 머리말."""
    law = doc.get("law", "")
    article = doc.get("article", "")
    title = doc.get("title", "")
    head = " ".join(x for x in (law, article) if x)
    if title:
        head = f"{head}({title})" if head else title
    return f"[{head}]" if head else ""


def split_units(doc: dict) -> list[Unit]:
    """문서를 단위 리스트로 분해하고, 각 단위의 문자 구간을 계산한다.

    ``units`` 키가 없는 문서(분쟁해결기준 등 조문 구조가 없는 자료)는
    문서 전체가 하나의 단위가 되며 ``unit_id == doc_id`` 가 된다.
    """
    doc_id = doc["doc_id"]
    header = doc_header(doc)
    cursor = len(header) + 1 if header else 0

    raw_units = doc.get("units")
    if not raw_units:
        text = doc.get("text", "")
        return [
            Unit(
                unit_id=doc_id,
                doc_id=doc_id,
                label="",
                text=text,
                start=cursor,
                end=cursor + len(text),
            )
        ]

    units: list[Unit] = []
    for raw in raw_units:
        u = Unit(
            unit_id=raw["unit_id"],
            doc_id=doc_id,
            label=raw.get("label", ""),
            text=raw["text"],
        )
        body = u.rendered
        u.start = cursor
        u.end = cursor + len(body)
        cursor = u.end + 1  # 단위 사이의 개행 1칸
        units.append(u)
    return units


def doc_text(doc: dict, units: list[Unit] | None = None) -> str:
    """split_units가 계산한 구간과 정확히 일치하는 문서 전체 텍스트."""
    units = units if units is not None else split_units(doc)
    header = doc_header(doc)
    body = "\n".join(u.rendered for u in units)
    return f"{header}\n{body}" if header else body


def corpus_unit_ids(corpus: dict) -> set[str]:
    """코퍼스에 존재하는 모든 unit_id (gold_citations 오타 검증용)."""
    ids: set[str] = set()
    for doc in corpus["documents"]:
        for u in split_units(doc):
            ids.add(u.unit_id)
    return ids


def _fixed_chunks(corpus: dict, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")
    step = max(1, chunk_size - max(0, overlap))

    chunks: list[Chunk] = []
    for doc in corpus["documents"]:
        units = split_units(doc)
        text = doc_text(doc, units)

        start = 0
        n = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            full, partial = [], []
            for u in units:
                if u.start >= end or u.end <= start:
                    continue  # 겹치지 않음
                if u.start >= start and u.end <= end:
                    full.append(u.unit_id)
                else:
                    partial.append(u.unit_id)

            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}#f{n}",
                    text=text[start:end],
                    units_full=full,
                    units_partial=partial,
                    doc_ids=[doc["doc_id"]],
                )
            )
            n += 1
            if end >= len(text):
                break
            start += step
    return chunks


def _article_chunks(corpus: dict) -> list[Chunk]:
    """항 단위 청킹. 단서는 본문 단위와 같은 청크에 둔다."""
    chunks: list[Chunk] = []
    for doc in corpus["documents"]:
        units = split_units(doc)
        header = doc_header(doc)

        # parent_id 기준으로 묶어 단서가 본문에서 떨어지지 않게 한다
        groups: dict[str, list[Unit]] = {}
        order: list[str] = []
        for u in units:
            key = u.parent_id
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(u)

        for i, key in enumerate(order):
            members = groups[key]
            body = "\n".join(u.rendered for u in members)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}#a{i}",
                    text=f"{header}\n{body}" if header else body,
                    units_full=[u.unit_id for u in members],
                    units_partial=[],
                    doc_ids=[doc["doc_id"]],
                )
            )
    return chunks


def chunk_corpus(
    corpus: dict,
    strategy: str = "fixed",
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    if strategy == "fixed":
        return _fixed_chunks(corpus, chunk_size, overlap)
    if strategy == "article":
        return _article_chunks(corpus)
    raise ValueError(f"알 수 없는 청킹 전략: {strategy}")

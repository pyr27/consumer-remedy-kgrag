"""Naive RAG 파이프라인 — 검색 한 번, 생성 한 번.

의도적으로 다음을 '하지 않는다'. 이것들이 제안 시스템에서 추가될 요소이며,
베이스라인과의 차이를 만드는 지점이기 때문이다.

- 질의 재작성 / 다단계 검색
- 검색 결과와 질의의 적합성 판정
- 조문 간 참조 관계(모법→시행령→기준) 추적
- 생성된 답변의 근거 검증 및 재검색
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .index import Hit, VectorIndex

PROMPT_TEMPLATE = """다음은 소비자가 보낸 질문입니다.

[질문]
{question}

[참고 자료]
{context}

위 참고 자료에 적힌 내용만 근거로 질문에 답하십시오.
근거가 된 조문 번호를 반드시 밝히고, 자료에 근거가 없으면
'제공된 자료만으로는 확인할 수 없습니다'라고 답하십시오."""


@dataclass
class RAGResult:
    qid: str | None
    question: str
    hits: list[Hit]
    answer: str
    prompt: str = ""
    retrieved_units_full: set[str] = field(default_factory=set)
    retrieved_units_any: set[str] = field(default_factory=set)


class NaiveRAG:
    def __init__(self, index: VectorIndex, llm, top_k: int = 4) -> None:
        self.index = index
        self.llm = llm
        self.top_k = top_k

    @staticmethod
    def _format_context(hits: list[Hit]) -> str:
        return "\n\n".join(f"({h.rank}) {h.chunk.text}" for h in hits)

    def _oracle_hits(self, units: set[str]) -> list[Hit]:
        """정답 근거 단위를 담은 청크를 강제로 컨텍스트에 넣는다.

        검색을 '완벽하게' 만든 조건이므로, 이 상태에서도 답이 틀리면
        그 실패는 검색이 아니라 추론에서 비롯된 것이다.
        """
        picked = [
            c
            for c in self.index.chunks
            if units & (set(c.units_full) | set(c.units_partial))
        ]
        return [Hit(rank=i + 1, score=1.0, chunk=c) for i, c in enumerate(picked)]

    def answer(
        self,
        question: str,
        qid: str | None = None,
        oracle_units: set[str] | None = None,
    ) -> RAGResult:
        if oracle_units:
            hits = self._oracle_hits(oracle_units)
            if not hits:  # 근거를 못 찾으면 일반 검색으로 대체
                hits = self.index.search(question, top_k=self.top_k)
        else:
            hits = self.index.search(question, top_k=self.top_k)
        prompt = PROMPT_TEMPLATE.format(
            question=question, context=self._format_context(hits)
        )
        answer = self.llm.generate(prompt)

        full: set[str] = set()
        any_: set[str] = set()
        for h in hits:
            full.update(h.chunk.units_full)
            any_.update(h.chunk.units_full)
            any_.update(h.chunk.units_partial)

        return RAGResult(
            qid=qid,
            question=question,
            hits=hits,
            answer=answer,
            prompt=prompt,
            retrieved_units_full=full,
            retrieved_units_any=any_,
        )

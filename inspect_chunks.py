#!/usr/bin/env python3
"""청킹 진단 — LLM 호출 없이 실행 가능.

고정 길이 청킹이 조문 구조를 어떻게 훼손하는지 수치로 보여준다.
'단서 조항이 본문과 분리된 비율'은 교수님 피드백에 대한 1차 근거로 바로 쓸 수 있다.

사용:
    python inspect_chunks.py
    python inspect_chunks.py --chunk-size 300 --overlap 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.chunking import chunk_corpus, split_units

ROOT = Path(__file__).resolve().parent


def analyze(corpus, strategy, chunk_size, overlap):
    chunks = chunk_corpus(
        corpus, strategy=strategy, chunk_size=chunk_size, overlap=overlap
    )
    units = [u for doc in corpus["documents"] for u in split_units(doc)]

    intact, split_only, absent = [], [], []
    for u in units:
        if u.unit_id == u.doc_id:
            continue
        if any(u.unit_id in c.units_full for c in chunks):
            intact.append(u.unit_id)
        elif any(u.unit_id in c.units_partial for c in chunks):
            split_only.append(u.unit_id)
        else:
            absent.append(u.unit_id)

    provisos = [u.unit_id for u in units if u.unit_id.endswith("-proviso")]
    proviso_broken = [p for p in provisos if p in split_only]

    return {
        "strategy": strategy,
        "chunk_size": chunk_size if strategy == "fixed" else None,
        "overlap": overlap if strategy == "fixed" else None,
        "n_chunks": len(chunks),
        "n_units": len(intact) + len(split_only) + len(absent),
        "units_intact": len(intact),
        "units_split_across_chunks": len(split_only),
        "unit_intact_ratio": round(
            len(intact) / max(1, len(intact) + len(split_only) + len(absent)), 3
        ),
        "n_provisos": len(provisos),
        "provisos_split": len(proviso_broken),
        "proviso_split_list": proviso_broken,
        "split_unit_list": split_only,
    }


def colocation(corpus, questions, strategy, chunk_size, overlap):
    """질문별로 정답 근거 조문이 몇 개 청크에 흩어져 있는지 센다.

    Naive RAG는 질의 1개를 임베딩해 상위 k개 청크만 가져온다. 근거가 여러 청크에
    흩어져 있고 그 청크들이 서로 다른 어휘를 쓰고 있으면, 한 번의 검색으로
    모두 회수될 확률은 급격히 떨어진다. 이것이 다단계 참조 실패의 구조적 원인이다.
    """
    chunks = chunk_corpus(
        corpus, strategy=strategy, chunk_size=chunk_size, overlap=overlap
    )
    rows = []
    for q in questions:
        gold = set(q["gold_citations"])
        if not gold:
            continue
        needed = set()
        for g in gold:
            owners = [c.chunk_id for c in chunks if g in c.units_full]
            if owners:
                needed.add(owners[0])
        single = any(gold <= set(c.units_full) for c in chunks)
        rows.append(
            {
                "qid": q["qid"],
                "hops": q["hops"],
                "n_gold_units": len(gold),
                "min_chunks_needed": len(needed),
                "single_chunk_covers_all": single,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=100)
    args = ap.parse_args()

    corpus = json.loads((ROOT / "data" / "legal_corpus.json").read_text(encoding="utf-8"))
    questions = json.loads(
        (ROOT / "data" / "test_questions.json").read_text(encoding="utf-8")
    )["questions"]

    reports = [
        analyze(corpus, "fixed", args.chunk_size, args.overlap),
        analyze(corpus, "fixed", 300, 50),
        analyze(corpus, "article", 0, 0),
    ]

    for r in reports:
        label = (
            f"고정 {r['chunk_size']}자 (overlap {r['overlap']})"
            if r["strategy"] == "fixed"
            else "항 단위"
        )
        print(f"\n[{label}]")
        print(f"  청크 수                     : {r['n_chunks']}")
        print(f"  조문 단위 총 개수           : {r['n_units']}")
        print(f"  온전히 한 청크에 담긴 단위  : {r['units_intact']} ({r['unit_intact_ratio']:.1%})")
        print(f"  청크 경계로 쪼개진 단위     : {r['units_split_across_chunks']}")
        print(f"  단서 조항 {r['n_provisos']}개 중 분리된 것 : {r['provisos_split']}")
        if r["proviso_split_list"]:
            print(f"    → {', '.join(r['proviso_split_list'])}")

    coloc = colocation(corpus, questions, "fixed", args.chunk_size, args.overlap)
    print(f"\n[근거 조문 분산도 — 고정 {args.chunk_size}자 청킹]")
    print("  qid   hops  근거수  최소필요청크  한청크로커버")
    for r in coloc:
        print(
            f"  {r['qid']}  {r['hops']:>4}  {r['n_gold_units']:>5}  "
            f"{r['min_chunks_needed']:>11}  {'예' if r['single_chunk_covers_all'] else '아니오'}"
        )
    multi = [r for r in coloc if r["hops"] >= 2]
    covered = sum(1 for r in multi if r["single_chunk_covers_all"])
    print(
        f"\n  다단계 질문 {len(multi)}개 중 단일 청크로 근거가 모두 커버되는 경우: "
        f"{covered}개 ({covered / max(1, len(multi)):.1%})"
    )
    print(
        f"  다단계 질문 평균 최소 필요 청크 수: "
        f"{sum(r['min_chunks_needed'] for r in multi) / max(1, len(multi)):.2f}"
    )

    out = ROOT / "results" / "chunking_diagnosis.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {"chunking_reports": reports, "colocation": coloc},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()

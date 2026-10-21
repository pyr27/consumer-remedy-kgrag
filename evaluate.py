#!/usr/bin/env python3
"""Naive RAG 베이스라인 실험 실행 및 자동 채점.

사용 예:
    # 키 없이 파이프라인 검증
    python evaluate.py --embedder tfidf --llm mock --tag smoke

    # 실제 측정 (보고서용)
    export OPENAI_API_KEY=...
    python evaluate.py --embedder openai --llm openai --tag naive_fixed
    python evaluate.py --embedder openai --llm openai --chunking article --tag naive_article

출력:
    results/run_<tag>.json          질문별 검색 결과·답변 원본
    results/metrics_<tag>.csv       질문별 자동 지표
    results/summary_<tag>.json      전체 요약 지표
    results/scoring_<tag>.csv       수동 채점용 시트 (정확성·환각 직접 기입)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from rag.backends import get_embedder, get_llm
from rag.chunking import chunk_corpus, corpus_unit_ids, split_units
from rag.index import VectorIndex
from rag.pipeline import NaiveRAG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# "제17조", "제20조의2", "제17조제2항제5호" 등을 포착
CITE_RE = re.compile(r"제\s*(\d{1,3})\s*조(?:\s*의\s*(\d{1,2}))?")

ABSTAIN_PATTERNS = [
    "확인할 수 없",
    "확인되지 않",
    "자료에 없",
    "자료에는 없",
    "포함되어 있지 않",
    "포함되지 않",
    "판단할 수 없",
    "알 수 없",
    "범위에 없",
    "제공된 자료만으로",
    "참고 자료만으로",
]


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def doc_all_text(doc: dict) -> str:
    """조문 구조가 있는 문서(units)와 없는 문서(text) 양쪽을 모두 처리."""
    if doc.get("units"):
        return "\n".join(u.get("text", "") for u in doc["units"])
    return doc.get("text", "")


def corpus_article_sets(corpus: dict) -> tuple[set[str], set[str]]:
    """(원문이 코퍼스에 있는 조, 코퍼스 본문에서 언급만 되는 조)"""
    present: set[str] = set()
    for doc in corpus["documents"]:
        m = CITE_RE.search(doc.get("article", ""))
        if m:
            present.add(_cite_key(m))

    mentioned: set[str] = set()
    for doc in corpus["documents"]:
        for m in CITE_RE.finditer(doc_all_text(doc)):
            mentioned.add(_cite_key(m))

    return present, mentioned - present


def _cite_key(m: re.Match) -> str:
    base, sub = m.group(1), m.group(2)
    return f"{base}-{sub}" if sub else base


def extract_citations(text: str) -> set[str]:
    return {_cite_key(m) for m in CITE_RE.finditer(text)}


def is_abstention(text: str) -> bool:
    return any(p in text for p in ABSTAIN_PATTERNS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="tfidf", choices=["tfidf", "openai", "local"])
    ap.add_argument("--llm", default="mock", choices=["mock", "openai", "anthropic"])
    ap.add_argument("--chunking", default="fixed", choices=["fixed", "article"])
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--tag", default="run")
    ap.add_argument(
        "--oracle",
        action="store_true",
        help="정답 근거 조문을 강제로 컨텍스트에 넣는다. "
             "이 조건에서도 틀리면 검색이 아니라 추론의 실패다.",
    )
    args = ap.parse_args()

    corpus = load_json(DATA / "legal_corpus.json")
    qset = load_json(DATA / "test_questions.json")
    questions = qset["questions"]

    all_units = corpus_unit_ids(corpus)
    present_articles, referenced_articles = corpus_article_sets(corpus)

    # gold_citations 오타 검증 — 코퍼스에 없는 unit_id를 참조하면 즉시 알려준다
    bad = {
        q["qid"]: [c for c in q["gold_citations"] if c not in all_units]
        for q in questions
    }
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        print("[경고] 코퍼스에 존재하지 않는 gold_citations:", bad, file=sys.stderr)

    chunks = chunk_corpus(
        corpus,
        strategy=args.chunking,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"청킹: {args.chunking}, 청크 수 {len(chunks)}")

    index = VectorIndex(get_embedder(args.embedder)).build(chunks)
    rag = NaiveRAG(index, get_llm(args.llm), top_k=args.top_k)

    RESULTS.mkdir(exist_ok=True)
    rows, runs = [], []

    for q in questions:
        gold = set(q["gold_citations"])
        res = rag.answer(
            q["question"],
            qid=q["qid"],
            oracle_units=gold if args.oracle else None,
        )
        got_full = res.retrieved_units_full
        got_any = res.retrieved_units_any

        recall_strict = len(gold & got_full) / len(gold) if gold else None
        recall_loose = len(gold & got_any) / len(gold) if gold else None
        hit_chunks = sum(
            1
            for h in res.hits
            if gold & (set(h.chunk.units_full) | set(h.chunk.units_partial))
        )
        precision = hit_chunks / len(res.hits) if res.hits else 0.0

        cites = extract_citations(res.answer)
        unknown = sorted(cites - present_articles - referenced_articles)
        ref_only = sorted(cites & referenced_articles)
        abstained = is_abstention(res.answer)

        rows.append(
            {
                "qid": q["qid"],
                "hops": q["hops"],
                "out_of_corpus": q["out_of_corpus"],
                "recall_strict": "" if recall_strict is None else f"{recall_strict:.2f}",
                "recall_loose": "" if recall_loose is None else f"{recall_loose:.2f}",
                "precision@k": f"{precision:.2f}",
                "missing_gold_units": ";".join(sorted(gold - got_full)),
                "cited_unknown_articles": ";".join(unknown),
                "cited_referenced_only": ";".join(ref_only),
                "abstained": abstained,
                "auto_hallucination_flag": bool(unknown)
                or (q["out_of_corpus"] and not abstained),
            }
        )
        runs.append(
            {
                "qid": q["qid"],
                "question": q["question"],
                "hops": q["hops"],
                "out_of_corpus": q["out_of_corpus"],
                "gold_citations": q["gold_citations"],
                "gold_points": q["gold_points"],
                "failure_hypothesis": q["failure_hypothesis"],
                "retrieved": [
                    {
                        "rank": h.rank,
                        "score": round(h.score, 4),
                        "chunk_id": h.chunk.chunk_id,
                        "units_full": h.chunk.units_full,
                        "units_partial": h.chunk.units_partial,
                        "text": h.chunk.text,
                    }
                    for h in res.hits
                ],
                "answer": res.answer,
            }
        )
        print(
            f"{q['qid']} hops={q['hops']} "
            f"recall_strict={rows[-1]['recall_strict'] or '-'} "
            f"halluc_flag={rows[-1]['auto_hallucination_flag']}"
        )

    # ---------------- 출력 ---------------- #
    cfg = {
        "embedder": args.embedder,
        "llm": args.llm,
        "chunking": args.chunking,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "top_k": args.top_k,
        "n_chunks": len(chunks),
        "oracle": args.oracle,
    }

    (RESULTS / f"run_{args.tag}.json").write_text(
        json.dumps({"config": cfg, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (RESULTS / f"metrics_{args.tag}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def avg(key, pred=lambda r: True):
        vals = [float(r[key]) for r in rows if r[key] != "" and pred(r)]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "config": cfg,
        "n_questions": len(rows),
        "recall_strict_mean": avg("recall_strict"),
        "recall_loose_mean": avg("recall_loose"),
        "precision_at_k_mean": avg("precision@k"),
        "recall_strict_1hop": avg("recall_strict", lambda r: r["hops"] == 1),
        "recall_strict_multihop": avg("recall_strict", lambda r: r["hops"] >= 2),
        "auto_hallucination_flagged": sum(1 for r in rows if r["auto_hallucination_flag"]),
        "out_of_corpus_questions": sum(1 for r in rows if r["out_of_corpus"]),
        "out_of_corpus_abstained": sum(
            1 for r in rows if r["out_of_corpus"] and r["abstained"]
        ),
        "answers_citing_unknown_articles": sum(
            1 for r in rows if r["cited_unknown_articles"]
        ),
    }
    (RESULTS / f"summary_{args.tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 수동 채점 시트
    with (RESULTS / f"scoring_{args.tag}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "qid", "hops", "질문", "정답 근거(gold)", "모델 답변",
                "answer_correctness(0/1/2)", "hallucination(0/1/2)",
                "누락된 쟁점", "관찰 메모",
            ]
        )
        for q, r in zip(questions, runs):
            w.writerow(
                [
                    q["qid"], q["hops"], q["question"],
                    " | ".join(q["gold_points"]),
                    r["answer"], "", "", "", "",
                ]
            )

    print("\n===== 요약 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n결과 저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""보류(held-out) 문항 후보 뽑기 — 검색·라우팅을 조정할 때 한 번도 보지 않은 소비자원 FAQ.

개발셋(questions_faq.jsonl 42문항)은 검색 v1을 조정하며 실패 사례를 들여다봤으므로 낙관적이다.
같은 모집단(품목별 FAQ 중 인용 법령이 코퍼스 안, contexts 있음)에서, 예전에 검수용으로 뽑힌 60문항과
개발셋 질문과 거의 같은 문항(소비자원 FAQ에 같은 질문이 다른 번호로 중복 수록됨)을 빼고 층화 추출한다.

    python analyze/make_heldout.py --n 30
        → data/v2/heldout_review.csv (검수용: AI 사전 점검 열 + 저자 확정 열)

정답 근거는 소비자원 답변이 인용한 조문(contexts_정확)과, 조까지만 인용해 항을 추정한 것(contexts_추정)이다.
추정분은 반드시 사람이 확인한다. 저자 검수 후 build_heldout.py로 questions_heldout.jsonl을 만든다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent / "naive_rag_baseline" / "naive_rag_baseline" / "data" / "v2"
OUT = ROOT / "data" / "v2" / "heldout_review.csv"


def grams(s: str, n: int = 3) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default=str(BASE / "kca_cite_all.csv"))
    ap.add_argument("--seen", default=str(BASE / "kca_cite_sample.csv"), help="예전 검수 표본 (제외)")
    ap.add_argument("--dev", default=str(ROOT / "data" / "v2" / "questions_faq.jsonl"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--dup", type=float, default=0.5, help="개발셋 질문과 3-gram 자카드가 이 이상이면 중복으로 제외")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.all, encoding="utf-8-sig")))
    seen = {r["faq_id"] for r in csv.DictReader(open(args.seen, encoding="utf-8-sig"))}
    dev = [json.loads(l) for l in open(args.dev, encoding="utf-8") if l.strip()]
    dev_g = [(q["id"], grams(q["question"])) for q in dev]

    pool, dropped = defaultdict(list), Counter()
    seen_q: dict[str, str] = {}
    for r in rows:
        if r["scope"] != "in_corpus" or not r["contexts"].strip():
            dropped["범위 밖·근거 없음"] += 1
            continue
        if r["faq_id"] in seen:
            dropped["예전 표본(60)"] += 1
            continue
        g = grams(r["원 질문"])
        near = max(((jaccard(g, dg), did) for did, dg in dev_g), default=(0, ""))
        if near[0] >= args.dup:
            dropped["개발셋과 중복"] += 1
            continue
        key = re.sub(r"\s+", "", r["원 질문"])
        if key in seen_q:
            dropped["후보끼리 중복"] += 1
            continue
        seen_q[key] = r["faq_id"]
        r["_near_dev"] = f"{near[1]}:{near[0]:.2f}"
        pool[(r["dispute_type"], r["구분"].split("_")[0])].append(r)

    print(f"전체 {len(rows)}, 제외 {dict(dropped)}, 후보 {sum(len(v) for v in pool.values())}")
    print("후보 유형:", dict(Counter(k[0] for k, v in pool.items() for _ in v)))
    rng = random.Random(args.seed)
    for v in pool.values():
        rng.shuffle(v)
    sample, keys = [], sorted(pool)
    while len(sample) < args.n and any(pool[k] for k in keys):
        for k in keys:
            if pool[k] and len(sample) < args.n:
                sample.append(pool[k].pop())
    sample.sort(key=lambda r: r["faq_id"])

    cols = ["faq_id", "품목", "구분", "dispute_type", "원 질문", "정답(소비자원 답변)", "인용(원문)",
            "contexts_정확", "contexts_추정", "contexts 원문(확인용)", "추정 대상 조의 전체 항(고를 때 참고)",
            "가장 비슷한 개발셋 문항", "ai_keep", "ai_contexts", "ai_note", "keep", "contexts", "메모"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sample:
            w.writerow({**{c: r.get(c, "") for c in cols}, "가장 비슷한 개발셋 문항": r["_near_dev"],
                        "ai_keep": "", "ai_contexts": r["contexts"], "ai_note": "", "keep": "", "contexts": "", "메모": ""})
    print(f"표본 {len(sample)}: 유형 {dict(Counter(r['dispute_type'] for r in sample))}, "
          f"근거 수 {dict(sorted(Counter(len(r['contexts'].split()) for r in sample).items()))}")
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

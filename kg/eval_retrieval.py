#!/usr/bin/env python3
"""검색만 평가 (답변 생성 없음): FAQ 42문항의 정답 근거가 LLM에 넘길 묶음에 들어오는가.

조건
    flat_vec4      범위 없이 vector 상위 4개 (≈ 베이스라인 cite: 항 단위 top-4)
    flat_hyb8      범위 없이 hybrid 상위 8개
    route_entry    라우팅 범위 안 hybrid 상위 8개 (확장 없음)
    route_pack     route_entry + 관계 확장 (근거 묶음, 예산 40)
    pack_rw        + 질의 재작성 (LLM이 뽑은 법률 용어로 fulltext, 질문+쟁점+용어로 vector)
    pack_art       + 조 단위 진입 (단위 순위를 조로 묶어 상위 조의 항들을 진입으로)
    pack_rw_art    재작성 + 조 단위 진입
    flat_sized     범위 없이 hybrid 상위 N개, N = 그 문항 pack_rw_art의 크기
                   (묶음이 '조항을 많이 넣어서' 이긴 것인지 가르는 대조군)
라우팅
    --route llm     LLM이 Category를 고름 (실제 시스템)
    --route oracle  FAQ dispute_type으로 정답 Category를 줌 (라우팅 오류를 뺀 상한)

사용
    python kg/eval_retrieval.py --route llm --tag r1
    python kg/eval_retrieval.py --route oracle --tag r1
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_graph import DATA, ROOT, connect  # noqa: E402
from retrieve import analyze, embed_query, entry, retrieve  # noqa: E402

ORACLE = {"continuous_contract": ["continuous"], "ecommerce_refund": ["ecommerce"],
          "door_sales": ["door_sales"], "term": [], "product_quality": ["dispute_standard"]}
# pack_*: 라우팅 범위 + 확장. rw = 법률 용어 재작성, art = 조 단위 진입
CONDS = ["flat_vec4", "flat_hyb8", "route_entry", "route_pack", "pack_rw", "pack_art", "pack_rw_art", "flat_sized"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faq", default=str(DATA / "v2" / "questions_faq.jsonl"))
    ap.add_argument("--route", choices=["llm", "oracle"], default="llm")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--tag", default="r1")
    args = ap.parse_args()

    qs = [json.loads(l) for l in Path(args.faq).open(encoding="utf-8") if l.strip()]
    d = connect()
    rows = []
    try:
        for q in qs:
            text = f"{q.get('background', '')}\n{q['question']}".strip()
            vec = embed_query(text)
            info = analyze(d, text)
            cats = ORACLE[q["dispute_type"]] if args.route == "oracle" else info["categories"]
            got = {
                "flat_vec4": [u for u, _ in entry(d, text, None, k=4, vec=vec, mode="vector")],
                "flat_hyb8": [u for u, _ in entry(d, text, None, k=args.k, vec=vec)],
            }
            base = retrieve(d, text, cats, k=args.k, budget=args.budget, vec=vec)
            got["route_entry"] = base.entries
            got["route_pack"] = base.unit_ids
            got["pack_rw"] = retrieve(d, text, cats, k=args.k, budget=args.budget, rewrite=info).unit_ids
            got["pack_art"] = retrieve(d, text, cats, k=args.k, budget=args.budget, vec=vec,
                                       by_article=True).unit_ids
            pack = retrieve(d, text, cats, k=args.k, budget=args.budget, rewrite=info, by_article=True)
            got["pack_rw_art"] = pack.unit_ids
            got["flat_sized"] = [u for u, _ in entry(d, text, None, k=len(pack.unit_ids), vec=vec)]
            gold = set(q["contexts"])
            row = {"id": q["id"], "dispute_type": q["dispute_type"], "n_gold": len(gold),
                   "multi": len(gold) >= 2, "route": "|".join(cats),
                   "route_ok": all(c in cats for c in ORACLE[q["dispute_type"]])}
            for c in CONDS:
                row[f"{c}_hit"] = len(gold & set(got[c]))
                row[f"{c}_n"] = len(got[c])
            row["pack_roles"] = json.dumps({i.unit_id: i.role for i in pack.items if i.unit_id in gold},
                                           ensure_ascii=False)
            row["missing"] = "|".join(sorted(gold - set(got["pack_rw_art"])))
            rows.append(row)
            print(f"  {q['id']} {q['dispute_type']:20} route={cats} pack_rw_art {row['pack_rw_art_hit']}/{len(gold)} "
                  f"({row['pack_rw_art_n']}개)")
    finally:
        d.close()

    n_gold = sum(r["n_gold"] for r in rows)
    multi = [r for r in rows if r["multi"]]
    print(f"\n라우팅={args.route}, 문항 {len(rows)} (근거 2개+ {len(multi)}), 정답 근거 {n_gold}")
    if args.route == "llm":
        print(f"라우팅 정확(oracle Category 포함): {sum(r['route_ok'] for r in rows)}/{len(rows)}")
    print(f"  {'조건':12} {'근거 재현':>12} {'전부 포함':>8} {'2개+ 전부':>9} {'묶음 크기 중앙값':>14}")
    for c in CONDS:
        hit = sum(r[f"{c}_hit"] for r in rows)
        full = sum(r[f"{c}_hit"] == r["n_gold"] for r in rows)
        mfull = sum(r[f"{c}_hit"] == r["n_gold"] for r in multi)
        size = statistics.median(r[f"{c}_n"] for r in rows)
        print(f"  {c:12} {hit:>4}/{n_gold} ({hit / n_gold:4.0%}) {full:>4}/{len(rows)} {mfull:>5}/{len(multi)} {size:>10.0f}")
    print("\npack_rw_art에서 빠진 근거:")
    for r in rows:
        if r["missing"]:
            print(f"  {r['id']} ({r['dispute_type']}, route={r['route']}): {r['missing']}")

    out = ROOT / "results" / "kg" / f"retrieval_{args.route}_{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

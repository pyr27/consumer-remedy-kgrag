#!/usr/bin/env python3
"""heldout_review.csv → data/v2/questions_heldout.jsonl

저자 확정 열(keep·contexts)이 채워진 행은 그것을, 비어 있으면 AI 사전 점검(ai_keep·ai_contexts)을 쓴다.
어느 쪽을 썼는지 verified_by에 남긴다: "author" 또는 "ai_precheck"(저자 검수 전 — 확정 수치로 쓰지 않는다).

    python analyze/build_heldout.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "data" / "v2" / "heldout_review.csv"
OUT = ROOT / "data" / "v2" / "questions_heldout.jsonl"


def main() -> int:
    corpus = json.loads((ROOT / "data" / "legal_corpus.json").read_text(encoding="utf-8"))
    valid = {u["unit_id"] for d in corpus["documents"] for u in (d.get("units") or [])} | \
            {d["doc_id"] for d in corpus["documents"]}
    out, errors = [], []
    for r in csv.DictReader(REVIEW.open(encoding="utf-8-sig")):
        author = bool(r["keep"].strip())
        keep = (r["keep"] if author else r["ai_keep"]).strip().upper()
        if keep != "Y":
            continue
        ctx = (r["contexts"] if author and r["contexts"].strip() else r["ai_contexts"]).split()
        bad = [c for c in ctx if c not in valid]
        if bad:
            errors.append(f"{r['faq_id']}: 코퍼스에 없는 unit_id {bad}")
        out.append({"id": r["faq_id"], "source": "faq_heldout", "background": "", "question": r["원 질문"].strip(),
                    "answer": r["정답(소비자원 답변)"].strip(), "contexts": ctx, "n_hops": len(ctx),
                    "dispute_type": r["dispute_type"], "verified_by": "author" if author else "ai_precheck",
                    "note": (r["메모"] if author else r["ai_note"]).strip()})
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"{len(out)}문항 → {OUT.name}  검수: {dict(Counter(q['verified_by'] for q in out))}  "
          f"유형: {dict(Counter(q['dispute_type'] for q in out))}  근거 수: {dict(sorted(Counter(q['n_hops'] for q in out).items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""질문셋 v2 평가. v1(evaluate.py)은 재현성을 위해 그대로 둔다.

조건:
    naive        검색 1회 + 생성 1회 (v1과 같은 프롬프트)
    oracle       정답 근거 조문을 강제로 주입 — 추론 실패만 분리
    cite         검색은 naive와 같고, 조문을 [unit_id]로만 인용하도록 강제
                 → 조문 번호 오귀속이 인용 방식만으로 얼마나 줄어드는지 측정
    full         선택한 법령 전체를 컨텍스트에 넣음 (검색 없음, [unit_id] 인용)
                 → "상위 모델에 법령을 통째로 넣으면 되지 않나"에 대한 답

사용:
    python evaluate_v2.py --cond naive  --tag v2_naive
    python evaluate_v2.py --cond oracle --tag v2_oracle
    python evaluate_v2.py --cond cite   --tag v2_cite
    OPENAI_MODEL=gpt-4o python evaluate_v2.py --cond full --laws ecom,enf,door,doorenf,std --tag v2_full_4o

    # 일부 세트만
    python evaluate_v2.py --cond naive --sets faq,generated --tag v2_naive_ab

출력: results/v2/{run,summary}_<tag>.json, scoring_<tag>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag.backends import get_embedder  # noqa: E402
from rag.chunking import chunk_corpus, corpus_unit_ids, split_units  # noqa: E402
from rag.index import VectorIndex  # noqa: E402

V2 = ROOT / "data" / "v2"
OUT = ROOT / "results" / "v2"

ABSTAIN = ["확인할 수 없", "제공된 자료만으로", "자료에 없", "판단할 수 없"]
ID_RE = re.compile(r"\[([a-z]+-[0-9][0-9a-z_\-]*)\]")

SYSTEM_NAIVE = (
    "당신은 한국 소비자 피해구제 상담을 돕는 어시스턴트입니다. "
    "아래 [참고 자료]에 실제로 적힌 내용만 근거로 답하십시오. "
    "참고 자료에 근거가 없으면 추측하지 말고 '제공된 자료만으로는 확인할 수 없습니다'라고 답하십시오. "
    "답변에는 근거가 된 조문 번호를 명시하십시오."
)
SYSTEM_CITE = (
    "당신은 한국 소비자 피해구제 상담을 돕는 어시스턴트입니다. "
    "아래 [참고 자료]에 실제로 적힌 내용만 근거로 답하십시오. "
    "근거를 밝힐 때는 각 조항 앞에 붙은 [ID]를 그대로 써서 인용하고, 조문 번호를 직접 쓰지 마십시오. "
    "예: '구매 후 7일 이내라면 철회할 수 있습니다 [ecom-17-1].' "
    "참고 자료에 근거가 없으면 '제공된 자료만으로는 확인할 수 없습니다'라고 답하십시오."
)
USER_TMPL = "[상황]\n{background}\n\n[질문]\n{question}\n\n[참고 자료]\n{context}"


# --------------------------------------------------------------------------- #
def load_questions(sets: list[str]) -> list[dict]:
    qs = []
    for name in sets:
        p = V2 / f"questions_{name}.jsonl"
        if not p.exists():
            print(f"[건너뜀] {p.name} 없음", file=sys.stderr)
            continue
        qs += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return qs


class Chat:
    def __init__(self, kind: str):
        self.kind = kind
        if kind == "openai":
            from openai import OpenAI
            base = os.getenv("OPENAI_BASE_URL")
            self.client = OpenAI(base_url=base) if base else OpenAI()
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif kind == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        else:
            self.model = "mock"

    def __call__(self, system: str, user: str) -> str:
        if self.kind == "mock":
            ids = ID_RE.findall(user)
            return f"검증용 답변입니다 [{ids[0]}]." if ids else "제공된 자료만으로는 확인할 수 없습니다."
        if self.kind == "openai":
            r = self.client.chat.completions.create(
                model=self.model, temperature=0,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return (r.choices[0].message.content or "").strip()
        r = self.client.messages.create(model=self.model, max_tokens=1500, temperature=0,
                                        system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text").strip()


def unit_table(corpus: dict) -> dict[str, dict]:
    tbl = {}
    for d in corpus["documents"]:
        for u in split_units(d):
            tbl[u.unit_id] = {"text": u.rendered, "doc": d, "prefix": d["doc_id"].split("-")[0]}
    return tbl


def parents(uid: str, tbl) -> list[str]:
    """호·단서는 상위 항 문장 없이는 뜻이 안 통한다(예: '…한 경우' + '철회할 수 없다').
    ecom-17-2-1 → [ecom-17-2], ecom-17-2-1-proviso → [ecom-17-2, ecom-17-2-1]."""
    doc_id = tbl[uid]["doc"]["doc_id"]
    parts, chain = uid.split("-"), []
    n_doc = len(doc_id.split("-"))
    for k in range(n_doc + 1, len(parts)):
        p = "-".join(parts[:k])
        if p in tbl:
            chain.append(p)
    return chain


def fmt_units(ids, tbl, with_parents: bool = True) -> str:
    out, shown = [], set()
    for uid in ids:
        if uid not in tbl:  # 코퍼스에 없는 gold — 시작 시 경고로 이미 알렸다
            continue
        d = tbl[uid]["doc"]
        head = f"{d.get('law', '')} {d.get('article', '')}({d.get('title', '')})".strip()
        if with_parents:
            for p in parents(uid, tbl):
                if p not in shown and p not in ids:
                    out.append(f"[{p}] {head} {tbl[p]['text']}  ※상위 조항(맥락)")
                    shown.add(p)
        out.append(f"[{uid}] {head} {tbl[uid]['text']}")
        shown.add(uid)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["naive", "oracle", "cite", "full"])
    ap.add_argument("--sets", default="faq,generated,routing,oos,proc")
    ap.add_argument("--embedder", default="openai", choices=["tfidf", "openai", "local"])
    ap.add_argument("--llm", default="openai", choices=["openai", "anthropic", "mock"])
    ap.add_argument("--chunking", default="fixed", choices=["fixed", "article"])
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--laws", default="ecom,enf,door,doorenf,term,std",
                    help="full 조건에서 컨텍스트에 넣을 법령 prefix")
    ap.add_argument("--corpus", default=str(ROOT / "data" / "legal_corpus.json"))
    ap.add_argument("--include-handmade", action="store_true",
                    help="v1용 수작업 분쟁해결기준 요약문도 포함")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    # 태그에 \ / 공백 등이 섞이면 파일 경로가 깨져 실행 끝에 결과를 잃는다 → 미리 정리
    clean = re.sub(r"[^0-9A-Za-z_\-]", "", args.tag)
    if clean != args.tag:
        print(f"[안내] 태그 '{args.tag}' → '{clean}' (파일명에 쓸 수 없는 문자 제거)")
        args.tag = clean

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    if not args.include_handmade:  # v1용 수작업 요약문(std-fitness 등)은 공식 원문이 아니므로 제외
        corpus = {**corpus, "documents": [d for d in corpus["documents"] if not d.get("handmade")]}
    tbl = unit_table(corpus)
    valid = corpus_unit_ids(corpus)
    qs = load_questions(args.sets.split(","))
    if not qs:
        sys.exit("문항이 없습니다. data/v2/questions_*.jsonl 을 확인하세요.")

    bad = {q["id"]: [c for c in q["contexts"] if c not in valid] for q in qs}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        print("[경고] 코퍼스에 없는 contexts:", bad, file=sys.stderr)

    chat = Chat(args.llm)
    cite_mode = args.cond in ("cite", "full")
    system = SYSTEM_CITE if cite_mode else SYSTEM_NAIVE

    index = None
    if args.cond in ("naive", "cite"):
        # cite 조건은 청크 안의 각 조항에 ID를 붙여야 하므로 항 단위 청킹을 쓴다
        strategy = "article" if args.cond == "cite" else args.chunking
        chunks = chunk_corpus(corpus, strategy=strategy)
        index = VectorIndex(get_embedder(args.embedder)).build(chunks)
        print(f"청킹 {strategy}, 청크 {len(chunks)}개")

    full_ctx = None
    if args.cond == "full":
        keep = set(args.laws.split(","))
        ids = [u for u in tbl if tbl[u]["prefix"] in keep]
        full_ctx = fmt_units(ids, tbl)
        print(f"법령 전체 주입: 조항 {len(ids)}개, 약 {len(full_ctx):,}자 "
              f"(토큰 수는 모델 토크나이저에 따라 다름 — 컨텍스트 한도 확인)")

    print(f"모델 {chat.model}, 문항 {len(qs)}개, 조건 {args.cond}")
    runs, rows = [], []
    for q in qs:
        gold = set(q["contexts"])
        query = f"{q.get('background', '')}\n{q['question']}".strip()
        provided: list[str] = []

        if args.cond == "full":
            context = full_ctx
            provided = None  # 전체가 제공됨
        elif args.cond == "oracle":
            provided = list(q["contexts"])
            context = fmt_units(provided, tbl) if provided else "(근거 조항 없음)"
        else:
            hits = index.search(query, top_k=args.top_k)
            if cite_mode:
                for h in hits:
                    provided += [u for u in h.chunk.units_full if u not in provided]
                context = fmt_units(provided, tbl)
            else:
                for h in hits:
                    provided += [u for u in h.chunk.units_full if u not in provided]
                context = "\n\n".join(f"({h.rank}) {h.chunk.text}" for h in hits)

        answer = chat(system, USER_TMPL.format(background=q.get("background", ""),
                                               question=q["question"], context=context))

        rec = None
        if gold and provided is not None:
            rec = len(gold & set(provided)) / len(gold)
        cited = set(ID_RE.findall(answer)) if cite_mode else set()
        invalid = sorted(cited - valid)
        outside = sorted((cited & valid) - set(provided)) if (cite_mode and provided is not None) else []
        gold_hit = (len(cited & gold) / len(gold)) if (cite_mode and gold) else None

        row = {
            "id": q["id"], "source": q["source"], "n_hops": q.get("n_hops", 0),
            "dispute_type": q.get("dispute_type", ""),
            "recall": "" if rec is None else f"{rec:.2f}",
            "full_recall": "" if rec is None else str(rec >= 1.0),
            "abstained": str(any(p in answer for p in ABSTAIN)),
            "cited_ids": " ".join(sorted(cited)),
            "cited_invalid": " ".join(invalid),
            "cited_outside_context": " ".join(outside),
            "cite_gold_hit": "" if gold_hit is None else f"{gold_hit:.2f}",
        }
        rows.append(row)
        runs.append({**q, "condition": args.cond, "provided_units": provided, "model_answer": answer})
        print(f"  {q['id']:16} hop={row['n_hops']} recall={row['recall'] or '-':>4} "
              f"cite_hit={row['cite_gold_hit'] or '-':>4} invalid={len(invalid)}")

    # ---- 요약 ----
    def mean(vals):
        vals = [float(v) for v in vals if v != ""]
        return round(sum(vals) / len(vals), 3) if vals else None

    by = defaultdict(list)
    for r in rows:
        by[(r["source"], r["n_hops"])].append(r)
    summary = {
        "config": {**vars(args), "model": chat.model, "n_questions": len(rows)},
        "full_recall_rate": mean([1.0 if r["full_recall"] == "True" else 0.0
                                  for r in rows if r["full_recall"] != ""]),
        "abstain_rate": mean([1.0 if r["abstained"] == "True" else 0.0 for r in rows]),
        # 세트마다 '거부가 정답'인지가 달라 전체 평균은 오해를 부른다 → 세트별로 따로 본다
        "abstain_by_source": {
            src: mean([1.0 if r["abstained"] == "True" else 0.0 for r in rows if r["source"] == src])
            for src in sorted({r["source"] for r in rows})
        },
        "cite_gold_hit_mean": mean([r["cite_gold_hit"] for r in rows]),
        "answers_with_invalid_ids": sum(1 for r in rows if r["cited_invalid"]),
        "answers_citing_outside_context": sum(1 for r in rows if r["cited_outside_context"]),
        "by_source_hop": {
            f"{s}/{h}hop": {
                "n": len(g),
                "full_recall_rate": mean([1.0 if r["full_recall"] == "True" else 0.0
                                          for r in g if r["full_recall"] != ""]),
                "cite_gold_hit": mean([r["cite_gold_hit"] for r in g]),
            } for (s, h), g in sorted(by.items())
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"run_{args.tag}.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"summary_{args.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / f"scoring_{args.tag}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "n_hops", "상황", "질문", "정답", "기대 안내기관", "모델 답변",
                    "결론정확성(0/1/2)", "근거적합성(0/1/2)", "안내적절성(0/1/2)", "멈춘 지점", "메모"])
        for q, r in zip(qs, runs):
            agencies = ", ".join(q.get("expected_route", {}).get("agencies", []))
            w.writerow([q["id"], q["source"], q.get("n_hops", 0), q.get("background", ""),
                        q["question"], q["answer"], agencies, r["model_answer"], "", "", "", "", ""])

    print("\n===== 요약 =====")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, ensure_ascii=False, indent=2))
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

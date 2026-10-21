#!/usr/bin/env python3
"""생성형 문항(B세트) — KoBLEX 방식을 소비자 법령에 적용한다.

참고: Lee et al., "KoBLEX: Open Legal Question Answering with Multi-hop
Reasoning", EMNLP 2025. 생성·검증 프롬프트를 한국어 소비자 분쟁 맥락으로
번안했다 (원 저장소 라이선스 CC BY-NC 4.0, 학술 용도).

KoBLEX와 다른 점:
  1. 조문 조합을 무작위로 고르지 않고 **법령 원문에 적힌 상호참조**
     ("제N조제M항에 따른", "법 제N조에서 대통령령으로 정하는")에서 뽑는다.
     시스템이 사용할 의미 엣지(OVERRIDES 등)에서 뽑으면 시험지가 시스템에
     유리하게 짜이므로 피한다.
  2. 질문은 법률 용어를 쓰지 않는 **일상어**로 만든다. Naive RAG 실험에서
     확인한 어휘 격차를 시험지가 감추지 않게 하기 위해서다.

단계:
    python generate_questions.py seeds                 # 조문 조합 추출 (API 호출 없음)
    python generate_questions.py generate --limit 5    # 소량으로 먼저 확인
    python generate_questions.py generate              # 전체 생성 + 자동 검사
    → data/v2/gen_review.csv 를 열어 사람이 검수 (keep, 수정 열)
    python generate_questions.py build data/v2/gen_review.csv
    → data/v2/questions_generated.jsonl

자동 검사 (KoBLEX full/partial check 번안):
    full    : 전체 조문만으로 시나리오·정답이 성립하고 도출 가능한가
    partial : 조문을 하나씩 뺐을 때 답할 수 없어야 한다 (hop 수가 실제임을 확인)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "data" / "v2"

# 시행령 → 모법
PARENT_LAW = {"enf": "ecom", "consenf": "cons", "doorenf": "door"}

# 소비자가 물을 일이 없는 조문은 제외한다
DENY_TITLE = re.compile(
    r"벌칙|과태료|양벌|과징금|시정조치|시정권고|조사|공무원|위임|위탁|수수료|"
    r"보고|검사|등록|신고|협회|공제조합|교육|포상|비밀|청문|임시중지|동의의결|이행강제금"
)
# 행정기관이 주어인 조문(고시·지침 제정 등)도 소비자 분쟁 문제로 만들 수 없다
DENY_TEXT = re.compile(
    r"^(공정거래위원회|시ㆍ도지사|시장ㆍ군수|국가|지방자치단체|위원회|원장|장관)[은는]|"
    r"고시할 수 있다|지침을.*정할 수 있다|대통령령으로 정한다\.?$"
)
# 소비자기본법은 범위가 넓어 피해구제·분쟁조정 관련 조문만 쓴다
CONS_ALLOW_TITLE = re.compile(r"피해구제|분쟁조정|분쟁|합의|조정|처리기간|소비자단체")

REF_RE = re.compile(
    r"(?P<law>법\s*)?제\s*(?P<art>\d+)\s*조(?:\s*의\s*(?P<br>\d+))?"
    r"(?:\s*제\s*(?P<para>\d+)\s*항)?(?:\s*제\s*(?P<item>\d+)\s*호)?"
)
SAME_ART_PARA_RE = re.compile(r"(?<!조)(?<!조\s)제\s*(?P<para>\d+)\s*항(?:\s*제\s*(?P<item>\d+)\s*호)?")


# --------------------------------------------------------------------------- #
# 코퍼스와 상호참조
# --------------------------------------------------------------------------- #
def load_corpus(path: Path):
    corpus = json.loads(path.read_text(encoding="utf-8"))
    units, meta = {}, {}
    for d in corpus["documents"]:
        prefix = d["doc_id"].split("-")[0]
        for u in d.get("units") or [{"unit_id": d["doc_id"], "label": "", "text": d.get("text", "")}]:
            units[u["unit_id"]] = u["text"]
            meta[u["unit_id"]] = {
                "doc_id": d["doc_id"], "prefix": prefix,
                "law": d.get("law", ""), "article": d.get("article", ""),
                "title": d.get("title", ""), "label": u.get("label", ""),
            }
    return units, meta


def relevant(uid: str, meta: dict, units: dict | None = None) -> bool:
    m = meta[uid]
    if m["prefix"] == "std":
        return False  # 분쟁해결기준은 상호참조가 없어 seed로 쓰지 않는다
    if DENY_TITLE.search(m["title"]):
        return False
    if units is not None and DENY_TEXT.search(units[uid].strip()):
        return False
    if m["prefix"] in ("cons", "consenf") and not CONS_ALLOW_TITLE.search(m["title"]):
        return False
    return True


def resolve(prefix: str, art: str, br: str | None, para: str | None,
            item: str | None, units: dict) -> str | None:
    """참조를 가장 구체적인 unit_id로 해석한다. 없으면 덜 구체적인 쪽으로 물러난다."""
    base = f"{prefix}-{art}" + (f"_{br}" if br and br != "0" else "")
    for cand in (
        f"{base}-{para}-{item}" if para and item else None,
        f"{base}-{para}" if para else None,
        base,
        f"{base}-1",
    ):
        if cand and cand in units:
            return cand
    return None


def extract_refs(units: dict, meta: dict) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    for uid, text in units.items():
        m = meta[uid]
        prefix = m["prefix"]
        art_no = re.match(r"제(\d+)조(?:의(\d+))?", m["article"] or "")
        for r in REF_RE.finditer(text):
            # 다른 법률(「…」) 인용은 제외
            before = text[max(0, r.start() - 3): r.start()]
            if "」" in before:
                continue
            target_prefix = PARENT_LAW.get(prefix, prefix) if r.group("law") else prefix
            tgt = resolve(target_prefix, r.group("art"), r.group("br"),
                          r.group("para"), r.group("item"), units)
            if tgt and tgt != uid:
                edges[uid].add(tgt)
        # "제N항"처럼 조 없이 쓴 참조 → 같은 조
        if art_no:
            for r in SAME_ART_PARA_RE.finditer(text):
                tgt = resolve(prefix, art_no.group(1), art_no.group(2),
                              r.group("para"), r.group("item"), units)
                if tgt and tgt != uid:
                    edges[uid].add(tgt)
    return edges


def build_seeds(units, meta, edges, per_hop: int, rng: random.Random):
    ok = lambda u: relevant(u, meta, units) and len(units[u]) >= 30  # noqa: E731
    seeds = []

    # 2홉: 원문 상호참조 a → b
    pairs = sorted({tuple(sorted((a, b))) for a, bs in edges.items() for b in bs
                    if ok(a) and ok(b)})
    # 2홉(단서): 본문과 단서 — 원문의 "다만," 구분에서 온 것이므로 시스템 설계와 무관
    prov = sorted((u[: -len("-proviso")], u) for u in units
                  if u.endswith("-proviso") and u[: -len("-proviso")] in units
                  and ok(u[: -len("-proviso")]))
    # 3홉: a → b → c 경로, 또는 a → b, a → c
    triples = set()
    for a, bs in edges.items():
        if not ok(a):
            continue
        for b in bs:
            if not ok(b):
                continue
            for c in edges.get(b, ()):
                if ok(c) and c not in (a, b):
                    triples.add(tuple(sorted((a, b, c))))
        for b, c in combinations(sorted(x for x in bs if ok(x)), 2):
            triples.add(tuple(sorted((a, b, c))))
    triples = sorted(triples)
    # 1홉: 참조 관계에 등장하는 조문 위주로 (소비자 분쟁의 중심 조문일 가능성이 높다)
    hubs = sorted({u for a, bs in edges.items() for u in (a, *bs) if ok(u)})

    def take(pool, k):
        pool = list(pool)
        rng.shuffle(pool)
        return pool[:k]

    for u in take(hubs, per_hop):
        seeds.append({"seed_type": "single", "contexts": [u]})
    n_prov = min(len(prov), per_hop // 3)
    for p in take(prov, n_prov):
        seeds.append({"seed_type": "proviso", "contexts": list(p)})
    for p in take(pairs, per_hop - n_prov):
        seeds.append({"seed_type": "xref", "contexts": list(p)})
    for t in take(triples, per_hop):
        seeds.append({"seed_type": "xref_chain", "contexts": list(t)})
    return seeds, {"pairs": len(pairs), "proviso": len(prov), "triples": len(triples), "hubs": len(hubs)}


def fmt_ctx(ids, units, meta) -> str:
    lines = []
    for i, u in enumerate(ids, 1):
        m = meta[u]
        lines.append(f"[조항{i}] {m['law']} {m['article']}({m['title']}) {m['label']} — {units[u]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
GEN_PROMPT = """다음은 한국 법령 조항들이다.

{ctx}

이 조항들을 **모두** 적용해야만 답할 수 있는 소비자 분쟁 문제를 하나 만들어라.

규칙:
1. background: 일반 소비자가 실제로 겪을 법한 구체적인 사실관계. 결론을 좌우하는 사실(경과 일수, 사용 여부, 사전 고지 여부, 결제 방식 등)을 빠짐없이 넣는다. 등장인물은 '소비자', '판매자', '업체' 등으로 쓴다.
2. question: 그 소비자가 실제로 물을 법한 일상어 질문. 조문 번호, 법령 이름, '청약철회등', '통신판매업자', '재화등' 같은 법률 용어를 질문에 쓰지 않는다.
3. answer: 결론과 그 이유. 위 조항에 적힌 내용만 근거로 하고, 조항 밖의 법률 지식을 가정하지 않는다.
4. 조항 중 어느 하나라도 빠지면 정답을 낼 수 없어야 한다.
5. 조항들이 논리적으로 이어지지 않아 이런 문제를 만들 수 없다면 {{"skip": true}} 만 출력한다.

JSON으로만 출력한다: {{"background": "...", "question": "...", "answer": "..."}}"""

FULL_CHECK = """다음은 법률 문제의 사실관계, 질문, 제안된 정답, 그리고 근거 조항이다.

[사실관계]
{background}

[질문]
{question}

[제안된 정답]
{answer}

[근거 조항]
{ctx}

다음 세 가지를 판정하라. 근거 조항에 적힌 내용만 기준으로 하고 외부 법률 지식을 가정하지 마라.
- consistent: 사실관계와 질문이 근거 조항만으로 구성 가능한가
- correct: 제안된 정답이 사실관계와 조항에 비추어 법적으로 맞는가
- derivable: 정답이 근거 조항만으로 빠짐없이 도출되는가

JSON으로만 출력: {{"consistent": true/false, "correct": true/false, "derivable": true/false, "reason": "한국어로 간단히"}}"""

PARTIAL_CHECK = """다음 질문에, 아래 조항만 가지고 **완전한** 답을 낼 수 있는지 판정하라. 조항 밖의 법률 지식은 가정하지 마라.

[사실관계]
{background}

[질문]
{question}

[조항]
{ctx}

JSON으로만 출력: {{"answerable": true/false, "reason": "한국어로 간단히"}}"""


class LLM:
    def __init__(self, provider: str, model: str | None):
        self.provider = provider
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
            self.model = model or os.getenv("GEN_MODEL", "gpt-4o")
        else:
            import anthropic
            self.client = anthropic.Anthropic()
            self.model = model or os.getenv("GEN_MODEL", "claude-sonnet-4-5")

    def json(self, prompt: str) -> dict:
        if self.provider == "openai":
            r = self.client.chat.completions.create(
                model=self.model, temperature=0.7 if "만들어라" in prompt else 0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = r.choices[0].message.content or "{}"
        else:
            r = self.client.messages.create(
                model=self.model, max_tokens=1500,
                temperature=0.7 if "만들어라" in prompt else 0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in r.content if b.type == "text")
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


# --------------------------------------------------------------------------- #
# 명령
# --------------------------------------------------------------------------- #
def cmd_seeds(args) -> int:
    units, meta = load_corpus(Path(args.corpus))
    edges = extract_refs(units, meta)
    seeds, stats = build_seeds(units, meta, edges, args.per_hop, random.Random(args.seed))
    V2.mkdir(parents=True, exist_ok=True)
    out = V2 / "gen_seeds.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, s in enumerate(seeds, 1):
            s["seed_id"] = f"seed_{i:03d}"
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    n_edges = sum(len(v) for v in edges.values())
    print(f"단위 {len(units)} / 원문 상호참조 {n_edges}건")
    print(f"후보: 2홉 상호참조 {stats['pairs']}, 본문-단서 {stats['proviso']}, "
          f"3홉 {stats['triples']}, 1홉 {stats['hubs']}")
    print("seed:", dict(Counter(s["seed_type"] for s in seeds)), f"→ {out}")
    if args.show:
        for s in seeds[: args.show]:
            print(f"\n[{s['seed_id']} {s['seed_type']}]")
            print(fmt_ctx(s["contexts"], units, meta)[:600])
    return 0


def cmd_generate(args) -> int:
    units, meta = load_corpus(Path(args.corpus))
    seeds = [json.loads(l) for l in (V2 / "gen_seeds.jsonl").open(encoding="utf-8")]
    if args.limit:
        seeds = seeds[: args.limit]
    llm = LLM(args.provider, args.model)
    print(f"생성 모델: {llm.provider}/{llm.model}, seed {len(seeds)}개")

    drafts = []
    for s in seeds:
        ids = s["contexts"]
        ctx = fmt_ctx(ids, units, meta)
        g = llm.json(GEN_PROMPT.format(ctx=ctx))
        if g.get("skip") or not g.get("question"):
            print(f"  {s['seed_id']}: 해당 없음")
            continue
        full = llm.json(FULL_CHECK.format(ctx=ctx, **{k: g.get(k, "") for k in ("background", "question", "answer")}))
        partial = []
        if len(ids) > 1:
            for drop in ids:
                sub = [u for u in ids if u != drop]
                p = llm.json(PARTIAL_CHECK.format(ctx=fmt_ctx(sub, units, meta),
                                                  background=g.get("background", ""),
                                                  question=g.get("question", "")))
                partial.append({"without": drop, **p})
        full_ok = all(full.get(k) is True for k in ("consistent", "correct", "derivable"))
        hop_ok = not any(p.get("answerable") is True for p in partial)
        status = "pass" if full_ok and hop_ok else "flag"
        reasons = []
        if not full_ok:
            reasons.append("full: " + full.get("reason", ""))
        for p in partial:
            if p.get("answerable") is True:
                reasons.append(f"{p['without']} 없이도 답 가능 → hop 과대")
        drafts.append({**s, **{k: g.get(k, "") for k in ("background", "question", "answer")},
                       "n_hops": len(ids), "full_check": full, "partial_check": partial,
                       "auto_status": status, "auto_reason": " / ".join(reasons),
                       "gen_model": f"{llm.provider}/{llm.model}"})
        print(f"  {s['seed_id']} {s['seed_type']:10} {len(ids)}홉 → {status}")

    with (V2 / "gen_drafts.jsonl").open("w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    review = V2 / "gen_review.csv"
    with review.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed_id", "seed_type", "n_hops", "auto_status", "auto_reason",
                    "contexts", "근거 조항", "background", "question", "answer",
                    "keep", "수정_background", "수정_question", "수정_answer", "verified_by", "메모"])
        for d in drafts:
            w.writerow([d["seed_id"], d["seed_type"], d["n_hops"], d["auto_status"], d["auto_reason"],
                        " ".join(d["contexts"]), fmt_ctx(d["contexts"], units, meta),
                        d["background"], d["question"], d["answer"],
                        "", "", "", "", "", ""])
    c = Counter((d["n_hops"], d["auto_status"]) for d in drafts)
    print("\n결과 (hop, 상태):", dict(sorted(c.items())))
    print(f"검수 시트: {review}")
    return 0


def cmd_build(args) -> int:
    with Path(args.review).open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r.get("keep", "").strip().upper() != "Y":
            continue
        pick = lambda k: (r.get(f"수정_{k}") or r.get(k) or "").strip()  # noqa: E731
        ctx = r["contexts"].split()
        out.append({
            "id": f"gen_{len(ctx)}hop_{r['seed_id'].split('_')[-1]}",
            "source": "generated",
            "background": pick("background"),
            "question": pick("question"),
            "answer": pick("answer"),
            "contexts": ctx,
            "n_hops": len(ctx),
            "dispute_type": "ecommerce_refund",
            "verified_by": r.get("verified_by", "").strip() or "author",
            "seed_type": r["seed_type"],
            "note": r.get("메모", "").strip(),
        })
    dst = V2 / "questions_generated.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"{len(out)}문항 저장: {dst}")
    print("hop 분포:", dict(sorted(Counter(q["n_hops"] for q in out).items())))
    print("검수자:", dict(Counter(q["verified_by"] for q in out)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "legal_corpus.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seeds")
    s.add_argument("--per-hop", type=int, default=25, help="hop별 seed 수 (검수 탈락을 감안해 목표보다 넉넉히)")
    s.add_argument("--seed", type=int, default=20261015)
    s.add_argument("--show", type=int, default=0, help="앞 N개 seed의 조문을 출력")

    g = sub.add_parser("generate")
    g.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    g.add_argument("--model")
    g.add_argument("--limit", type=int, default=0)

    b = sub.add_parser("build")
    b.add_argument("review")

    args = ap.parse_args()
    return {"seeds": cmd_seeds, "generate": cmd_generate, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

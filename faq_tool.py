#!/usr/bin/env python3
"""한국소비자원 FAQ로 실사용 문항(A세트)을 만든다.

정답을 AI가 아니라 한국소비자원 공식 답변으로 고정하는 것이 목적이다.
사람이 해야 하는 일(근거 조문 태깅)은 줄이되, 판단은 사람에게 남긴다.

사용 (권장 — 품목별 FAQ: 소비자원 답변이 직접 인용한 조문을 정답 근거로 쓴다):
    python faq_tool.py cite 한국소비자원_품목별FAQ.csv
        → data/v2/kca_cite_all.csv (전체), kca_cite_sample.csv (검수용 표본)
    python faq_tool.py proc 한국소비자원_피해구제분쟁조정FAQ.csv
        → data/v2/questions_proc.jsonl (절차 안내 문항)
    # kca_cite_sample.csv 검수(keep=Y, contexts 확인) 후
    python faq_tool.py build data/v2/kca_cite_sample.csv

사용 (구 방식 — 인용 조문이 없는 FAQ에 TF-IDF로 후보를 붙인다):
    # 1) 후보 추출 — 전자상거래·계속거래 관련 FAQ만 골라 근거 조문 후보를 붙인다
    python faq_tool.py extract <다운로드한_FAQ.csv>
        → data/v2/faq_candidates.csv

    # 2) faq_candidates.csv를 엑셀로 열어 태깅
    #    keep / contexts / n_hops / dispute_type 열을 채운다 (아래 '태깅 규칙' 참고)

    # 3) 태깅 결과를 v2 JSONL로 변환 + 검증
    python faq_tool.py build data/v2/faq_candidates.csv
        → data/v2/questions_faq.jsonl

태깅 규칙:
    keep         Y면 포함, 비우거나 N이면 제외
    contexts     정답 근거 unit_id를 공백으로 구분 (예: ecom-17-1 ecom-17-3)
                 후보조문 열에 제안이 있으나 맞는지는 반드시 사람이 확인한다
    n_hops       근거 조문 개수가 아니라 '추론 단계 수'. 보통 contexts 개수와 같다
    dispute_type ecommerce_refund / continuous_contract / 그 외 routing 유형
    background   FAQ 질문이 일반론이면 비워도 된다
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "v2"

KEYWORDS = [
    "전자상거래", "통신판매", "온라인", "인터넷", "쇼핑몰", "오픈마켓", "앱",
    "청약철회", "반품", "환불", "환급", "교환", "배송", "택배",
    "계속거래", "헬스", "체력단련", "필라테스", "요가", "학원", "해지", "위약금",
    "구독", "정기결제", "해외구매", "직구", "라이브커머스", "중고거래",
    # 절차 판단 기능 평가용 — 피해구제·분쟁조정 절차를 묻는 FAQ도 포함한다
    "피해구제", "분쟁조정", "조정결정", "합의권고", "내용증명", "1372",
]

QUESTION_COL_HINTS = ["질문", "제목", "question"]
ANSWER_COL_HINTS = ["답변", "내용", "answer"]
CATEGORY_COL_HINTS = ["구분", "분류", "카테고리", "category"]

TAG_COLUMNS = ["keep", "background", "question", "contexts", "n_hops", "dispute_type", "메모"]


# --------------------------------------------------------------------------- #
# 공통
# --------------------------------------------------------------------------- #
def read_csv_any(path: Path) -> list[dict]:
    """공공데이터 CSV는 cp949인 경우가 많아 인코딩을 차례로 시도한다."""
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            with path.open(encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                print(f"인코딩: {enc}, {len(rows)}행")
                return rows
        except UnicodeDecodeError:
            continue
    sys.exit("CSV 인코딩을 판별하지 못했습니다.")


def find_col(cols: list[str], hints: list[str]) -> str | None:
    for h in hints:
        for c in cols:
            if h.lower() in c.lower():
                return c
    return None


def load_units(corpus_path: Path) -> list[tuple[str, str]]:
    """(unit_id, 표시용 텍스트) 목록."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    out = []
    for d in corpus["documents"]:
        head = f"{d.get('law', '')} {d.get('article', '')}".strip()
        units = d.get("units") or [{"unit_id": d["doc_id"], "label": "", "text": d.get("text", "")}]
        for u in units:
            out.append((u["unit_id"], f"[{head} {u.get('label', '')}] {u['text']}"))
    return out


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
def cmd_extract(args) -> int:
    rows = read_csv_any(Path(args.csv))
    cols = list(rows[0].keys())
    qcol = args.question_col or find_col(cols, QUESTION_COL_HINTS)
    acol = args.answer_col or find_col(cols, ANSWER_COL_HINTS)
    ccol = find_col(cols, CATEGORY_COL_HINTS)
    if not qcol or not acol:
        sys.exit(f"질문/답변 열을 찾지 못했습니다. 열 목록: {cols}\n"
                 f"--question-col / --answer-col 로 직접 지정하세요.")
    print(f"질문 열: {qcol} / 답변 열: {acol} / 구분 열: {ccol or '-'}")

    pat = re.compile("|".join(map(re.escape, KEYWORDS)))
    picked = []
    for i, r in enumerate(rows, start=1):
        text = f"{r.get(qcol, '')} {r.get(acol, '')}"
        hits = sorted(set(pat.findall(text)))
        if hits:
            picked.append((i, r, hits))
    print(f"키워드 일치: {len(picked)}/{len(rows)}건")

    # 근거 조문 후보 제안 — 답변 본문에 법률 용어가 많아 답변까지 함께 매칭한다
    units = load_units(Path(args.corpus))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    mat = vec.fit_transform([t for _, t in units])
    queries = [f"{r.get(qcol, '')} {r.get(acol, '')}" for _, r, _ in picked]
    sims = linear_kernel(vec.transform(queries), mat) if queries else []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "faq_candidates.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["faq_id", "구분", "원 질문", "정답(소비자원 답변)", "매칭 키워드",
                    *TAG_COLUMNS, *[f"후보조문{k}" for k in range(1, args.top_k + 1)]])
        for (idx, r, hits), row_sims in zip(picked, sims):
            top = row_sims.argsort()[::-1][: args.top_k]
            cands = [f"{units[j][0]} | {units[j][1][:80]}" for j in top]
            w.writerow([f"faq_{idx:04d}", r.get(ccol, "") if ccol else "",
                        r.get(qcol, "").strip(), r.get(acol, "").strip(), " ".join(hits),
                        "", "", r.get(qcol, "").strip(), "", "", "", "", *cands])
    print(f"저장: {out}")
    print("후보조문은 TF-IDF 제안일 뿐입니다. contexts 열은 반드시 직접 확인해 채우세요.")
    return 0


# --------------------------------------------------------------------------- #
# cite — 소비자원 답변이 직접 인용한 조문으로 contexts를 만든다 (품목별 FAQ용)
# --------------------------------------------------------------------------- #
LAW_TOKEN = re.compile(
    r"「(?P<law>[^」]{2,60})」"
    r"|(?P<same>동법|같은\s*법)"
    r"|(?P<short>전자상거래법|방문판매법|약관규제법|약관법|소비자기본법|할부거래법|민법|상법)"
    r"|(?P<art>제\s*(?P<a>\d+)\s*조(?:\s*의\s*(?P<b>\d+))?(?:\s*\([^)]{0,40}\))?"
    r"(?P<tail>(?:\s*(?:제\s*\d+\s*[항호]|및|,|ㆍ|·|와|과))*))"
)
SUB_RE = re.compile(r"제\s*(\d+)\s*([항호])")


def law_prefix(name: str) -> str:
    """법령명을 코퍼스 prefix로. 코퍼스 밖이면 'ext:<법령명>'."""
    n = re.sub(r"\s+", "", name).split(",")[0]
    rules = [("전자상거래", "ecom"), ("방문판매", "door"), ("소비자기본법", "cons")]
    for key, pre in rules:
        if key in n:
            if "시행규칙" in n:
                return f"ext:{n}"
            return {"ecom": "enf", "door": "doorenf", "cons": "consenf"}[pre] if "시행령" in n else pre
    if ("약관의규제" in n or "약관규제" in n or n == "약관법") and "표준" not in n:
        return "term"
    if "소비자분쟁해결기준" in n:
        return "std"
    return f"ext:{n}"


# 품목별 FAQ의 '품목명' → 분쟁해결기준 공공데이터의 '품종' (std_kca.json의 title)
ITEM_TO_STD = {
    "소파": "가구", "TV": "가전제품", "냉장고": "가전제품", "룸에어컨": "가전제품",
    "노트북컴퓨터": "사무용기기", "스마트폰": "스마트폰",
    "대형승용자동차": "자동차", "중형승용자동차": "자동차",
    "국내결혼중개": "결혼정보업", "이동전화서비스": "전화서비스",
}


def parse_citations(text: str) -> tuple[list[dict], set[str]]:
    """답변에서 (법령, 조, 가지, 항들, 호들)을 순서대로 뽑는다. '동법'은 직전 법령.
    조문 번호 없이 법령명만 언급된 경우(예: 「소비자분쟁해결기준」에 따르면)도 mentioned에 남긴다."""
    cur, out, mentioned = None, [], set()
    for m in LAW_TOKEN.finditer(text):
        if m.group("law") or m.group("short"):
            cur = law_prefix(m.group("law") or m.group("short"))
            mentioned.add(cur)
        elif m.group("same"):
            pass  # 직전 법령 유지
        elif m.group("art") and cur:
            paras, hos, last_p = [], [], None
            for num, kind in SUB_RE.findall(m.group("tail") or ""):
                if kind == "항":
                    paras.append(int(num)); last_p = int(num)
                else:
                    hos.append((last_p, int(num)))
            out.append({"law": cur, "art": int(m.group("a")),
                        "branch": int(m.group("b")) if m.group("b") else None,
                        "paras": paras, "hos": hos, "raw": m.group(0).strip()})
    return out, mentioned


def resolve_citation(c: dict, units_by_doc: dict[str, list[str]]) -> tuple[list[str], str]:
    """인용 → unit_id. 반환 (units, 정밀도: exact|article|missing)."""
    doc = f"{c['law']}-{c['art']}" + (f"_{c['branch']}" if c["branch"] else "")
    have = units_by_doc.get(doc)
    if not have:
        return [], "missing"
    hits = []
    for p, h in c["hos"]:
        for cand in ([f"{doc}-{p}-{h}"] if p else [f"{doc}-1-{h}", f"{doc}-{h}"]):
            if cand in have:
                hits.append(cand); break
    ho_paras = {p for p, _ in c["hos"]}
    for p in c["paras"]:
        if p not in ho_paras and f"{doc}-{p}" in have:
            hits.append(f"{doc}-{p}")
    return (hits, "exact") if hits else (have, "article")


def cmd_cite(args) -> int:
    rows = read_csv_any(Path(args.csv))
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    units_by_doc, unit_text = {}, {}
    for d in corpus["documents"]:
        us = d.get("units") or [{"unit_id": d["doc_id"], "text": d.get("text", "")}]
        units_by_doc[d["doc_id"]] = [u["unit_id"] for u in us if not u["unit_id"].endswith("-proviso")]
        for u in us:
            unit_text[u["unit_id"]] = u.get("text", "")
    corpus_laws = {k.split("-")[0] for k in units_by_doc}
    std_by_title = {d["title"]: d["doc_id"] for d in corpus["documents"]
                    if d["doc_id"].startswith("std-") and not d.get("handmade")}

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(list(unit_text.values()) or ["x"])

    out_rows, stats = [], {"in_corpus": 0, "needs_ext": 0, "no_cite": 0, "needs_std": 0}
    for i, r in enumerate(rows, start=1):
        q, a = r.get("질문", "").strip(), r.get("답변", "").strip()
        cites, laws = parse_citations(a)
        item = r.get("품목명", "").strip()
        std_doc = std_by_title.get(ITEM_TO_STD.get(item, item)) if "std" in laws else None
        ext = sorted({l[4:] for l in laws if l.startswith("ext:")} |
                     {l for l in laws if not l.startswith("ext:") and l not in corpus_laws and l != "std"} |
                     ({"분쟁해결기준(해당 품종 없음)"} if "std" in laws and not std_doc else set()))
        if not laws:
            scope = "no_cite"
        elif ext:
            scope = "needs_ext"
        else:
            scope = "in_corpus"
        stats[scope] += 1
        if "std" in laws:
            stats["needs_std"] += 1

        exact, guess, missing = [], [], []
        pool: list[str] = []  # 조 단위로만 인용된 조의 모든 항·호 — 검수자가 고를 수 있게 보여 준다
        for c in cites:
            if c["law"].startswith("ext:") or c["law"] == "std":
                continue
            units, how = resolve_citation(c, units_by_doc)
            if how == "exact":
                exact += [u for u in units if u not in exact]
                # '제2항'처럼 항만 인용했는데 그 항이 호로 나뉘어 있으면 실제 내용은 호에 있다 → 호를 후보로 보여 준다
                doc_units = units_by_doc["-".join(units[0].split("-")[:2])]
                for u in units:
                    pool += [c for c in doc_units if c.startswith(u + "-") and c not in pool]
            elif how == "article":
                pool += [u for u in units if u not in pool]
                # 조 단위 인용 → 그 조의 항·호 중 답변과 가장 가까운 것을 '제안'으로만 붙인다
                sims = cosine_similarity(vec.transform([a]), vec.transform([unit_text[u] for u in units]))[0]
                best = units[int(sims.argmax())]
                if best not in exact and best not in guess:
                    guess.append(best)
            else:
                missing.append(c["raw"])

        if std_doc:
            # 분쟁해결기준은 조문 번호 없이 인용되므로 해당 품종 안에서 가장 가까운 기준을 '추정'한다
            us = units_by_doc[std_doc]
            sims = cosine_similarity(vec.transform([q + " " + a]), vec.transform([unit_text[u] for u in us]))[0]
            best = us[int(sims.argmax())]
            if best not in guess:
                guess.append(best)
            pool += [u for u in us if u not in pool]

        main = next((c["law"] for c in cites if not c["law"].startswith("ext:") and c["law"] != "std"), "")
        arts = {c["art"] for c in cites if c["law"] == "door"}
        dtype = ("ecommerce_refund" if main in ("ecom", "enf") else
                 "continuous_contract" if main in ("door", "doorenf") and arts & {29, 30, 31, 32, 33, 34}
                 else "door_sales" if main in ("door", "doorenf")
                 else "product_quality" if std_doc else main or "")
        ctx = exact + guess
        show = lambda ids: "\n".join(f"[{u}] {unit_text.get(u, '(코퍼스에 없음)')}" for u in ids)
        out_rows.append({
            "faq_id": f"kca_{i:04d}", "품목": r.get("품목명", "").strip(), "구분": r.get("구분", "").strip(),
            "scope": scope, "원 질문": q, "정답(소비자원 답변)": a,
            "인용(원문)": " / ".join(f"{c['law'].replace('ext:', '')} {c['raw']}" for c in cites),
            "코퍼스밖 법령": " ".join(ext), "분쟁해결기준 인용": "Y" if "std" in laws else "",
            "contexts_정확": " ".join(exact), "contexts_추정": " ".join(guess),
            "조문 미존재": " ".join(missing),
            "contexts 원문(확인용)": show(ctx),
            "추정 대상 조의 전체 항(고를 때 참고)": show([u for u in pool if u not in ctx]),
            "keep": "", "background": "", "question": q, "contexts": " ".join(ctx),
            "n_hops": str(max(1, len(ctx))) if ctx else "", "dispute_type": dtype, "메모": "",
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(out_rows[0].keys())
    all_path = OUT_DIR / "kca_cite_all.csv"
    with all_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out_rows)

    # 검수용 표본: 코퍼스 안에서 답이 닫히는 문항만, (주 법령, 구분 대분류)로 층화
    import random
    from collections import defaultdict
    rng = random.Random(args.seed)
    pool = defaultdict(list)
    for r in out_rows:
        if r["scope"] == "in_corpus" and r["contexts"]:
            pool[(r["dispute_type"], r["구분"].split("_")[0])].append(r)
    for v in pool.values():
        rng.shuffle(v)
    sample, keys = [], sorted(pool)
    while len(sample) < args.sample and any(pool[k] for k in keys):
        for k in keys:
            if pool[k] and len(sample) < args.sample:
                sample.append(pool[k].pop())
    sample.sort(key=lambda r: r["faq_id"])
    s_path = OUT_DIR / "kca_cite_sample.csv"
    with s_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(sample)

    from collections import Counter
    print(f"전체 {len(out_rows)}: {stats}")
    print("표본 hop 분포:", dict(sorted(Counter(r['n_hops'] for r in sample).items())),
          "/ 유형:", dict(Counter(r['dispute_type'] for r in sample)))
    print(f"저장: {all_path.name} (전체), {s_path.name} (검수용 {len(sample)}문항)")
    print("contexts_정확 = 소비자원이 항·호까지 인용 / contexts_추정 = 조까지만 인용해 항을 추정 → 반드시 확인")
    return 0


# --------------------------------------------------------------------------- #
# proc — 피해구제·분쟁조정 절차 FAQ → 절차 안내 문항 (법조문 근거 없음, 수동 채점)
# --------------------------------------------------------------------------- #
def cmd_proc(args) -> int:
    rows = read_csv_any(Path(args.csv))
    out = OUT_DIR / "questions_proc.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, start=1):
            f.write(json.dumps({
                "id": f"proc_{i:03d}", "source": "faq_proc", "background": "",
                "question": r.get("질문", "").strip(), "answer": r.get("답변", "").strip(),
                "contexts": [], "n_hops": 0, "dispute_type": "procedure",
                "expected_route": {"depth": "guide", "agencies": []},
                "verified_by": "kca", "note": f"구분: {r.get('구분', '').strip()}",
            }, ensure_ascii=False) + "\n")
    print(f"{len(rows)}문항 저장: {out}")
    print("법조문 근거가 없는 절차 문항입니다. 자료에 없는 기관·주소·기간을 지어내는지(환각)를 봅니다.")
    return 0


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def cmd_build(args) -> int:
    rows = read_csv_any(Path(args.tagged))
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT))
    from rag.chunking import corpus_unit_ids

    valid = corpus_unit_ids(corpus)
    out_rows, errors, seen_q = [], [], {}
    for r in rows:
        if r.get("keep", "").strip().upper() != "Y":
            continue
        qkey = re.sub(r"\s+", "", (r.get("question") or r.get("원 질문", "")))
        if qkey in seen_q:  # 소비자원 FAQ에 같은 질문이 다른 번호로 중복 수록된 경우가 있다
            print(f"[중복 제외] {r['faq_id']} = {seen_q[qkey]}")
            continue
        seen_q[qkey] = r["faq_id"]
        ctx = r.get("contexts", "").split()
        bad = [c for c in ctx if c not in valid]
        if bad:
            errors.append(f"{r['faq_id']}: 코퍼스에 없는 unit_id {bad}")
        n_hops = r.get("n_hops", "").strip()
        out_rows.append({
            "id": r["faq_id"],
            "source": "faq",
            "background": r.get("background", "").strip(),
            "question": (r.get("question") or r.get("원 질문", "")).strip(),
            "answer": r.get("정답(소비자원 답변)", "").strip(),
            "contexts": ctx,
            "n_hops": int(n_hops) if n_hops.isdigit() else len(ctx),
            "dispute_type": r.get("dispute_type", "").strip() or "ecommerce_refund",
            "verified_by": "kca",
            "note": r.get("메모", "").strip(),
        })

    for e in errors:
        print("[오류]", e, file=sys.stderr)
    if errors:
        print(f"\n{len(errors)}건 오류 — 수정 후 다시 실행하세요. 파일은 쓰지 않았습니다.",
              file=sys.stderr)
        return 1

    out = OUT_DIR / "questions_faq.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for q in out_rows:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(out_rows)}문항 저장: {out}")
    print("hop 분포:", dict(sorted(Counter(q["n_hops"] for q in out_rows).items())))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("csv")
    e.add_argument("--corpus", default=str(ROOT / "data" / "legal_corpus.json"))
    e.add_argument("--top-k", type=int, default=5)
    e.add_argument("--question-col")
    e.add_argument("--answer-col")

    b = sub.add_parser("build")
    b.add_argument("tagged")
    b.add_argument("--corpus", default=str(ROOT / "data" / "legal_corpus.json"))

    c = sub.add_parser("cite", help="품목별 FAQ: 소비자원 답변의 인용 조문으로 contexts 생성")
    c.add_argument("csv")
    c.add_argument("--corpus", default=str(ROOT / "data" / "legal_corpus.json"))
    c.add_argument("--sample", type=int, default=60)
    c.add_argument("--seed", type=int, default=20261015)

    pr = sub.add_parser("proc", help="피해구제·분쟁조정 절차 FAQ → questions_proc.jsonl")
    pr.add_argument("csv")

    args = ap.parse_args()
    return {"extract": cmd_extract, "cite": cmd_cite, "proc": cmd_proc, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

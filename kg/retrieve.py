#!/usr/bin/env python3
"""법령 지식그래프 검색: 라우팅 → 범위 안 Hybrid 진입검색 → 관계 확장 → 역할별 근거 묶음.

    route()      질문 → 거래 유형 Category (LLM, 최대 2개) — 또는 평가용 oracle 라우팅
    scope_ids()  Category → 조항 범위 (cross Category는 항상 포함)
    entry()      범위 안 fulltext(cjk) + vector, RRF로 합쳐 상위 k개
    expand()     진입 조항에서 관계 종류별 규칙으로 근거를 모아 역할을 붙인다
    format_pack() LLM에 넘길 텍스트

확장 규칙 (hop 수가 아니라 관계 종류로 정한다)
    상위 문맥   (진입)-[:SUB_OF*1..2]->(항·호)
    예외        (x)-[:PROVISO_OF|OVERRIDES]->(진입)      단서, "~에도 불구하고"
    원칙        (진입)-[:OVERRIDES]->(x)                  진입이 예외일 때 그 원칙
    정의        (진입)-[:USES_TERM]->(정의)               쓰는 조항 → 정의 방향만
    시행령      (진입)-[:DELEGATES_TO]->(시행령)
    위임 근거   (법률)-[:DELEGATES_TO]->(진입)            진입이 시행령일 때
    인용        (진입)-[:REFERS_TO]->(x)                  대상이 조 전체면 그 조의 단위 몇 개
    인용됨      (x)-[:REFERS_TO]->(진입)                  예: door-52(불리한 계약 무효)가 door-31을 인용
    해석 지침   ecomg가 진입을 인용 / 산정 기준: contfee가 진입을 인용
    적용 순서   묶음에 법이 둘 이상이면 각 법의 '다른 법률과의 관계' 조항 + 단서, 약관법 제30조제2항

사용
    python kg/retrieve.py "헬스장 1년 계약했는데 3개월 만에 그만두면 위약금은?"
    python kg/retrieve.py --route continuous "…"      # 라우팅 지정
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_graph import DATA, ROOT, connect  # noqa: E402

ROUTE_CACHE = ROOT / "results" / "kg" / "route_cache.json"
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# 법이 둘 이상 섞이면 넣는 경합 규정 (각 법의 '다른 법률과의 관계' + 약관법 적용 범위)
CONFLICT_RULES = {"ecom": ["ecom-4", "ecom-4-proviso"], "door": ["door-4-1", "door-4-1-proviso", "door-4-3"],
                  "inst": ["inst-4", "inst-4-proviso"], "term": ["term-30-2"]}
FAMILY = {"ecom": "ecom", "enf": "ecom", "ecomg": "ecom", "door": "door", "doorenf": "door", "contfee": "door",
          "inst": "inst", "instenf": "inst", "term": "term"}
ROLE_ORDER = ["진입", "예외", "원칙", "정의", "시행령", "위임 근거", "상위 문맥", "해석 지침", "산정 기준",
              "인용", "인용됨", "적용 순서"]
LUCENE_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


@dataclass
class Item:
    unit_id: str
    role: str
    via: str           # 어느 진입 조항에서 어떤 관계로 왔는가 (진입이면 검색 점수)
    path: str = ""
    text: str = ""


@dataclass
class Pack:
    question: str
    categories: list[str]
    entries: list[str]
    items: list[Item] = field(default_factory=list)

    @property
    def unit_ids(self) -> list[str]:
        return [i.unit_id for i in self.items]


# --------------------------------------------------------------------------- #
# ① 라우팅
# --------------------------------------------------------------------------- #
def categories(driver) -> list[dict]:
    recs, _, _ = driver.execute_query(
        "MATCH (c:Category) RETURN c.category_id AS id, c.name AS name, c.kind AS kind ORDER BY id")
    return [dict(r) for r in recs]


ANALYZE_PROMPT = """소비자 상담 질문을 분석하세요.

[거래 유형]
{options}

1) categories: 질문의 거래에 해당하는 유형 1~2개. 확실한 하나가 있으면 하나만.
   - 인터넷·앱·홈쇼핑 구매는 ecommerce, 집·길거리에서 권유받아 산 경우는 door_sales,
     헬스·필라테스·학습지처럼 2개월 이상 계속 공급받는 계약은 continuous, 상조·선불식은 prepaid.
   - 제품 하자·수리·교환·보상 기준을 묻거나 판매 방식이 중요하지 않으면 dispute_standard를 포함.
   - 약관(계약서 조항)의 효력만 묻는 경우 []도 됩니다(약관 조항은 항상 함께 검색됨).
2) keywords: 법령 조문에 실제로 쓰일 법률 용어 3~8개 (예: 청약철회, 계약의 해지, 위약금, 손해배상액의 예정,
   원상회복, 대금 환급, 약관 무효, 재화등의 훼손). 조문 번호·법 이름은 쓰지 마세요.
3) issue: 법적 쟁점 한 문장.

JSON으로만 답하세요: {{"categories": [...], "keywords": [...], "issue": "..."}}

[질문]
{question}"""


def analyze(driver, question: str, use_cache: bool = True) -> dict:
    """라우팅 + 질의 재작성(법률 용어·쟁점)을 LLM 한 번으로. 결과는 캐시(같은 질문은 비용 0)."""
    cats = [c for c in categories(driver) if c["kind"] != "cross"]
    ids = {c["id"] for c in cats}
    key = hashlib.sha1(f"analyze|{CHAT_MODEL}|{question}".encode()).hexdigest()
    cache = json.loads(ROUTE_CACHE.read_text(encoding="utf-8")) if ROUTE_CACHE.exists() else {}
    if use_cache and key in cache:
        return cache[key]
    from rag.backends import _openai_client
    options = "\n".join(f"- {c['id']}: {c['name']}" for c in cats)
    resp = _openai_client().chat.completions.create(
        model=CHAT_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": ANALYZE_PROMPT.format(options=options, question=question)}])
    try:
        got = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        got = {}
    out = {"categories": [g for g in got.get("categories", []) if g in ids][:2],
           "keywords": [str(k) for k in got.get("keywords", [])][:8],
           "issue": str(got.get("issue", ""))}
    cache[key] = out
    ROUTE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ROUTE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def route(driver, question: str) -> list[str]:
    return analyze(driver, question)["categories"]


# --------------------------------------------------------------------------- #
# ② 범위 + 진입 검색
# --------------------------------------------------------------------------- #
def scope_ids(driver, cats: list[str] | None) -> set[str] | None:
    """None이면 범위 제한 없음(전체)."""
    if cats is None:
        return None
    recs, _, _ = driver.execute_query(
        """MATCH (c:Category)-[:APPLIES_TO]->(:Article)-[:HAS_UNIT]->(u:Unit)
           WHERE c.category_id IN $cats OR c.kind = 'cross' RETURN DISTINCT u.unit_id AS id""", {"cats": cats})
    return {r["id"] for r in recs}


def embed_query(text: str) -> list[float]:
    from rag.backends import _openai_client
    return _openai_client().embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


def entry(driver, question: str, scope: set[str] | None, k: int = 8, pool: int = 400,
          vec: list[float] | None = None, mode: str = "hybrid") -> list[tuple[str, float]]:
    """범위 안에서 fulltext·vector 순위를 RRF(1/(60+rank))로 합친다. mode: hybrid|vector|fulltext."""
    allowed = list(scope) if scope is not None else None
    ranks: dict[str, float] = {}
    if mode in ("hybrid", "fulltext"):
        q = LUCENE_SPECIAL.sub(" ", question).strip() or "소비자"
        recs, _, _ = driver.execute_query(
            """CALL db.index.fulltext.queryNodes('unit_text', $q, {limit: $pool}) YIELD node, score
               WHERE $allowed IS NULL OR node.unit_id IN $allowed
               RETURN node.unit_id AS id ORDER BY score DESC""", {"q": q, "pool": pool, "allowed": allowed})
        for r, rec in enumerate(recs):
            ranks[rec["id"]] = ranks.get(rec["id"], 0) + 1 / (60 + r)
    if mode in ("hybrid", "vector"):
        vec = vec or embed_query(question)
        recs, _, _ = driver.execute_query(
            """CALL db.index.vector.queryNodes('unit_embedding', $pool, $v) YIELD node, score
               WHERE $allowed IS NULL OR node.unit_id IN $allowed
               RETURN node.unit_id AS id ORDER BY score DESC""", {"v": vec, "pool": pool, "allowed": allowed})
        for r, rec in enumerate(recs):
            ranks[rec["id"]] = ranks.get(rec["id"], 0) + 1 / (60 + r)
    return sorted(ranks.items(), key=lambda x: -x[1])[:k]


def article_entries(driver, ranked: list[tuple[str, float]], m: int = 5, per_article: int = 4,
                    m_rule: int = 2) -> list[tuple[str, float]]:
    """단위 순위를 조(Article) 순위로 묶는다: 조 점수 = 그 조 단위들의 최고 점수.
    상위 조마다 [가장 높은 단위 + 그 조의 항 단위(본문·항, 단서 제외) 몇 개]를 진입으로 쓴다.
    호 하나만 맞고 그 호가 속한 항(원칙)은 못 찾는 문제를 줄인다.
    법령(법률·시행령) 조 m개와 고시·지침·분쟁해결기준 조 m_rule개를 따로 뽑는다 — 생활 언어로 쓰인 고시가
    상위를 차지해 핵심 법률 조항(예: door-31 계약 해지)을 밀어내지 않게."""
    best: dict[str, tuple[str, float]] = {}
    recs, _, _ = driver.execute_query(
        """MATCH (l:Law)-[:HAS_ARTICLE]->(:Article)-[:HAS_UNIT]->(u:Unit) WHERE u.unit_id IN $ids
           RETURN u.unit_id AS id, u.doc_id AS doc, l.level AS level""",
        {"ids": [u for u, _ in ranked]})
    doc_of = {r["id"]: r["doc"] for r in recs}
    level = {r["doc"]: r["level"] for r in recs}
    for uid, s in ranked:
        doc = doc_of.get(uid)
        if doc and (doc not in best or s > best[doc][1]):
            best[doc] = (uid, s)
    ordered = sorted(best.items(), key=lambda x: -x[1][1])
    statute = [x for x in ordered if level.get(x[0], 9) <= 2]
    rules = [x for x in ordered if level.get(x[0], 9) > 2]
    top = statute[:m] + rules[:m_rule]
    if len(top) < m + m_rule:  # 한쪽이 모자라면 다른 쪽으로 채운다 (예: 분쟁해결기준만 있는 범위)
        top += [x for x in ordered if x not in top][:m + m_rule - len(top)]
    recs, _, _ = driver.execute_query(
        """MATCH (a:Article)-[:HAS_UNIT]->(u:Unit) WHERE a.doc_id IN $docs
             AND u.kind IN ['article', 'body', 'paragraph', 'criterion', 'annex']
           RETURN a.doc_id AS doc, u.unit_id AS id ORDER BY id""", {"docs": [d for d, _ in top]})
    paras: dict[str, list[str]] = {}
    for r in recs:
        paras.setdefault(r["doc"], []).append(r["id"])
    out: list[tuple[str, float]] = []
    for doc, (uid, s) in top:
        out.append((uid, s))
        out += [(p, s) for p in paras.get(doc, []) if p != uid][:per_article - 1]
    seen, uniq = set(), []
    for u, s in out:
        if u not in seen:
            seen.add(u)
            uniq.append((u, s))
    return uniq


# --------------------------------------------------------------------------- #
# ③ 확장
# --------------------------------------------------------------------------- #
EXPAND = """
MATCH (u:Unit {unit_id: $id})
CALL (u) { OPTIONAL MATCH (u)-[:SUB_OF*1..2]->(p:Unit) RETURN collect(DISTINCT p.unit_id) AS parents }
CALL (u) { OPTIONAL MATCH (x:Unit)-[:PROVISO_OF|OVERRIDES]->(u) RETURN collect(DISTINCT x.unit_id) AS exceptions }
CALL (u) { OPTIONAL MATCH (u)-[:SUB_OF]->(:Unit)<-[:PROVISO_OF]-(x:Unit) RETURN collect(DISTINCT x.unit_id) AS parent_exc }
CALL (u) { OPTIONAL MATCH (u)-[:OVERRIDES]->(x:Unit) RETURN collect(DISTINCT x.unit_id) AS principles }
CALL (u) { OPTIONAL MATCH (u)-[:USES_TERM]->(x:Unit) RETURN collect(DISTINCT x.unit_id) AS defs }
CALL (u) { OPTIONAL MATCH (u)-[:DELEGATES_TO]->(x:Unit) RETURN collect(DISTINCT x.unit_id) AS enf }
CALL (u) { OPTIONAL MATCH (x:Unit)-[:DELEGATES_TO]->(u) RETURN collect(DISTINCT x.unit_id) AS basis }
CALL (u) { OPTIONAL MATCH (u)-[:REFERS_TO]->(x:Unit) RETURN collect(DISTINCT x.unit_id) AS refs }
CALL (u) { OPTIONAL MATCH (u)-[:REFERS_TO]->(:Article)-[:HAS_UNIT]->(x:Unit)
           WITH x ORDER BY x.unit_id RETURN collect(DISTINCT x.unit_id)[..4] AS ref_art }
CALL (u) { OPTIONAL MATCH (x:Unit)-[:REFERS_TO]->(u) RETURN collect(DISTINCT x.unit_id) AS cited_by }
CALL (u) { OPTIONAL MATCH (x:Unit)-[:REFERS_TO]->(:Article)-[:HAS_UNIT]->(u) RETURN collect(DISTINCT x.unit_id) AS cited_by_art }
RETURN parents, exceptions + parent_exc AS exceptions, principles, defs, enf, basis, refs + ref_art AS refs,
       cited_by + cited_by_art AS cited_by
"""


def role_of_citer(uid: str) -> str:
    law = uid.split("-")[0]
    return {"ecomg": "해석 지침", "contfee": "산정 기준"}.get(law, "인용됨")


def expand(driver, entries: list[tuple[str, float]], budget: int = 40, cited_by_cap: int = 3) -> list[Item]:
    items: dict[str, Item] = {}

    def add(uid: str, role: str, via: str) -> None:
        if uid not in items:
            items[uid] = Item(uid, role, via)

    for uid, score in entries:
        add(uid, "진입", f"검색 {score:.4f}")
    for uid, _ in entries:
        r = driver.execute_query(EXPAND, {"id": uid})[0][0]
        for x in r["exceptions"]:
            add(x, "예외", f"{uid}의 단서·예외")
        for x in r["principles"]:
            add(x, "원칙", f"{uid}가 예외로 두는 원칙")
        for x in r["defs"]:
            add(x, "정의", f"{uid}의 용어")
        for x in r["enf"]:
            add(x, "시행령", f"{uid}의 위임")
        for x in r["basis"]:
            add(x, "위임 근거", f"{uid}에 위임한 법률")
        for x in r["parents"]:
            add(x, "상위 문맥", f"{uid}의 상위")
        cited = sorted(set(r["cited_by"]), key=lambda x: (role_of_citer(x) == "인용됨", x))
        n = 0
        for x in cited:
            role = role_of_citer(x)
            if role == "인용됨":
                if n >= cited_by_cap:
                    continue
                n += 1
            add(x, role, f"{uid}를 인용")
        for x in r["refs"]:
            add(x, "인용", f"{uid}가 인용")

    # 적용 순서: 묶음에 법 계열이 둘 이상이면 경합 규정
    fams = {FAMILY.get(u.split("-")[0]) for u in items} - {None}
    if len(fams) >= 2:
        for f in sorted(fams):
            for x in CONFLICT_RULES.get(f, []):
                add(x, "적용 순서", f"{'·'.join(sorted(fams))} 경합")

    ordered = sorted(items.values(), key=lambda i: ROLE_ORDER.index(i.role))
    keep = [i for i in ordered if i.role in ("진입", "적용 순서")]
    rest = [i for i in ordered if i.role not in ("진입", "적용 순서")]
    return sorted(keep + rest[:max(0, budget - len(keep))], key=lambda i: ROLE_ORDER.index(i.role))


def fill_text(driver, items: list[Item]) -> None:
    recs, _, _ = driver.execute_query(
        "MATCH (u:Unit) WHERE u.unit_id IN $ids RETURN u.unit_id AS id, u.path AS path, u.text AS text",
        {"ids": [i.unit_id for i in items]})
    got = {r["id"]: r for r in recs}
    for i in items:
        if i.unit_id in got:
            i.path, i.text = got[i.unit_id]["path"], got[i.unit_id]["text"]


def retrieve(driver, question: str, cats: list[str] | None, k: int = 8, budget: int = 40,
             expand_on: bool = True, vec: list[float] | None = None, rewrite: dict | None = None,
             by_article: bool = False) -> Pack:
    """rewrite: analyze() 결과. 주면 fulltext는 법률 용어로, vector는 질문+쟁점+용어로 검색한다.
    by_article: 단위 순위를 조 단위로 묶어 진입 (article_entries)."""
    ft_query = question
    if rewrite and rewrite.get("keywords"):
        ft_query = " ".join(rewrite["keywords"])
        vec = embed_query(f"{question}\n쟁점: {rewrite.get('issue', '')}\n용어: {', '.join(rewrite['keywords'])}")
    scope = scope_ids(driver, cats)
    if by_article:
        hits = article_entries(driver, entry(driver, ft_query, scope, k=60, vec=vec), m=max(3, k // 2), m_rule=2)
    else:
        hits = entry(driver, ft_query, scope, k=k, vec=vec)
    items = expand(driver, hits, budget=budget) if expand_on else [Item(u, "진입", f"검색 {s:.4f}") for u, s in hits]
    fill_text(driver, items)
    return Pack(question, cats or [], [u for u, _ in hits], items)


def format_pack(pack: Pack) -> str:
    """LLM 입력. 역할 순서대로, 조항 ID를 앞에 붙인다."""
    out = []
    for i in pack.items:
        out.append(f"[{i.role}] {i.unit_id} | {i.path}\n{i.text}")
    return "\n\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--route", help="쉼표로 Category 지정 (생략하면 LLM 라우팅, 'none'이면 전체)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--no-rewrite", action="store_true", help="법률 용어 재작성 없이 질문 그대로 검색")
    ap.add_argument("--unit-entry", action="store_true", help="조 단위 묶음 없이 단위 순위 그대로 진입")
    args = ap.parse_args()
    d = connect()
    try:
        info = analyze(d, args.question)
        cats = (None if args.route == "none" else args.route.split(",")) if args.route else info["categories"]
        pack = retrieve(d, args.question, cats, k=args.k, budget=args.budget,
                        rewrite=None if args.no_rewrite else info, by_article=not args.unit_entry)
    finally:
        d.close()
    print(f"라우팅: {pack.categories or '전체'} / 용어: {info['keywords']} / 쟁점: {info['issue']}")
    print(f"진입: {pack.entries}\n")
    for i in pack.items:
        print(f"[{i.role}] {i.unit_id}  ({i.via})  {i.path}\n    {i.text[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

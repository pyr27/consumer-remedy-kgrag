#!/usr/bin/env python3
"""합격 기준 점검: FAQ 정답 근거가 그래프에서 얼마나 가깝게 이어지는가.

1) 경로 연결 — 정답 근거 쌍의 shortestPath ≤ 4 hop 비율
    cite      Law·Category 노드를 거치지 않는 경로 (인용·위임·예외·용어 정의·구조 관계만)
    category  Law만 제외 (Category 허용)
    Law·Category는 묶음 노드라서, 허용하면 같은 묶음 안의 아무 두 조항이나 4 hop(Unit→Article→묶음←Article←Unit)이 된다.
    그래서 같은 그래프에서 '정답과 무관한 무작위 쌍'의 통과율(대조군)을 함께 낸다. 정답 쌍 통과율이 대조군보다
    충분히 높아야 연결이 의미 있다.

2) 범위 포함 (oracle 라우팅) — 질문의 거래 유형(FAQ dispute_type)을 정확히 맞혔다고 가정할 때,
    그 Category(+ 모든 거래에 걸치는 cross Category)가 가리키는 조항 범위에 정답 근거가 몇 % 들어오는가, 범위는 몇 조항인가.
    +1hop: 범위 조항에서 의미 관계(인용·위임·예외·용어) 한 단계로 닿는 조항까지 포함.

사용
    python kg/check_paths.py                  # 요약 + cite 기준 실패 쌍
    python kg/check_paths.py --all            # 모든 쌍 출력
    python kg/check_paths.py --out results/kg/paths.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_graph import DATA, SEMANTIC, connect  # noqa: E402

EXCLUDE: set[str] = set()  # kca_0542는 정답 근거를 -20으로 정정해 포함 (2026-10-21 저자 확인)
MAX_HOPS = 4
SEARCH_HOPS = 10  # 실패한 쌍도 실제 거리를 보여 주기 위해 더 멀리까지 찾는다
N_CONTROL = 200
MODES = {"cite": ["Law", "Category"], "category": ["Law"]}
# FAQ dispute_type → 거래 유형 Category (oracle 라우팅). product_quality는 품목 라우팅이 필요해 제외.
ROUTE = {"continuous_contract": ["continuous"], "ecommerce_refund": ["ecommerce"],
         "door_sales": ["door_sales"], "term": []}


def pairs(questions: list[dict]) -> list[tuple[str, str, str]]:
    out = []
    for q in questions:
        ctx = list(dict.fromkeys(q["contexts"]))  # 중복 제거, 순서 유지
        out += [(q["id"], a, b) for a, b in itertools.combinations(ctx, 2)]
    return out


def shortest(driver, a: str, b: str, banned: list[str]) -> tuple[int | None, str]:
    where = ("WHERE none(n IN nodes(p) WHERE " + " OR ".join(f"n:{l}" for l in banned) + ")") if banned else ""
    recs, _, _ = driver.execute_query(
        f"""MATCH (a:Unit {{unit_id: $a}}), (b:Unit {{unit_id: $b}})
            MATCH p = shortestPath((a)-[*..{SEARCH_HOPS}]-(b)) {where}
            RETURN length(p) AS hops,
                   [n IN nodes(p) | coalesce(n.unit_id, n.doc_id, n.category_id, n.law_id)] AS ns,
                   [r IN relationships(p) | type(r)] AS rs""",
        {"a": a, "b": b})
    if not recs:
        return None, ""
    r = recs[0]
    route = r["ns"][0] + "".join(f" -{t}- {n}" for t, n in zip(r["rs"], r["ns"][1:]))
    return r["hops"], route


def reach(driver, uid: str, banned: list[str]) -> int:
    recs, _, _ = driver.execute_query(
        f"""MATCH (a:Unit {{unit_id: $u}})
            MATCH p = (a)-[*1..{MAX_HOPS}]-(b:Unit)
            WHERE none(n IN nodes(p) WHERE {" OR ".join(f"n:{l}" for l in banned)})
            RETURN count(DISTINCT b) AS n""", {"u": uid})
    return recs[0]["n"]


def scope(driver, cats: list[str], hop: bool) -> set[str]:
    rels = "|".join(SEMANTIC)
    q = """MATCH (c:Category)-[:APPLIES_TO]->(:Article)-[:HAS_UNIT]->(u:Unit)
           WHERE c.category_id IN $cats OR c.kind = 'cross' RETURN DISTINCT u.unit_id AS id"""
    if hop:
        # USES_TERM은 '쓰는 조항 → 정의' 방향으로만 따라간다. 거꾸로 가면 정의 하나가 그 용어를 쓰는 조항 수십 개를 끌어온다.
        both = "|".join(r for r in SEMANTIC if r != "USES_TERM")
        q = f"""MATCH (c:Category)-[:APPLIES_TO]->(:Article)-[:HAS_UNIT]->(u:Unit)
                WHERE c.category_id IN $cats OR c.kind = 'cross'
                OPTIONAL MATCH (u)-[:{both}]-(v:Unit)
                OPTIONAL MATCH (u)-[:{both}]-(:Article)-[:HAS_UNIT]->(w:Unit)
                OPTIONAL MATCH (u)-[:USES_TERM]->(t:Unit)
                WITH collect(DISTINCT u.unit_id) + collect(DISTINCT v.unit_id) + collect(DISTINCT w.unit_id)
                     + collect(DISTINCT t.unit_id) AS ids
                UNWIND ids AS id RETURN DISTINCT id"""
    return {r["id"] for r in driver.execute_query(q, {"cats": cats})[0] if r["id"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faq", default=str(DATA / "v2" / "questions_faq.jsonl"))
    ap.add_argument("--all", action="store_true", help="통과한 쌍도 출력")
    ap.add_argument("--out", help="쌍별 결과 CSV 경로")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qs = [q for q in (json.loads(l) for l in Path(args.faq).open(encoding="utf-8")) if q["id"] not in EXCLUDE]
    ps = pairs(qs)
    gold = {frozenset((a, b)) for _, a, b in ps}
    driver = connect()
    rows, control = [], {}
    try:
        for qid, a, b in ps:
            row = {"qid": qid, "a": a, "b": b}
            for mode, banned in MODES.items():
                row[mode], row[f"{mode}_route"] = shortest(driver, a, b, banned)
            row["loose"], _ = shortest(driver, a, b, [])
            rows.append(row)

        # 대조군: 정답 근거가 속한 법령들의 조항에서 뽑은 무작위 쌍 (정답 쌍 제외)
        laws = sorted({u.split("-")[0] for _, a, b in ps for u in (a, b)})
        pool = [r["id"] for r in driver.execute_query(
            "MATCH (u:Unit) WHERE u.law_id IN $laws RETURN u.unit_id AS id ORDER BY id", {"laws": laws})[0]]
        rng = random.Random(args.seed)
        ctrl = []
        while len(ctrl) < N_CONTROL:
            a, b = rng.sample(pool, 2)
            if frozenset((a, b)) not in gold:
                ctrl.append((a, b))
        for mode, banned in MODES.items():
            control[mode] = sum(1 for a, b in ctrl if (h := shortest(driver, a, b, banned)[0]) is not None
                                and h <= MAX_HOPS)

        units = sorted({u for _, a, b in ps for u in (a, b)})
        reach_n = {m: {u: reach(driver, u, banned) for u in units} for m, banned in MODES.items()}
        n_units = driver.execute_query("MATCH (u:Unit) RETURN count(u) AS n")[0][0]["n"]

        routed = [q for q in qs if q["dispute_type"] in ROUTE]
        scopes = {}
        for hop in (False, True):
            for dt, cats in ROUTE.items():
                scopes[(dt, hop)] = scope(driver, cats, hop)
    finally:
        driver.close()

    ok = lambda h: h is not None and h <= MAX_HOPS
    qids = sorted({r["qid"] for r in rows})
    print(f"[1] 경로 연결 — 다중 근거 {len(qids)}문항 / 근거 쌍 {len(rows)}개 (제외: {', '.join(sorted(EXCLUDE))}), "
          f"대조군 무작위 쌍 {N_CONTROL}개 ({'·'.join(laws)} 조항)")
    print(f"  {'기준':9} {'정답 쌍':>10} {'문항':>7} {'대조군':>9}  4hop 도달 Unit 중앙값/최대 (전체 {n_units})")
    for mode in MODES:
        q_ok = sum(all(ok(r[mode]) for r in rows if r["qid"] == q) for q in qids)
        n_ok = sum(ok(r[mode]) for r in rows)
        vals = list(reach_n[mode].values())
        print(f"  {mode:9} {n_ok:>4}/{len(rows)} ({n_ok / len(rows):4.0%}) {q_ok:>3}/{len(qids)} "
              f"{control[mode]:>4} ({control[mode] / N_CONTROL:4.0%})  {statistics.median(vals):.0f}/{max(vals)}")
    print(f"  loose    {sum(ok(r['loose']) for r in rows):>4}/{len(rows)}  (Law 허브 포함, 참고용)")

    print(f"\n[2] 범위 포함 (oracle 라우팅) — {len(routed)}문항, product_quality 제외")
    for hop in (False, True):
        hit = tot = 0
        q_full = 0
        sizes = []
        for q in routed:
            s = scopes[(q["dispute_type"], hop)]
            ev = set(q["contexts"])
            hit += len(ev & s)
            tot += len(ev)
            q_full += ev <= s
            sizes.append(len(s))
        print(f"  {'+1hop' if hop else '범위만':6} 근거 {hit}/{tot} ({hit / tot:.0%}), 근거 전부 포함 문항 {q_full}/{len(routed)}, "
              f"범위 크기 중앙값 {statistics.median(sizes):.0f} Unit")
    for dt in ROUTE:
        print(f"    {dt:20} 범위 {len(scopes[(dt, False)]):>4} → +1hop {len(scopes[(dt, True)]):>4} Unit")
    miss = sorted({c for q in routed for c in q["contexts"] if c not in scopes[(q["dispute_type"], True)]})
    print(f"  +1hop으로도 범위 밖인 근거: {miss}")

    print("\n[쌍별]" if args.all else "\n[cite 기준 실패 쌍]")
    for r in rows:
        if args.all or not ok(r["cite"]):
            h = "∞" if r["cite"] is None else r["cite"]
            print(f"  {r['qid']}  {r['a']} ↔ {r['b']}  cite={h} category={r['category']}  "
                  f"{r['cite_route'] or r['category_route']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

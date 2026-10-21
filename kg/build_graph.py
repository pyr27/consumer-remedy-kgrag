#!/usr/bin/env python3
"""법령 코퍼스(legal_corpus.json) → Neo4j 지식그래프 (1단계: 구조 뼈대).

원칙: 그래프는 코퍼스 파일에서 언제든 다시 만들 수 있는 '파생물'이다.
      정답 원본은 legal_corpus.json(+ 검토 승인된 관계 CSV)이고, DB는 지워도 된다.

노드
    (:Law     {law_id, name, level})          level 1=법률, 2=시행령, 3=고시
    (:Article {doc_id, law_id, article, title})
    (:Unit    {unit_id, doc_id, law_id, kind, label, text, path, embedding?})
              kind: article(항 없는 조) | body(조 본문) | paragraph(항) | item(호) | subitem(목)
                    | proviso(단서) | criterion(분쟁해결기준 항목) | annex(고시 별표)
관계
    (Law)-[:HAS_ARTICLE]->(Article)-[:HAS_UNIT]->(Unit)
    (Unit 호)-[:SUB_OF]->(Unit 항),  (Unit 목)-[:SUB_OF]->(Unit 호)
    (Unit 단서)-[:PROVISO_OF]->(Unit 본문)
    조문 간 의미 관계(REFERS_TO, DELEGATES_TO, OVERRIDES, USES_TERM): extract_edges.py → data/edges.csv(검토) → approved=y만 적재
    (:Category {category_id, name, kind})-[:APPLIES_TO]->(Article)   data/categories.csv (법령 장 구조 기준, approved=y만)

사용
    python kg/build_graph.py --dry-run             # DB 없이 노드·관계 수만 확인
    python kg/build_graph.py --wipe                # 전체 삭제 후 적재 (임베딩 없이)
    python kg/build_graph.py --wipe --embed        # + OpenAI 임베딩 (캐시 사용, 재실행 시 비용 0)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "embedding_cache.jsonl"

LAW_LEVEL = {"ecom": 1, "term": 1, "cons": 1, "door": 1, "inst": 1,
             "enf": 2, "consenf": 2, "doorenf": 2, "instenf": 2,
             "std": 3, "stdg": 3, "contfee": 3, "ecomg": 3}


# --------------------------------------------------------------------------- #
# 코퍼스 → 노드·관계 (DB와 무관한 순수 변환. 테스트 가능하게 분리)
# --------------------------------------------------------------------------- #
def unit_kind(unit_id: str, doc_id: str, law_id: str) -> str:
    if law_id == "std":
        return "criterion"
    if re.match(r"^[a-z]+-b\d", doc_id):  # 고시 별표 (fetch_admrul.py)
        return "annex"
    if unit_id.endswith("-proviso"):
        return "proviso"
    if unit_id == doc_id:
        return "article"
    rest = unit_id[len(doc_id) + 1:].split("-")
    if rest == ["0"]:
        return "body"
    return {1: "paragraph", 2: "item", 3: "subitem"}.get(len(rest), "subitem")


def header(doc: dict) -> str:
    art = f"{doc.get('article', '')}({doc.get('title', '')})" if doc.get("title") else doc.get("article", "")
    return f"{doc.get('law', '')} {art}".strip()


def build_rows(corpus: dict) -> dict[str, list[dict]]:
    laws, articles, units, sub_of, proviso_of = {}, [], [], [], []
    for d in corpus["documents"]:
        if d.get("handmade"):  # v1용 수작업 요약문은 그래프에 넣지 않는다
            continue
        doc_id = d["doc_id"]
        law_id = doc_id.split("-")[0]
        laws.setdefault(law_id, {"law_id": law_id, "name": d.get("law", law_id),
                                 "level": LAW_LEVEL.get(law_id, 9)})
        articles.append({"doc_id": doc_id, "law_id": law_id, "article": d.get("article", ""),
                         "title": d.get("title", "")})
        raw = d.get("units") or [{"unit_id": doc_id, "label": "", "text": d.get("text", "")}]
        ids = {u["unit_id"] for u in raw}
        for u in raw:
            uid = u["unit_id"]
            kind = unit_kind(uid, doc_id, law_id)
            units.append({"unit_id": uid, "doc_id": doc_id, "law_id": law_id, "kind": kind,
                          "label": u.get("label", ""), "text": u["text"],
                          "path": f"{header(d)} {u.get('label', '')}".strip()})
            if kind == "proviso":
                main = uid[: -len("-proviso")]
                if main in ids:
                    proviso_of.append({"child": uid, "parent": main})
            elif kind in ("item", "subitem"):
                parent = uid.rsplit("-", 1)[0]
                if parent in ids:
                    sub_of.append({"child": uid, "parent": parent})
    return {"laws": list(laws.values()), "articles": articles, "units": units,
            "sub_of": sub_of, "proviso_of": proviso_of}


def embed_text(u: dict) -> str:
    """임베딩 입력: 법령·조 제목을 앞에 붙여 짧은 호·목도 맥락을 갖게 한다."""
    return f"{u['path']} {u['text']}".strip()


# --------------------------------------------------------------------------- #
# 임베딩 (텍스트 해시 캐시 — 코퍼스가 바뀐 조항만 새로 계산)
# --------------------------------------------------------------------------- #
def load_cache() -> dict[str, list[float]]:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.open(encoding="utf-8"):
        r = json.loads(line)
        out[r["key"]] = r["vec"]
    return out


def attach_embeddings(units: list[dict], model: str) -> int:
    cache = load_cache()
    keys = [hashlib.sha1(f"{model}|{embed_text(u)}".encode()).hexdigest() for u in units]
    todo = [(k, u) for k, u in zip(keys, units) if k not in cache]
    if todo:
        sys.path.insert(0, str(ROOT))
        from rag.backends import _openai_client
        client = _openai_client()
        with CACHE.open("a", encoding="utf-8") as f:
            for i in range(0, len(todo), 64):
                batch = todo[i:i + 64]
                resp = client.embeddings.create(model=model, input=[embed_text(u) for _, u in batch])
                for (k, _), d in zip(batch, resp.data):
                    cache[k] = d.embedding
                    f.write(json.dumps({"key": k, "vec": d.embedding}) + "\n")
                print(f"  임베딩 {min(i + 64, len(todo))}/{len(todo)}")
    for k, u in zip(keys, units):
        u["embedding"] = cache[k]
    return len(todo)


# --------------------------------------------------------------------------- #
# Neo4j 적재
# --------------------------------------------------------------------------- #
SCHEMA = [
    "CREATE CONSTRAINT law_id IF NOT EXISTS FOR (n:Law) REQUIRE n.law_id IS UNIQUE",
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (n:Article) REQUIRE n.doc_id IS UNIQUE",
    "CREATE CONSTRAINT unit_id IF NOT EXISTS FOR (n:Unit) REQUIRE n.unit_id IS UNIQUE",
    # cjk 분석기: 한글을 2글자 단위로 잘라 색인 → 형태소 분석기 없이도 부분 일치 검색 가능
    "CREATE FULLTEXT INDEX unit_text IF NOT EXISTS FOR (n:Unit) ON EACH [n.text, n.path] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}}",
]
VECTOR_INDEX = (
    "CREATE VECTOR INDEX unit_embedding IF NOT EXISTS FOR (n:Unit) ON (n.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}"
)
LOAD = {
    "laws": "UNWIND $rows AS r MERGE (l:Law {law_id: r.law_id}) SET l.name = r.name, l.level = r.level",
    "articles": """UNWIND $rows AS r
        MERGE (a:Article {doc_id: r.doc_id}) SET a.law_id = r.law_id, a.article = r.article, a.title = r.title
        WITH a, r MATCH (l:Law {law_id: r.law_id}) MERGE (l)-[:HAS_ARTICLE]->(a)""",
    "units": """UNWIND $rows AS r
        MERGE (u:Unit {unit_id: r.unit_id})
        SET u.doc_id = r.doc_id, u.law_id = r.law_id, u.kind = r.kind, u.label = r.label,
            u.text = r.text, u.path = r.path, u.embedding = r.embedding
        WITH u, r MATCH (a:Article {doc_id: r.doc_id}) MERGE (a)-[:HAS_UNIT]->(u)""",
    "sub_of": """UNWIND $rows AS r MATCH (c:Unit {unit_id: r.child}), (p:Unit {unit_id: r.parent})
        MERGE (c)-[:SUB_OF]->(p)""",
    "proviso_of": """UNWIND $rows AS r MATCH (c:Unit {unit_id: r.child}), (p:Unit {unit_id: r.parent})
        MERGE (c)-[:PROVISO_OF]->(p)""",
}


KEY = {"Unit": "unit_id", "Article": "doc_id"}
SEMANTIC = ("REFERS_TO", "DELEGATES_TO", "OVERRIDES", "USES_TERM")


def category_rows(path: Path, articles: list[dict]) -> tuple[list[dict], list[dict]]:
    """data/categories.csv(approved=y) → Category 노드, (Category)-[:APPLIES_TO]->(Article) 쌍.
    articles 열: '5-22'(조 번호 범위, 가지조 포함) 또는 '*'(그 법령·고시 전체)."""
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["approved"].strip().lower() == "y"]
    cats, links = {}, []
    for r in rows:
        cats.setdefault(r["category_id"], {"category_id": r["category_id"], "name": r["name"], "kind": r["kind"]})
        lo, hi = (None, None) if r["articles"] == "*" else map(int, r["articles"].split("-"))
        for a in articles:
            if a["law_id"] != r["law_id"]:
                continue
            num = a["doc_id"].split("-", 1)[1].split("_")[0]
            if r["articles"] == "*" or (num.isdigit() and lo <= int(num) <= hi):
                links.append({"category_id": r["category_id"], "doc_id": a["doc_id"]})
    return list(cats.values()), links


def load_categories(driver, cats: list[dict], links: list[dict]) -> None:
    driver.execute_query("CREATE CONSTRAINT category_id IF NOT EXISTS FOR (n:Category) REQUIRE n.category_id IS UNIQUE")
    driver.execute_query("UNWIND $rows AS r MERGE (c:Category {category_id: r.category_id}) "
                         "SET c.name = r.name, c.kind = r.kind", {"rows": cats})
    driver.execute_query("UNWIND $rows AS r MATCH (c:Category {category_id: r.category_id}), "
                         "(a:Article {doc_id: r.doc_id}) MERGE (c)-[:APPLIES_TO]->(a)", {"rows": links})
    print(f"  category   {len(cats)}  APPLIES_TO {len(links)}")


def read_edges(path: Path) -> list[dict]:
    """extract_edges.py가 만든 검토 파일에서 승인(approved=y)된 관계만."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r["approved"].strip().lower() == "y"]


def load_edges(driver, edges: list[dict]) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for e in edges:
        assert e["rel"] in SEMANTIC and e["src_type"] in KEY and e["dst_type"] in KEY, e
        groups.setdefault((e["rel"], e["src_type"], e["dst_type"]), []).append(
            {"src": e["src"], "dst": e["dst"], "cite": e["cite"]})
    for (rel, st, dt), rows in sorted(groups.items()):
        # 관계 종류·라벨은 파라미터로 못 넘기므로 위 assert로 검증한 값만 문자열에 넣는다
        cypher = (f"UNWIND $rows AS r MATCH (s:{st} {{{KEY[st]}: r.src}}), (d:{dt} {{{KEY[dt]}: r.dst}}) "
                  f"MERGE (s)-[x:{rel}]->(d) SET x.cite = r.cite")
        for i in range(0, len(rows), 500):
            driver.execute_query(cypher, {"rows": rows[i:i + 500]})
    print(f"  의미 관계  {len(edges)}  " + str({k: sum(len(v) for g, v in groups.items() if g[0] == k)
                                             for k in SEMANTIC}))


def connect():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    driver = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER", "neo4j"),
                                             os.getenv("NEO4J_PASSWORD", "legalkg1234")))
    driver.verify_connectivity()
    return driver


def load(driver, rows: dict, wipe: bool, dim: int | None) -> None:
    q = lambda cypher, **p: driver.execute_query(cypher, p)
    if wipe:
        # IN TRANSACTIONS는 자동 커밋 트랜잭션에서만 동작 → execute_query가 아닌 session.run 사용
        with driver.session() as s:
            s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 2000 ROWS").consume()
        print("기존 그래프 삭제")
    for s in SCHEMA:
        q(s)
    if dim:
        q(VECTOR_INDEX, dim=dim)
    for key in ("laws", "articles", "units", "sub_of", "proviso_of"):
        data = rows[key]
        for i in range(0, len(data), 500):
            q(LOAD[key], rows=data[i:i + 500])
        print(f"  {key:11} {len(data)}")


def report(driver) -> None:
    recs, _, _ = driver.execute_query(
        "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label")
    print("노드:", {r["label"]: r["n"] for r in recs})
    recs, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY t")
    print("관계:", {r["t"]: r["n"] for r in recs})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "legal_corpus.json"))
    ap.add_argument("--dry-run", action="store_true", help="DB 연결 없이 변환 결과만 확인")
    ap.add_argument("--wipe", action="store_true", help="기존 그래프를 지우고 새로 적재")
    ap.add_argument("--embed", action="store_true", help="OpenAI 임베딩 계산·적재 (캐시 사용)")
    ap.add_argument("--embed-model", default=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    ap.add_argument("--edges", default=str(DATA / "edges.csv"),
                    help="승인된 의미 관계 CSV (extract_edges.py 산출물, 없으면 건너뜀)")
    ap.add_argument("--no-edges", action="store_true", help="구조 관계만 적재")
    ap.add_argument("--categories", default=str(DATA / "categories.csv"),
                    help="거래 유형 Category 정의 CSV (없으면 건너뜀)")
    ap.add_argument("--no-categories", action="store_true", help="Category 없이 적재")
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    rows = build_rows(corpus)
    kinds = Counter(u["kind"] for u in rows["units"])
    print(f"법령 {len(rows['laws'])} / 조 {len(rows['articles'])} / 조항 {len(rows['units'])} "
          f"/ SUB_OF {len(rows['sub_of'])} / PROVISO_OF {len(rows['proviso_of'])}")
    print("조항 종류:", dict(kinds))
    print("법령:", {l["law_id"]: l["level"] for l in rows["laws"]})

    dim = None
    if args.embed:
        n_new = attach_embeddings(rows["units"], args.embed_model)
        dim = len(rows["units"][0]["embedding"])
        print(f"임베딩: 새로 계산 {n_new}개, 캐시 {len(rows['units']) - n_new}개, 차원 {dim}")
    else:
        for u in rows["units"]:
            u["embedding"] = None

    edges = [] if args.no_edges else read_edges(Path(args.edges))
    print(f"승인된 의미 관계: {len(edges)}개 ({args.edges})")
    cats, links = ([], []) if args.no_categories else category_rows(Path(args.categories), rows["articles"])
    print(f"Category: {len(cats)}개, APPLIES_TO {len(links)}개")

    if args.dry_run:
        return 0
    driver = connect()
    try:
        load(driver, rows, args.wipe, dim)
        if edges:
            load_edges(driver, edges)
        if cats:
            load_categories(driver, cats, links)
        report(driver)
    finally:
        driver.close()
    print("완료. 브라우저 http://localhost:7474 에서 확인할 수 있습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

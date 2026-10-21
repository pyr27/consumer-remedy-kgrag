#!/usr/bin/env python3
"""조문 텍스트 → 의미 관계 후보 (규칙 기반, LLM 미사용).

관계
    REFERS_TO     (인용하는 조항)-[:REFERS_TO]->(인용된 조항)        "제17조제2항", "같은 조 제1항"
    OVERRIDES     (예외 조항)-[:OVERRIDES]->(원칙 조항)              "제1항 및 제2항에도 불구하고"
    DELEGATES_TO  (법률 조항)-[:DELEGATES_TO]->(시행령 조항)         시행령의 "법 제N조에서 "대통령령으로 정하는 ~"란"
    USES_TERM     (용어를 쓰는 조항)-[:USES_TERM]->(정의 호)          '"계속거래"란 …'의 '계속거래'가 나오는 같은 법 계열 조항
                  (쓰이는 조항이 TERM_MAX_UNITS 넘는 용어는 허브가 되므로 제외)
    대상이 조 전체("제31조")이고 그 조에 항이 있으면 Article 노드를 가리킨다.

검토 파일 data/edges.csv (git 관리, 사람이 approved 열을 고친다)
    confidence  high   인용한 단위(항·호·목)가 코퍼스에 그대로 있음
                medium 인용한 단위가 없어 상위 단위로 올려 붙임 (파싱 오류·코퍼스 누락 가능) → 검토 대상
    approved    y/n. high는 y·reviewer=auto로 미리 채우고, medium은 빈칸.
    재실행해도 같은 (src, rel, dst)의 approved·reviewer·note는 보존한다.

사용
    python kg/extract_edges.py              # data/edges.csv 갱신 + 요약
    python kg/extract_edges.py --show ecom-17-3   # 특정 조항에서 뽑힌 관계 확인
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "edges.csv"

PARENT = {"enf": "ecom", "consenf": "cons", "doorenf": "door", "instenf": "inst"}  # 시행령 → 모법
ENF = {v: k for k, v in PARENT.items()}
MOK = {ch: i + 1 for i, ch in enumerate("가나다라마바사아자차카타파하")}
FIELDS = ["src", "src_type", "rel", "dst", "dst_type", "cite", "context",
          "confidence", "approved", "reviewer", "note"]

# 법령 지시어(선택) + 인용 핵심(조/항/호/목) + 단서(선택)
CITE = re.compile(
    r"(?:(?P<law>「[^」]+」(?:\s*\(이하\s*\"[^\"]+\"(?:이라|라)\s*한다\))?|같은\s법\s시행령|같은\s법|이\s법|(?<![가-힣])법|(?<![가-힣])영)\s*)?"
    r"(?:(?P<same>같은\s조|이\s조|같은\s항|이\s항)\s*)?"
    r"(?P<core>제\d+조(?:의\d+)?(?:\s?제\d+항)?(?:\s?제\d+호(?:의\d+)?)?(?:\s?[가-하]목)?"
    r"|제\d+항(?:\s?제\d+호(?:의\d+)?)?(?:\s?[가-하]목)?"
    r"|제\d+호(?:의\d+)?(?:\s?[가-하]목)?)"
    r"(?P<proviso>\s?단서)?"
)
PART = re.compile(r"제(\d+)조(?:의(\d+))?|제(\d+)항|제(\d+)호(?:의(\d+))?|([가-하])목")
FOREIGN_BEFORE = re.compile(r"(?:[가-힣]법|법률|」)\s*$")  # "민법 제N조" 등 코퍼스 밖 법
# 같은 인용 사슬로 보는 사이 문구: "제1항부터 제3항까지 및 제7항", "제28조제2항 전단 및 제29조", "제1항 내지 제3항"
ALIAS_DEF = re.compile(r"「([^」]+)」\s*\(이하\s*\"법\"")
CHAIN_GAP = re.compile(r"^\s*(?:까지(?:의\s*규정)?)?\s*(?:전단|후단|본문|각\s*호\s*외의\s*부분)?\s*"
                       r"(?:,|ㆍ|·|및|또는|과|와|이나|나|부터|내지)?\s*$")
OVERRIDE_AFTER = re.compile(r"^\s*(?:까지)?\s*(?:의\s*규정)?\s*에도\s*불구하고")
DELEGATE_AFTER = re.compile(r"^\s*(?:에서|에\s*따른|에\s*따라)[^.]{0,40}?대통령령으로\s*정하")


def norm(name: str) -> str:
    return re.sub(r"\s+", "", name)


# --------------------------------------------------------------------------- #
# 코퍼스 색인
# --------------------------------------------------------------------------- #
class Corpus:
    def __init__(self, corpus: dict):
        self.units: dict[str, dict] = {}      # unit_id → {doc_id, law_id, text}
        self.docs: dict[str, list[str]] = {}  # doc_id → unit_ids
        self.law_by_name: dict[str, str] = {}
        self.with_para: set[str] = set()      # 항(①②…)이 있는 조
        self.alias: dict[str, str] = {}       # law_id → '법'이 가리키는 law_id (「X」(이하 "법"이라 한다))
        for d in corpus["documents"]:
            if d.get("handmade"):
                continue
            doc_id = d["doc_id"]
            law_id = doc_id.split("-")[0]
            self.law_by_name.setdefault(norm(d.get("law", "")), law_id)
            raw = d.get("units") or [{"unit_id": doc_id, "text": d.get("text", "")}]
            self.docs[doc_id] = [u["unit_id"] for u in raw]
            if any(re.match(r"[①-⑳]", u.get("label", "")) for u in raw):
                self.with_para.add(doc_id)
            for u in raw:
                self.units[u["unit_id"]] = {"doc_id": doc_id, "law_id": law_id, "text": u["text"]}
        for u in self.units.values():
            m = ALIAS_DEF.search(u["text"])
            if m and norm(m.group(1)) in self.law_by_name:
                self.alias.setdefault(u["law_id"], self.law_by_name[norm(m.group(1))])

    def has_paragraphs(self, doc_id: str) -> bool:
        return doc_id in self.with_para


# --------------------------------------------------------------------------- #
# 인용 → 좌표 (law, art, art_sub, para, item, item_sub, mok, proviso)
# --------------------------------------------------------------------------- #
def parse_core(core: str) -> dict:
    out = {}
    for m in PART.finditer(core):
        if m.group(1):
            out["art"], out["art_sub"] = int(m.group(1)), m.group(2) and int(m.group(2))
        elif m.group(3):
            out["para"] = int(m.group(3))
        elif m.group(4):
            out["item"], out["item_sub"] = int(m.group(4)), m.group(5) and int(m.group(5))
        elif m.group(6):
            out["mok"] = MOK[m.group(6)]
    return out


def own_coords(unit_id: str, doc_id: str, c: Corpus) -> dict:
    """현재 조항의 위치 (같은 조·같은 항·생략된 상위 단위를 채우는 데 사용)."""
    law, art = doc_id.split("-", 1)
    a, _, s = art.partition("_")
    if not a.isdigit():  # 조문 형식이 아닌 지침(ecomg-g10): 같은 문서 안의 '제N조'는 없다
        return {"law": law}
    out = {"law": law, "art": int(a), "art_sub": int(s) if s else None}
    rest = unit_id[len(doc_id) + 1:].replace("-proviso", "").split("-") if unit_id != doc_id else []
    rest = [r for r in rest if r and r != "0"]
    if rest and c.has_paragraphs(doc_id):
        out["para"] = int(rest[0].split("_")[0])
    return out


def unit_candidates(co: dict, c: Corpus) -> list[tuple[str, str]]:
    """좌표 → (id, 라벨) 후보. 앞에서부터 가장 구체적인 것."""
    doc = f"{co['law']}-{co['art']}" + (f"_{co['art_sub']}" if co.get("art_sub") else "")
    parts = []
    if co.get("para") is not None:
        parts.append(str(co["para"]))
    if co.get("item") is not None:  # 항 없는 조의 호는 doc-호
        parts.append(f"{co['item']}" + (f"_{co['item_sub']}" if co.get("item_sub") else ""))
    if co.get("mok") is not None:
        parts.append(str(co["mok"]))
    cands = []
    for k in range(len(parts), 0, -1):
        uid = "-".join([doc] + parts[:k])
        if co.get("proviso") and k == len(parts):
            cands.append((uid + "-proviso", "exact"))
        cands.append((uid, "exact" if k == len(parts) and not co.get("proviso") else "up"))
    if co.get("proviso") and not parts:
        cands.append((doc + "-proviso", "exact"))
    cands.append((doc, "exact" if not parts and not co.get("proviso") else "up"))
    return cands


def resolve(co: dict, c: Corpus) -> tuple[str, str, str] | None:
    """→ (dst_id, dst_type, confidence). 코퍼스에 없는 조이면 None."""
    for uid, how in unit_candidates(co, c):
        if uid in c.units:
            return uid, "Unit", "high" if how == "exact" else "medium"
        if uid in c.docs:  # 조 전체 인용인데 항이 있는 조 → Article 노드
            return uid, "Article", "high" if how == "exact" else "medium"
    return None


# --------------------------------------------------------------------------- #
# 조항 하나에서 관계 추출
# --------------------------------------------------------------------------- #
def extract_unit(unit_id: str, c: Corpus) -> tuple[list[dict], list[str]]:
    info = c.units[unit_id]
    text, doc_id, own_law = info["text"], info["doc_id"], info["law_id"]
    own = own_coords(unit_id, doc_id, c)
    is_enf = own_law in PARENT

    refs, external = [], []
    prev, prev_end, explicit_law = None, None, None
    for m in CITE.finditer(text):
        gap = text[prev_end:m.start()] if prev_end is not None else None
        chained = prev is not None and gap is not None and CHAIN_GAP.match(gap) is not None
        co = parse_core(m.group("core"))
        law_tok = m.group("law")
        same = norm(m.group("same") or "")

        # 1) 어느 법인가
        if law_tok:
            t = norm(law_tok)
            if t.startswith("「"):
                name = t[1:t.index("」")]
                law = c.law_by_name.get(name)
                explicit_law = law or f"EXT:{name}"
            elif t == "같은법시행령":
                law = ENF.get(explicit_law or "", "EXT")
            elif t == "같은법":
                law = explicit_law or "EXT"
            elif t == "이법":
                law = own_law
            elif t == "법":
                law = c.alias.get(own_law) or PARENT.get(own_law, own_law)
            else:  # 영
                law = own_law if is_enf else ENF.get(own_law, "EXT")
        elif chained or (same in ("같은조", "같은항") and prev):  # "법 제47조 … 같은 조 제2항" → 법
            law = prev["law"]
        elif FOREIGN_BEFORE.search(text[max(0, m.start() - 12):m.start()]):
            law = "EXT"
        else:
            law = own_law
        if law is None or law.startswith("EXT"):
            external.append(m.group(0))
            prev, prev_end = {"law": "EXT"}, m.end()
            continue
        co["law"] = law

        # 2) 생략된 상위 단위 채우기 (같은 조·같은 항, 사슬 상속, 자기 위치)
        base = None
        if same in ("같은조", "같은항"):
            base = prev
        elif same in ("이조", "이항"):
            base = own
        elif "art" not in co:
            base = prev if chained and prev.get("law") == law else (own if law == own_law else None)
        if base and "art" not in co:
            co["art"], co["art_sub"] = base.get("art"), base.get("art_sub")
            if "para" not in co and ("item" in co or same in ("같은항", "이항")):
                co["para"] = base.get("para")
            if "item" not in co and "mok" in co:
                co["item"], co["item_sub"] = base.get("item"), base.get("item_sub")
        if "art" not in co or co["art"] is None:
            prev, prev_end = None, m.end()
            continue
        co["proviso"] = bool(m.group("proviso"))

        # 3) 범위 "제A조부터 제B조까지", "제A항 내지 제B항" → 사이 단위 펼치기
        batch = [co]
        if chained and gap and (("부터" in gap and text[m.end():m.end() + 3].lstrip().startswith("까지"))
                                or "내지" in gap):
            lvl = next((k for k in ("mok", "item", "para", "art") if co.get(k) is not None
                        and co.get(k) != prev.get(k)), None)
            if lvl and isinstance(prev.get(lvl), int):
                batch = [dict(co, **{lvl: v}, **({"art_sub": None} if lvl == "art" else {}))
                         for v in range(prev[lvl] + 1, co[lvl])] + [co]
        for b in batch:
            refs.append({"co": b, "start": m.start(), "end": m.end(), "cite": m.group(0).strip()})
        prev, prev_end = co, m.end()

    # 사슬 단위로 관계 종류 판정 (사슬 끝 뒤의 문구를 본다)
    edges = []
    i = 0
    while i < len(refs):
        j = i
        while j + 1 < len(refs) and (refs[j + 1]["start"] == refs[j]["start"]
                                     or CHAIN_GAP.match(text[refs[j]["end"]:refs[j + 1]["start"]])):
            j += 1
        after = text[refs[j]["end"]:]
        for r in refs[i:j + 1]:
            hit = resolve(r["co"], c)
            if not hit:
                continue
            dst, dst_type, conf = hit
            if dst == unit_id or (dst_type == "Article" and dst == doc_id and r["co"]["law"] == own_law
                                  and not r["co"].get("para")):
                continue  # 자기 자신·자기 조 전체
            ctx = text[max(0, r["start"] - 15): r["end"] + 25].replace("\n", " ")
            row = {"src": unit_id, "src_type": "Unit", "rel": "REFERS_TO", "dst": dst, "dst_type": dst_type,
                   "cite": r["cite"], "context": ctx, "confidence": conf}
            if OVERRIDE_AFTER.match(after):
                row["rel"] = "OVERRIDES"
            elif is_enf and r["co"]["law"] == PARENT[own_law] and DELEGATE_AFTER.match(after):
                row.update(rel="DELEGATES_TO", src=dst, src_type=dst_type, dst=unit_id, dst_type="Unit")
            edges.append(row)
        i = j + 1
    return edges, external


def skip(uid: str, c: Corpus) -> bool:
    """분쟁해결기준 항목·고시 별표(표)는 조문 인용 문장이 아니라서 추출 대상에서 뺀다."""
    return c.units[uid]["law_id"] == "std" or re.search(r"-b\d", c.units[uid]["doc_id"]) is not None


# --------------------------------------------------------------------------- #
# USES_TERM: 정의 조항('"계속거래"란 …')의 용어를 쓰는 조항 → 그 정의
# --------------------------------------------------------------------------- #
# 정의는 같은 법 계열(법률·시행령·고시) 안에서만 통한다
FAMILY = {"ecom": ("ecom", "enf", "ecomg"), "door": ("door", "doorenf", "contfee"),
          "inst": ("inst", "instenf"), "term": ("term",), "cons": ("cons", "consenf"),
          "contfee": ("contfee",)}
DEF_RE = re.compile(r"\"([^\"]{2,15})\"(?:이)?란")
TERM_MAX_UNITS = 60  # 이보다 많이 쓰이는 용어('소비자', '통신판매' 등)는 허브가 되므로 제외
TERM_STOP = {"단위", "대통령령으로 정하는 자", "대통령령으로 정하는 경우"}


def extract_terms(c: Corpus) -> tuple[list[dict], dict[str, int]]:
    defs: dict[str, list[tuple[str, str]]] = {}  # law_id → [(term, 정의 unit)]
    for uid, u in c.units.items():
        if re.fullmatch(r"[a-z]+-2", u["doc_id"]):  # 정의 조(제2조)
            for m in DEF_RE.finditer(u["text"]):
                if m.group(1) not in TERM_STOP:
                    defs.setdefault(u["law_id"], []).append((m.group(1), uid))
    rows, freq = [], {}
    for law, terms in defs.items():
        fam = FAMILY.get(law, (law,))
        users = [uid for uid, u in c.units.items() if u["law_id"] in fam and not skip(uid, c)]
        terms = sorted(terms, key=lambda t: -len(t[0]))  # 긴 용어 우선: '후원방문판매' 안의 '방문판매'는 세지 않음
        hits: dict[str, list[str]] = {t: [] for t, _ in terms}
        for uid in users:
            text = c.units[uid]["text"]
            taken = [False] * len(text)
            for term, def_uid in terms:
                for m in re.finditer(re.escape(term), text):
                    if any(taken[m.start():m.end()]):
                        continue
                    taken[m.start():m.end()] = [True] * len(term)
                    if uid != def_uid and (not hits[term] or hits[term][-1] != uid):
                        hits[term].append(uid)
        for term, def_uid in terms:
            freq[f"{def_uid}:{term}"] = len(hits[term])
            if len(hits[term]) > TERM_MAX_UNITS:
                continue
            for uid in hits[term]:
                rows.append({"src": uid, "src_type": "Unit", "rel": "USES_TERM", "dst": def_uid,
                             "dst_type": "Unit", "cite": term, "context": "", "confidence": "high"})
    return rows, freq


def extract_all(c: Corpus) -> tuple[list[dict], Counter]:
    seen, rows, ext = set(), [], Counter()
    term_rows, _ = extract_terms(c)
    for e in term_rows:  # 한 정의 호가 용어 둘을 정의하면('할부거래'·'할부거래업자') 같은 쌍이 두 번 나온다
        k = (e["src"], e["rel"], e["dst"])
        if k not in seen:
            seen.add(k)
            rows.append(e)
    for uid in c.units:
        if skip(uid, c):
            continue
        edges, external = extract_unit(uid, c)
        ext.update(re.sub(r"\s+", "", e[: e.index("」") + 1]) if "」" in e else e for e in external)
        for e in edges:
            k = (e["src"], e["rel"], e["dst"])
            if k not in seen:
                seen.add(k)
                rows.append(e)
    return rows, ext


def merge_review(rows: list[dict], path: Path) -> None:
    """기존 검토 결과(approved·reviewer·note)를 보존한다."""
    old = {}
    if path.exists():
        with path.open(encoding="utf-8-sig") as f:
            old = {(r["src"], r["rel"], r["dst"]): r for r in csv.DictReader(f)}
    for r in rows:
        o = old.get((r["src"], r["rel"], r["dst"]))
        if o and o.get("reviewer") not in ("", "auto"):
            r.update(approved=o["approved"], reviewer=o["reviewer"], note=o.get("note", ""))
        else:
            auto = r["confidence"] == "high"
            r.update(approved="y" if auto else "", reviewer="auto" if auto else "",
                     note=o.get("note", "") if o else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "legal_corpus.json"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--show", help="이 unit_id에서 뽑힌 관계만 출력 (파일 저장 안 함)")
    args = ap.parse_args()

    c = Corpus(json.loads(Path(args.corpus).read_text(encoding="utf-8")))
    if args.show:
        edges, external = extract_unit(args.show, c)
        print(c.units[args.show]["text"])
        for e in edges:
            print(f"  {e['src']} -{e['rel']}-> {e['dst']} ({e['dst_type']}, {e['confidence']})  «{e['cite']}»")
        if external:
            print("  외부 법령:", external)
        return 0

    rows, ext = extract_all(c)
    rows.sort(key=lambda r: (r["src"], r["rel"], r["dst"]))
    out = Path(args.out)
    merge_review(rows, out)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"관계 후보 {len(rows)}개 → {out}")
    print("종류:", dict(Counter(r["rel"] for r in rows)))
    print("신뢰도:", dict(Counter(r["confidence"] for r in rows)))
    print("대상 노드:", dict(Counter(r["dst_type"] for r in rows)))
    print("승인 대기(빈칸):", sum(1 for r in rows if not r["approved"]))
    print("코퍼스 밖 법령 인용 상위:", ext.most_common(8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

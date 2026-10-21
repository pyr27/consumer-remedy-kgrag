#!/usr/bin/env python3
"""공정거래위원회 고시·지침(행정규칙) 수집 → data/admrul_corpus.json

법제처 Open API의 행정규칙(target=admrul)을 사용한다. 인증키(OC)는 fetch_corpus.py와 같다.

    python fetch_admrul.py --oc <인증키>          # 또는 setx LAW_OC 후 인자 없이
    python fetch_admrul.py --parse-only           # raw/에 저장된 응답만 다시 파싱 (API 호출 없음)

수집 대상
    stdg    소비자분쟁해결기준 (고시 원문 — 일반적 기준 + 별표 품목별 기준)
    contfee 계속거래 등의 해지·해제에 따른 위약금 및 대금의 환급에 관한 산정기준
    ecomg   전자상거래 등에서의 소비자보호 지침

문서 구조
    조문:  {prefix}-{조}            항이 있으면 unit {prefix}-{조}-{항}
    별표:  {prefix}-b{번호}[_{가지}]  unit {prefix}-b{번호}-{n} (줄 단위로 묶은 덩어리)

주의
    별표가 본문 텍스트 없이 HWP/PDF 링크로만 제공되면 내용이 비어 있다.
    실행 결과의 '별표 본문 없음' 목록과 raw/{prefix}_body.json 을 확인할 것.
    그 경우 별표서식PDF파일링크의 PDF를 따로 받아 파싱해야 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fetch_corpus import CIRCLED, _decode, _get  # noqa: E402

RAW = ROOT / "raw"
OUT = ROOT / "data" / "admrul_corpus.json"

RULES = [
    {"prefix": "stdg", "query": "소비자분쟁해결기준", "must": ["소비자분쟁해결기준"]},
    {"prefix": "contfee", "query": "계속거래 등의 해지",
     "must": ["계속거래", "위약금", "산정기준"]},
    {"prefix": "ecomg", "query": "전자상거래 등에서의 소비자보호 지침",
     "must": ["전자상거래", "소비자보호", "지침"]},
]

ART_RE = re.compile(r"^\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*(?:\(([^)]*)\))?")
CIRC_SPLIT = re.compile(r"(?=[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def search(oc: str, rule: dict) -> dict | None:
    raw, url, status = _get("lawSearch.do", {"OC": oc, "target": "admrul", "type": "JSON",
                                              "query": rule["query"], "display": 20})
    text = _decode(raw)
    (RAW / f"{rule['prefix']}_search.txt").write_text(f"# {url}\n# HTTP {status}\n\n{text}",
                                                      encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"  [원인] JSON 아님 (HTTP {status}): {text[:200]}", file=sys.stderr)
        return None
    root = next(iter(data.values())) if isinstance(data, dict) else {}
    items = root.get("admrul") or []
    if isinstance(items, dict):
        items = [items]
    cands = [it for it in items if all(m in (it.get("행정규칙명") or "") for m in rule["must"])]
    if not cands:
        print(f"  [원인] 일치 결과 없음. 검색 결과: {[it.get('행정규칙명') for it in items][:10]}",
              file=sys.stderr)
        return None
    cands.sort(key=lambda it: (it.get("현행여부") != "현행", -int(it.get("발령일자") or 0)))
    best = cands[0]
    print(f"  선택: {best.get('행정규칙명')} / {best.get('발령일자')} / {best.get('현행여부', '')}")
    return best


def fetch(oc: str, rule: dict, item: dict) -> dict | None:
    sid = item.get("행정규칙일련번호")
    raw, url, status = _get("lawService.do", {"OC": oc, "target": "admrul", "type": "JSON", "ID": sid})
    text = _decode(raw)
    path = RAW / f"{rule['prefix']}_body.json"
    path.write_text(text, encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  [원인] 본문 JSON 아님 (HTTP {status}) → {path.name} 확인", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# 파싱
# --------------------------------------------------------------------------- #
def as_list(x) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def flatten_text(x) -> str:
    """API가 문자열/리스트/중첩 리스트 어느 것으로 주어도 하나의 문자열로."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(flatten_text(v) for v in x)
    if isinstance(x, dict):
        return "\n".join(flatten_text(v) for v in x.values())
    return "" if x is None else str(x)


def parse_articles(prefix: str, name: str, body) -> list[dict]:
    docs = []
    for block in as_list(body):
        text = flatten_text(block).strip()
        m = ART_RE.match(text)
        if not m:
            continue  # 장·절 제목 등
        num = m.group(1) + (f"_{m.group(2)}" if m.group(2) else "")
        doc_id = f"{prefix}-{num}"
        rest = text[m.end():].strip()
        parts = [p.strip() for p in CIRC_SPLIT.split(rest) if p.strip()]
        units = []
        if len(parts) > 1 or (parts and parts[0][0] in CIRCLED):
            for p in parts:
                if p[0] in CIRCLED:
                    units.append({"unit_id": f"{doc_id}-{CIRCLED[p[0]]}", "label": p[0],
                                  "text": p[1:].strip()})
                else:  # 항 앞의 본문
                    units.append({"unit_id": f"{doc_id}-0", "label": "", "text": p})
        else:
            units.append({"unit_id": doc_id, "label": "", "text": rest})
        docs.append({"doc_id": doc_id, "law": name, "article": f"제{m.group(1)}조"
                     + (f"의{m.group(2)}" if m.group(2) else ""), "title": m.group(3) or "",
                     "units": units})
    return docs


def chunk_lines(text: str, limit: int = 400) -> list[str]:
    """별표 본문(표가 텍스트로 풀린 형태)을 줄 경계를 지키며 limit자 안팎으로 묶는다."""
    out, cur = [], ""
    for line in (l.rstrip() for l in text.splitlines()):
        if not line.strip():
            continue
        if cur and len(cur) + len(line) > limit:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


def fix_qmark(text: str) -> str:
    """API 응답에서 가운뎃점(ㆍ)이 '?'로 깨진 것을 되돌린다.
    한글 사이('학원?교습')와 표 칸 끝('헬스?   │')만 바꾼다 — 문장 끝 물음표는 건드리지 않는다."""
    text = re.sub(r"(?<=[가-힣])\?(?=[가-힣])", "ㆍ", text)
    return re.sub(r"(?<=[가-힣])\?(?=\s*[│┃])", "ㆍ", text)


SECTION = re.compile(r"^\s*(Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ)\.\s*(.+?)\s*$")
GROUP = re.compile(r"^\s?(\d+)\.\s*(\S.{0,38}?)\s*$")       # 표 밖의 "10. 학원ㆍ교습ㆍ레저 서비스 관련"
ANGLE = re.compile(r"^\s*<\s*(.+?)\s*>\s*$")                  # " < ①자동차운전학원(1개 업종) >"
BOX_ROW = re.compile(r"^[┃│]\s*(.*?)\s*[┃│]?\s*$")


def chunk_annex(text: str, limit: int = 400) -> list[tuple[str, str]]:
    """별표(표가 텍스트로 풀린 형태) → [(제목 경로, 덩어리)].

    줄 경계를 지키며 limit자 안팎으로 묶되, 표가 새로 시작하거나 업종 묶음이 바뀌면 끊는다.
    덩어리마다 '부문 > 업종 묶음 > 품목' 제목을 붙여, 덩어리만 보고도 어느 품목의 기준인지 알 수 있게 한다.
    업종 묶음 번호는 순서대로 증가할 때만 제목으로 인정한다(세탁업 표 안의 '1. 배상액의 산정방식' 등 제외).
    """
    out: list[tuple[str, str]] = []
    section = group = items = ""
    expect = 1
    cur, cur_label = "", ""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]

    def flush():
        nonlocal cur
        if cur:
            out.append((cur_label, cur))
            cur = ""

    for i, line in enumerate(lines):
        boxed = line.lstrip()[:1] in "┏┃┠┣┗│├┌└┝"
        m_sec, m_grp, m_ang = SECTION.match(line), GROUP.match(line), ANGLE.match(line)
        if not boxed and m_sec:
            flush()
            section, group, items, expect = f"{m_sec.group(1)}. {m_sec.group(2)}", "", "", 1
        elif not boxed and m_grp and int(m_grp.group(1)) == expect:
            flush()
            group, items, expect = f"{m_grp.group(1)}. {m_grp.group(2)}", "", expect + 1
        elif not boxed and m_ang:
            flush()
            items = m_ang.group(1).lstrip("?")  # 원문자 번호가 '?'로 깨진 경우('?세탁업')
        elif line.lstrip().startswith("┏"):
            flush()
            # 표 첫 칸이 칸 나눔(│) 없는 한 줄이면 품목 목록: "┃①란류, ②육류 …┃"
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            row = BOX_ROW.match(nxt)
            if row and "│" not in nxt and "분" not in row.group(1)[:3]:
                items = re.sub(r"\(\s*\d+\s*-\s*\d+\s*\)", "", row.group(1)).strip()
        label = " > ".join(x for x in (section, group, items) if x)
        if cur and (len(cur) + len(line) > limit or label != cur_label):
            flush()
        cur_label = label
        cur = f"{cur}\n{line}" if cur else line
    flush()
    return out


# 조문 형식이 아닌 고시·지침 (조문형식여부 N): "Ⅱ. 일반사항1. 일반적 준수사항사업자는 … 가. … 나. …"
OUTLINE_SECTIONS = {"I": "p", "Ⅰ": "p", "Ⅱ": "g", "Ⅲ": "r", "Ⅳ": "x", "Ⅴ": "y"}
OUTLINE_SEC = re.compile(r"(?:^|(?<=[.　\s]))(I|Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ)\.\s*")
HANGUL_SEQ = "가나다라마바사아자차카타파하"
TITLE_END = re.compile(r"(?=사업자는|사업자가|법\s?제|이 지침|이 고시|「|사이버몰 등에|소비자는|통신판매업자는)")


def _seq_split(body: str, marks: list[str], pos: int = 0) -> list[tuple[int, int]]:
    """순서대로 나타나는 번호 표시(1. 2. … 또는 가. 나. …)의 위치.
    숫자는 제목이 앞 글자에 붙어 오므로('구성1. 목적') 앞 숫자만 막고, 한글 기호는 앞 글자가 한글이면
    문장 끝('한다.')으로 보고 건너뛴다."""
    out = []
    for k in marks:
        if k.isdigit():
            pats = [rf"(?<![0-9(]){k}\.\s?(?=[가-힣「‘\"])"]
        else:
            pats = [rf"(?<![가-힣0-9A-Za-z(]){k}\.\s?(?=[가-힣「‘\"])",
                    # 예시 끝에 마침표 없이 붙어 오는 경우('…경우나. 법 제21조').
                    # '다.'는 문장 끝('한다.')과 겹치므로 항목 첫머리로 보이는 말이 이어질 때만
                    (rf"(?<=[가-힣]){k}\.\s(?=[가-힣「‘\"])" if k != "다"
                     else r"(?<=[가-힣])다\.\s(?=법\s?제|「|사업자|통신판매)")]
        ms = [m for p in pats if (m := re.compile(p).search(body, pos))]
        m = min(ms, key=lambda x: x.start()) if ms else None
        if not m:
            break
        out.append((m.start(), m.end()))
        pos = m.end()
    return out


def parse_outline(prefix: str, name: str, text: str) -> list[dict]:
    text = fix_qmark(text.replace("　", " "))
    secs = [(m.start(), m.end(), m.group(1)) for m in OUTLINE_SEC.finditer(text)]
    docs = []
    for (s, b, roman), nxt in zip(secs, secs[1:] + [(len(text), 0, "")]):
        body = text[b:nxt[0]].strip()
        sec_title = re.match(r"^[가-힣 ]{1,12}?(?=\d+\.|「|$)", body)
        sec_name = f"{roman}. {sec_title.group(0).strip() if sec_title else ''}".strip()
        code = OUTLINE_SECTIONS.get(roman, roman)
        heads = _seq_split(body, [str(n) for n in range(1, 40)])
        if not heads:  # 번호 없는 장 (재검토기한 등)
            rest = body[sec_title.end():].strip() if sec_title else body
            docs.append({"doc_id": f"{prefix}-{code}", "law": name, "article": sec_name, "title": "",
                         "units": [{"unit_id": f"{prefix}-{code}", "label": "", "text": rest}]})
            continue
        for n, ((hs, he), nx) in enumerate(zip(heads, heads[1:] + [(len(body), 0)]), start=1):
            item = body[he:nx[0]].strip()
            # 첫 '가.'는 제목 끝에 붙어 온다('청약철회 방해 등 관련가. 법 제18조') → 제목 경계로 쓴다
            first = re.compile(r"가\.\s?(?=[가-힣「‘\"])").search(item[:60])
            if first:
                title, item = item[:first.start()].strip(), item[first.start():].strip()
                lead = TITLE_END.search(title, 1)  # '일반적 준수사항사업자는 … 준수하여야 한다.가.'
                if lead and len(title) > 30:
                    title, item = title[:lead.start()].strip(), f"{title[lead.start():]}{item}"
                g = re.search(r"(?<!\S)가\.\s?|가\.\s?(?=[가-힣「‘\"])", item)
                subs = [(g.start(), g.end())] + _seq_split(item, list(HANGUL_SEQ[1:]), g.end())
            else:
                m = TITLE_END.search(item, 1)
                t_end = m.start() if m and m.start() <= 40 else 0
                title, item = item[:t_end].strip(), item[t_end:].strip()
                subs = _seq_split(item, list(HANGUL_SEQ))
            doc_id = f"{prefix}-{code}{n}"
            if not subs:
                units = [{"unit_id": doc_id, "label": "", "text": item}]
            else:
                units = []
                if item[:subs[0][0]].strip():
                    units.append({"unit_id": f"{doc_id}-0", "label": "본문", "text": item[:subs[0][0]].strip()})
                for k, ((ss, se), sx) in enumerate(zip(subs, subs[1:] + [(len(item), 0)]), start=1):
                    units.append({"unit_id": f"{doc_id}-{k}", "label": f"{HANGUL_SEQ[k - 1]}.",
                                  "text": item[se:sx[0]].strip()})
            docs.append({"doc_id": doc_id, "law": name, "article": f"{sec_name} {n}.", "title": title,
                         "units": units})
    return docs


def parse_annexes(prefix: str, name: str, annex) -> tuple[list[dict], list[str]]:
    docs, empty = [], []
    for a in as_list(annex):
        if not isinstance(a, dict):
            continue
        no = str(a.get("별표번호") or "").lstrip("0") or "0"
        br = str(a.get("별표가지번호") or "").lstrip("0")
        doc_id = f"{prefix}-b{no}" + (f"_{br}" if br else "")
        title = (a.get("별표제목") or "").strip()
        content = fix_qmark(flatten_text(a.get("별표내용")).strip())
        if not content:
            empty.append(f"{doc_id} {title} | PDF: {a.get('별표서식PDF파일링크', '')}")
            continue
        units = [{"unit_id": f"{doc_id}-{i}", "label": label, "text": c}
                 for i, (label, c) in enumerate(chunk_annex(content), start=1)]
        docs.append({"doc_id": doc_id, "law": name, "article": f"[별표 {no}"
                     + (f"의{br}" if br else "") + "]", "title": title, "units": units,
                     "pdf": a.get("별표서식PDF파일링크", "")})
    return docs, empty


def parse(prefix: str, data: dict) -> tuple[list[dict], list[str]]:
    root = data.get("AdmRulService") or next(iter(data.values()))
    info = root.get("행정규칙기본정보") or root
    name = (info.get("행정규칙명") or prefix).strip()
    if info.get("조문형식여부") == "N":  # 전자상거래 지침처럼 Ⅰ.Ⅱ. 번호 체계인 고시
        arts = parse_outline(prefix, name, flatten_text(root.get("조문내용")))
    else:
        arts = parse_articles(prefix, name, root.get("조문내용"))
    annex_root = root.get("별표") or {}
    annex = annex_root.get("별표단위") if isinstance(annex_root, dict) else annex_root
    annexes, empty = parse_annexes(prefix, name, annex)
    return arts + annexes, empty


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oc", default=os.getenv("LAW_OC"))
    ap.add_argument("--parse-only", action="store_true", help="raw/*_body.json 만 다시 파싱")
    args = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    if not args.parse_only and not args.oc:
        sys.exit("LAW_OC 환경변수 또는 --oc 를 지정하세요.")

    all_docs = []
    for rule in RULES:
        print(f"\n[{rule['prefix']}] {rule['query']}")
        path = RAW / f"{rule['prefix']}_body.json"
        if args.parse_only:
            if not path.exists():
                print(f"  {path.name} 없음 — 건너뜀")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            item = search(args.oc, rule)
            if not item:
                continue
            time.sleep(0.4)
            data = fetch(args.oc, rule, item)
            if not data:
                continue
        try:
            docs, empty = parse(rule["prefix"], data)
        except Exception as e:  # noqa: BLE001 — 구조가 예상과 다르면 raw를 보고 고친다
            print(f"  [파싱 실패] {type(e).__name__}: {e} → raw/{path.name} 확인", file=sys.stderr)
            continue
        n_art = sum(1 for d in docs if "-b" not in d["doc_id"])
        n_unit = sum(len(d["units"]) for d in docs)
        print(f"  조 {n_art} / 별표 {len(docs) - n_art} / 단위 {n_unit}")
        if empty:
            print(f"  별표 본문 없음 {len(empty)}건 (PDF로만 제공):")
            for e in empty[:10]:
                print("    -", e)
        all_docs += docs
        time.sleep(0.4)

    OUT.write_text(json.dumps({"documents": all_docs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT} (문서 {len(all_docs)}개) → 다음: python merge_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

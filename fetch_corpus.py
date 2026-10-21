#!/usr/bin/env python3
"""법제처 국가법령정보 Open API로 법령 코퍼스를 수집한다.

컨테이너가 아니라 **본인 PC에서** 실행해야 한다(법제처는 외부 프록시에서 막힐 수 있음).

사용:
    export LAW_OC=<본인 이메일의 @ 앞부분>

    # 1) 응답 구조부터 확인 (가장 먼저 할 것)
    python fetch_corpus.py --inspect

    # 2) 수집
    python fetch_corpus.py --out data/legal_corpus.json

주의:
    법제처 응답 스키마는 법령마다 조금씩 다르다. --inspect로 원본을 먼저 저장해
    raw/ 폴더의 JSON을 눈으로 확인한 뒤 수집하는 것을 권장한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "https://www.law.go.kr/DRF"

# 수집 대상. prefix는 unit_id의 앞머리가 된다.
LAWS = [
    {
        "name": "전자상거래 등에서의 소비자보호에 관한 법률",
        "prefix": "ecom",
        "target": "law",
    },
    {
        "name": "전자상거래 등에서의 소비자보호에 관한 법률 시행령",
        "prefix": "enf",
        "target": "law",
    },
    {
        "name": "약관의 규제에 관한 법률",
        "prefix": "term",
        "target": "law",
    },
    # 피해구제 신청·분쟁조정 절차의 근거법 — "현재 단계 판단" 기능에 필요
    {
        "name": "소비자기본법",
        "prefix": "cons",
        "target": "law",
    },
    {
        "name": "소비자기본법 시행령",
        "prefix": "consenf",
        "target": "law",
    },
    # 헬스장·학원 등 계속거래 중도해지의 근거법
    {
        "name": "방문판매 등에 관한 법률",
        "prefix": "door",
        "target": "law",
    },
    {
        "name": "방문판매 등에 관한 법률 시행령",
        "prefix": "doorenf",
        "target": "law",
    },
    # 신용카드 할부 결제 시 항변권·철회 — FAQ의 "카드사에 대금 지급 거절 요청" 근거
    {
        "name": "할부거래에 관한 법률",
        "prefix": "inst",
        "target": "law",
    },
    {
        "name": "할부거래에 관한 법률 시행령",
        "prefix": "instenf",
        "target": "law",
    },
]
# prefix 규칙: 하이픈 없이 영문 소문자만 쓴다. unit_id를 "-"로 분해할 때
# 첫 토큰이 곧 법령이 되도록 하기 위함이다. ('enf'는 v1 gold_citations와의
# 호환을 위해 전자상거래법 시행령에 그대로 둔다.)

# 한글 원문자 항번호 → 숫자
CIRCLED = {c: str(i + 1) for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮")}


def _get(path: str, params: dict) -> tuple[bytes, str, int]:
    """(응답본문, 요청URL, HTTP상태) — 실패해도 본문을 돌려준다."""
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params, encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read(), url, resp.status
    except urllib.error.HTTPError as e:
        return e.read(), url, e.code


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "euc-kr", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def search_law(
    oc: str, name: str, target: str, raw_dir: Path, prefix: str, verbose: bool = True
) -> dict | None:
    """법령명으로 검색해 첫 번째 결과의 식별자를 돌려준다.

    응답 원본을 항상 raw/에 저장한다. 결과가 없을 때 원인을 눈으로 확인하기 위해서다.
    """
    raw, url, status = _get(
        "lawSearch.do",
        {"OC": oc, "target": target, "type": "JSON", "query": name, "display": 5},
    )
    text = _decode(raw)

    dump = raw_dir / f"{prefix}_search.txt"
    dump.write_text(f"# {url}\n# HTTP {status}\n\n{text}", encoding="utf-8")
    if verbose:
        print(f"  요청: {url}")
        print(f"  HTTP {status}, 응답 {len(text)}자 → raw/{dump.name}")

    # JSON이 아니면(인증 실패 시 HTML 안내가 온다) 바로 알려준다
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        print(f"  [원인] JSON이 아닌 응답입니다. 앞부분: {stripped[:200]}", file=sys.stderr)
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [원인] JSON 파싱 실패: {e}", file=sys.stderr)
        return None

    root = next(iter(data.values())) if isinstance(data, dict) else None
    if not isinstance(root, dict):
        print(f"  [원인] 예상과 다른 최상위 구조: {list(data)[:5]}", file=sys.stderr)
        return None

    total = root.get("totalCnt")
    if verbose and total is not None:
        print(f"  totalCnt={total}")

    items = root.get("law") or root.get("admrul") or root.get("Law") or []
    if isinstance(items, dict):
        items = [items]
    if not items:
        print(f"  [원인] 결과 0건. 응답 키: {list(root)[:10]}", file=sys.stderr)
        return None

    for it in items:
        title = it.get("법령명한글") or it.get("행정규칙명") or ""
        if title.strip() == name:
            return it
    first = items[0]
    if verbose:
        got = first.get("법령명한글") or first.get("행정규칙명") or "?"
        print(f"  정확히 일치하는 법령명이 없어 첫 결과를 사용: {got}")
    return first


def fetch_law_body(oc: str, target: str, ident: dict, raw_dir: Path, prefix: str) -> dict:
    params = {"OC": oc, "target": target, "type": "JSON"}
    for key in ("법령일련번호", "법령ID", "행정규칙일련번호", "행정규칙ID"):
        if ident.get(key):
            params["MST" if "일련번호" in key else "ID"] = ident[key]
            break
    raw, url, status = _get("lawService.do", params)
    text = _decode(raw)
    (raw_dir / f"{prefix}_body.txt").write_text(
        f"# {url}\n# HTTP {status}\n\n{text}", encoding="utf-8"
    )
    print(f"  본문 HTTP {status} → raw/{prefix}_body.txt")
    return json.loads(text)


# --------------------------------------------------------------------------- #
# 파싱
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = text.replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def _strip_label(text: str) -> str:
    """'① ...', '1. ...', '1의2. ...', '가. ...' 등 앞에 붙은 번호 표시를 떼어낸다."""
    return re.sub(
        r"^\s*(?:[①-⑮]|\d+의\d+\.|\d+\.|[가-힣]\.)\s*", "", text or ""
    ).strip()


def _branch_suffix(raw: dict, key: str) -> str:
    """가지번호가 있으면 '_N'을 돌려준다. 1의2 → '1_2' 형태로 충돌을 막는다."""
    b = str(raw.get(key, "") or "").strip()
    return f"_{b}" if b and b != "0" else ""


def _split_proviso(text: str) -> tuple[str, str | None]:
    """'다만, ...' 이후를 단서로 분리한다."""
    m = re.search(r"(다만[,\s])", text)
    if not m or m.start() == 0:
        return text, None
    return text[: m.start()].strip(), text[m.start() :].strip()


def _as_list(x) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _emit(out: list[dict], unit_id: str, label: str, text: str) -> None:
    """본문과 단서를 분리해 단위 리스트에 추가한다."""
    if not text:
        return
    body, proviso = _split_proviso(text)
    out.append({"unit_id": unit_id, "label": label, "text": body})
    if proviso:
        out.append(
            {"unit_id": f"{unit_id}-proviso", "label": f"{label} 단서".strip(), "text": proviso}
        )


def parse_articles(body: dict, prefix: str, law_name: str) -> list[dict]:
    """법제처 본문 응답을 코퍼스 문서 리스트로 변환.

    실제 응답 구조에서 확인된 특성을 반영한다.
      - 단서('다만, ...')는 별도 필드가 아니라 항내용·호내용 문장 안에 들어 있다
      - '항'은 리스트이거나, 항번호 없이 호만 담은 딕셔너리일 수 있다 (제2조)
      - 항이 있는 조의 조문내용은 보통 제목뿐이지만, 본문이 붙는 경우도 있다
      - 호 아래 목(가./나.)이 중첩될 수 있다
      - 조문가지번호가 있는 조(제9조의2)는 항 번호와 충돌하지 않게 '_'로 구분한다
    """
    root = next(iter(body.values())) if isinstance(body, dict) else {}
    units = _as_list(root.get("조문", {}).get("조문단위"))
    if not units:
        print(f"  [경고] {law_name}: 조문단위를 찾지 못했습니다. raw/를 확인하세요.",
              file=sys.stderr)
        return []

    docs: list[dict] = []
    seen: set[str] = set()

    for art in units:
        if str(art.get("조문여부", "조문")) != "조문":  # 편/장/절 제목 행 제외
            continue
        no = str(art.get("조문번호", "")).strip()
        if not no:
            continue

        branch = str(art.get("조문가지번호", "") or "").strip()
        has_branch = bool(branch) and branch != "0"
        art_label = f"제{no}조" + (f"의{branch}" if has_branch else "")
        doc_id = f"{prefix}-{no}" + (f"_{branch}" if has_branch else "")
        if doc_id in seen:
            continue
        seen.add(doc_id)

        hangs = _as_list(art.get("항"))
        doc_units: list[dict] = []

        # 조문내용에서 '제N조(제목)' 머리말을 떼고 남는 본문
        head = _clean(art.get("조문내용", ""))
        head = re.sub(r"^제\s*\d+\s*조(?:의\s*\d+)?\s*(?:\([^)]*\))?\s*", "", head).strip()

        if not hangs:
            # 항이 없는 조 — 조문내용 전체가 본문이며 unit_id는 doc_id와 같다
            _emit(doc_units, doc_id, "", head)
        else:
            if head:  # 항 앞에 본문이 따로 있는 조 (예: 제2조 정의)
                _emit(doc_units, f"{doc_id}-0", "본문", head)

            for hang in hangs:
                h_raw = str(hang.get("항번호", "") or "").strip()
                h_text = _strip_label(_clean(hang.get("항내용", "")))
                numbered = bool(h_raw)
                h_no = CIRCLED.get(h_raw, re.sub(r"\D", "", h_raw)) if numbered else ""
                h_no += _branch_suffix(hang, "항가지번호")
                # 항번호가 없으면 호를 조 바로 아래로 붙인다 (제2조제5호 → ecom-2-5)
                h_id = f"{doc_id}-{h_no}" if h_no else doc_id

                if h_text:
                    _emit(doc_units, h_id, h_raw or "본문", h_text)

                for ho in _as_list(hang.get("호")):
                    ho_raw = str(ho.get("호번호", "") or "").strip()
                    ho_no = re.sub(r"\D", "", ho_raw)
                    if not ho_no:
                        continue
                    ho_branch = str(ho.get("호가지번호", "") or "").strip()
                    ho_id = f"{h_id}-{ho_no}" + (
                        f"_{ho_branch}" if ho_branch and ho_branch != "0" else ""
                    )
                    ho_label = ho_no + (f"의{ho_branch}" if ho_branch and ho_branch != "0" else "")
                    _emit(
                        doc_units,
                        ho_id,
                        f"{h_raw} {ho_label}.".strip(),
                        _strip_label(_clean(ho.get("호내용", ""))),
                    )

                    for i, mok in enumerate(_as_list(ho.get("목")), start=1):
                        mok_raw = str(mok.get("목번호", "") or "").strip()
                        _emit(
                            doc_units,
                            f"{ho_id}-{i}",
                            f"{ho_no}. {mok_raw}".strip(),
                            _strip_label(_clean(mok.get("목내용", ""))),
                        )

        if doc_units:
            docs.append(
                {
                    "doc_id": doc_id,
                    "law": law_name,
                    "article": art_label,
                    "title": _clean(art.get("조문제목", "")),
                    "units": doc_units,
                }
            )
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oc", default=os.getenv("LAW_OC"), help="법제처 OC (이메일 @ 앞부분)")
    ap.add_argument("--out", default=str(ROOT / "data" / "legal_corpus_fetched.json"))
    ap.add_argument("--inspect", action="store_true", help="원본 응답만 raw/에 저장하고 종료")
    args = ap.parse_args()

    if not args.oc:
        print("LAW_OC 환경변수 또는 --oc 를 지정하세요.", file=sys.stderr)
        return 1

    raw_dir = ROOT / "raw"
    raw_dir.mkdir(exist_ok=True)
    all_docs: list[dict] = []

    print(f"OC = {args.oc!r}  (raw/ 폴더에 모든 응답 원본이 저장됩니다)")
    ok_count = 0

    for law in LAWS:
        print(f"\n[{law['name']}]")
        try:
            ident = search_law(
                args.oc, law["name"], law["target"], raw_dir, law["prefix"]
            )
        except Exception as e:  # noqa: BLE001
            print(f"  검색 요청 실패: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if not ident:
            print("  → 건너뜀", file=sys.stderr)
            continue

        time.sleep(0.4)
        try:
            body = fetch_law_body(
                args.oc, law["target"], ident, raw_dir, law["prefix"]
            )
        except Exception as e:  # noqa: BLE001
            print(f"  본문 조회 실패: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        (raw_dir / f"{law['prefix']}.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ok_count += 1

        if args.inspect:
            continue

        docs = parse_articles(body, law["prefix"], law["name"])
        n_units = sum(len(d["units"]) for d in docs)
        n_prov = sum(
            1 for d in docs for u in d["units"] if u["unit_id"].endswith("-proviso")
        )
        print(f"  조 {len(docs)}개, 단위 {n_units}개 (단서 {n_prov}개)")
        all_docs.extend(docs)
        time.sleep(0.4)

    if args.inspect:
        if ok_count:
            print(f"\n{ok_count}건 원본 저장 완료. raw/ 안의 JSON 구조를 확인한 뒤 "
                  "--inspect 없이 다시 실행하세요.")
        else:
            print("\n[실패] 한 건도 가져오지 못했습니다.", file=sys.stderr)
            print("raw/*_search.txt 를 열어 응답 원본을 확인하세요. "
                  "OC가 틀리면 보통 JSON 대신 안내 페이지가 돌아옵니다.", file=sys.stderr)
            return 1
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "_note": "법제처 Open API 자동 수집본. 소비자분쟁해결기준(공정위 고시)은 "
                         "API 제공 범위가 다르므로 기존 legal_corpus.json에서 수동으로 합칠 것.",
                "documents": all_docs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n총 {len(all_docs)}개 조 저장: {out}")
    print("기존 코퍼스의 소비자분쟁해결기준 문서를 여기에 합친 뒤 legal_corpus.json으로 바꾸세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

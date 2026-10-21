#!/usr/bin/env python3
"""소비자분쟁해결기준 품목별 해결기준 CSV(공공데이터) → 코퍼스 문서(std-<품종코드>).

공공데이터 CSV 열: 분쟁유형, 해결기준, 계산 항목, 품종, 품종코드, 해결기준 채팅안내문
- 해결기준이 빈 행은 '상위 분쟁유형'(제목)이다. 바로 뒤 행들이 그 하위 항목이다.
- 같은 품종 안에 똑같은 블록이 반복되는 경우가 있어 중복을 제거한다.

사용:
    python build_std.py 소비자분쟁해결기준_품목별해결기준.csv
        → data/std_kca.json   (merge_corpus.py가 자동으로 합친다)

주의: 상위 분쟁유형(제목)은 CSV 순서로 추정한 것이라 고시 원문과 다를 수 있다.
주의: 이 데이터는 21개 품종만 담고 있다(가구·가전·자동차·스마트폰 등).
체력단련장·인터넷쇼핑몰·의류·숙박·공연 등은 없으므로 고시 원문 별표가 따로 필요하다.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "std_kca.json"
LAW = "소비자분쟁해결기준(공정거래위원회 고시) 품목별 해결기준"


def read_rows(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            with path.open(encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    sys.exit("CSV 인코딩을 판별하지 못했습니다.")


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = read_rows(Path(sys.argv[1]))

    docs: dict[str, dict] = {}
    heads: list[str] = []
    prev_was_head = False
    prev_code = None
    for r in rows:
        name, code = r["품종"].strip(), r["품종코드"].strip()
        typ, rule = r["분쟁유형"].strip(), r["해결기준"].strip()
        if code != prev_code:
            heads, prev_was_head = [], False
            prev_code = code
        if not rule:  # 상위 분쟁유형
            # CSV에는 들여쓰기 깊이 정보가 없어 다단 중첩을 복원할 수 없다.
            # 잘못된 긴 체인을 만드는 것보다 직전 제목 하나만 붙이는 편이 안전하다.
            heads = [typ]
            prev_was_head = True
            continue
        prev_was_head = False

        doc = docs.setdefault(code, {
            "doc_id": f"std-{code}", "law": LAW, "article": "", "title": name,
            "source": "data.go.kr 소비자분쟁해결기준 품목별 해결기준", "units": [], "_seen": set(),
        })
        label = " > ".join(heads + [typ])
        key = (re.sub(r"\s+", "", typ), re.sub(r"\s+", "", rule))  # 같은 기준이 블록 반복으로 여러 번 나오므로 제목과 무관하게 중복 제거
        if key in doc["_seen"]:
            continue
        doc["_seen"].add(key)
        calc = r.get("계산 항목", "").strip()
        text = f"{label}: {rule}" + (f" (계산 항목: {calc})" if calc else "")
        doc["units"].append({"unit_id": f"std-{code}-{len(doc['units']) + 1}",
                             "label": "", "text": text})

    for d in docs.values():
        d.pop("_seen")
    OUT.write_text(json.dumps({"documents": list(docs.values())}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    n = sum(len(d["units"]) for d in docs.values())
    print(f"품종 {len(docs)}개, 기준 {n}개 (원본 {len(rows)}행, 제목행·중복 제외) → {OUT}")
    for d in docs.values():
        print(f"  {d['doc_id']:14} {d['title']:10} {len(d['units'])}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

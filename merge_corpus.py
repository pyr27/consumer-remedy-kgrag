#!/usr/bin/env python3
"""법제처 수집본과 수작업 문서(소비자분쟁해결기준)를 합치고 검증한다.

fetch_corpus.py가 만든 legal_corpus_fetched.json에는 법령만 들어 있다.
소비자분쟁해결기준은 공정위 고시라 법령 API 범위 밖이므로, 기존
legal_corpus.json에 수작업으로 넣어 둔 std-* 문서를 가져와 합친다.

사용:
    python merge_corpus.py

동작:
    1. data/legal_corpus.json 을 legal_corpus_handmade.json 으로 백업
    2. 수집본 + std-* 문서를 합쳐 data/legal_corpus.json 으로 저장
    3. test_questions.json의 gold_citations가 모두 존재하는지 검증
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    fetched_path = DATA / "legal_corpus_fetched.json"
    current_path = DATA / "legal_corpus.json"

    if not fetched_path.exists():
        print(f"{fetched_path.name} 이 없습니다. 먼저 fetch_corpus.py를 실행하세요.",
              file=sys.stderr)
        return 1

    fetched = load(fetched_path)
    handmade = load(current_path) if current_path.exists() else {"documents": []}

    # 법령 API로 못 가져오는 문서(소비자분쟁해결기준 등)만 기존 코퍼스에서 가져온다
    carried = [
        d for d in handmade["documents"]
        if d["doc_id"].startswith("std-")
    ]
    fetched_ids = {d["doc_id"] for d in fetched["documents"]}
    carried = [d for d in carried if d["doc_id"] not in fetched_ids]
    # 수작업 std-* (std-fitness 등)는 공식 원문이 아니라 요약문이다.
    # v1 재현을 위해 남기되 표시해 두고, evaluate_v2는 기본적으로 제외한다.
    std_kca_path = DATA / "std_kca.json"
    official = load(std_kca_path)["documents"] if std_kca_path.exists() else []
    # 공정위 고시·지침 원문 (fetch_admrul.py) — 분쟁해결기준 원문, 계속거래 산정기준, 전자상거래 지침
    adm_path = DATA / "admrul_corpus.json"
    admrul = load(adm_path)["documents"] if adm_path.exists() else []
    if not admrul:
        print("[안내] data/admrul_corpus.json 없음 — fetch_admrul.py를 실행하면 고시·지침이 합쳐집니다.")
    official = official + admrul
    official_ids = {d["doc_id"] for d in official}
    carried = [d for d in carried if d["doc_id"] not in official_ids]
    for d in carried:
        d["handmade"] = True
    if not official:
        print("[안내] data/std_kca.json 없음 — build_std.py를 먼저 실행하면 공식 분쟁해결기준이 합쳐집니다.")

    merged = {
        "_note": "법제처 Open API 수집본 + 소비자분쟁해결기준(수작업). "
                 "분쟁해결기준 텍스트는 공정위 고시 원문과 대조 검증 필요.",
        "documents": fetched["documents"] + official + carried,
    }

    if current_path.exists():
        backup = DATA / "legal_corpus_handmade.json"
        shutil.copy2(current_path, backup)
        print(f"기존 코퍼스 백업: {backup.name}")

    current_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_units = sum(len(d.get("units", [])) or 1 for d in merged["documents"])
    print(f"병합 완료: 문서 {len(merged['documents'])}개 "
          f"(법령 {len(fetched['documents'])} + 분쟁해결기준 공식 {len(official)} + 수작업 {len(carried)}), "
          f"단위 약 {n_units}개")

    # ---- gold_citations 검증 ----
    sys.path.insert(0, str(ROOT))
    from rag.chunking import corpus_unit_ids

    available = corpus_unit_ids(merged)
    # v2 FAQ 정답 근거 (평가 기준). v1 test_questions.json은 dev/에 없을 수 있다.
    faq_path = DATA / "v2" / "questions_faq.jsonl"
    if faq_path.exists():
        faq = [json.loads(l) for l in faq_path.open(encoding="utf-8") if l.strip()]
        miss = sorted({c for q in faq for c in q["contexts"]} - set(available))
        print(f"v2 FAQ 정답 근거 검증: {'전부 일치' if not miss else f'코퍼스에 없음 {miss}'}")
        if miss:
            return 1
    v1_path = DATA / "test_questions.json"
    if not v1_path.exists():
        print("v1 test_questions.json 없음 — v1 검증 건너뜀")
        return 0
    qset = load(v1_path)

    missing_total = 0
    for q in qset["questions"]:
        missing = [c for c in q["gold_citations"] if c not in available]
        if missing:
            missing_total += len(missing)
            print(f"  [경고] {q['qid']}: 코퍼스에 없는 근거 {missing}", file=sys.stderr)

    if missing_total:
        print(f"\n{missing_total}건의 gold_citation이 코퍼스에 없습니다. "
              f"test_questions.json의 unit_id를 실제 코퍼스에 맞게 고치세요.", file=sys.stderr)
        return 1

    print("gold_citations 검증: 전부 일치")
    print("\n다음: python inspect_chunks.py 로 확충된 코퍼스의 청킹 진단을 다시 돌리세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

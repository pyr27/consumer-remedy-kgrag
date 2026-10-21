"""fetch_admrul 파서 규칙 회귀 테스트 (API·raw 파일 불필요).  python -m pytest -q"""

from fetch_admrul import chunk_annex, fix_qmark, parse_outline


def test_fix_qmark_only_between_hangul_or_cell_end():
    assert fix_qmark("학원?교습?레저") == "학원ㆍ교습ㆍ레저"
    assert fix_qmark("│헬스?    │") == "│헬스ㆍ    │"
    assert fix_qmark("환불되나요? 네") == "환불되나요? 네"


def test_outline_glued_numbers_and_subitems():
    text = ("I. 목적 및 구성1. 목적이 지침은 법 제21조에 따른다.2. 구성 등이 지침은 두 부분이다.　"
            "Ⅱ. 일반사항1. 청약철회 방해 등 관련가. 법 제18조제9항과 관련하여 예시하면 다음과 같다."
            "(1) 반품비를 과하게 받는 경우나. 법 제21조제1항제1호의 규정과 관련하여 금지한다.다. 사업자는 표시한다.")
    docs = {d["doc_id"]: d for d in parse_outline("ecomg", "지침", text)}
    assert set(docs) == {"ecomg-p1", "ecomg-p2", "ecomg-g1"}
    g1 = docs["ecomg-g1"]
    assert g1["title"] == "청약철회 방해 등 관련"
    assert [u["unit_id"] for u in g1["units"]] == ["ecomg-g1-1", "ecomg-g1-2", "ecomg-g1-3"]
    assert g1["units"][1]["text"].startswith("법 제21조제1항제1호")  # '…경우나.' 붙은 항목
    assert "금지한다." in g1["units"][1]["text"]                    # '한다.다.'의 '다.'는 문장 끝


def test_annex_headings_follow_group_order():
    text = "\n".join([
        "Ⅱ. 서비스업 부문",
        "1. 학원ㆍ교습ㆍ레저 서비스 관련",
        " < ①자동차운전학원(1개 업종) >",
        "┏━━━┓", "┃분 쟁 유 형 │해 결 기 준┃", "┃1) 계약해지 │환급┃", "┗━━━┛",
        "1. 배상액의 산정방식",  # 표 사이의 번호 매김 → 순서(다음은 2.)가 안 맞아 업종 묶음이 아님
        "┃ 내용 ┃",
        "2. 연수ㆍ유학ㆍ이민 서비스 관련",
        "┏━━━┓", "┃①해외 어학연수 수속 대행업┃", "┠───┨", "┃분 쟁 유 형 │해 결 기 준┃",
    ])
    labels = [label for label, _ in chunk_annex(text, limit=10_000)]
    assert labels[-1] == "Ⅱ. 서비스업 부문 > 2. 연수ㆍ유학ㆍ이민 서비스 관련 > ①해외 어학연수 수속 대행업"
    assert not any("배상액" in l for l in labels)
    assert any(l.endswith("> ①자동차운전학원(1개 업종)") for l in labels)

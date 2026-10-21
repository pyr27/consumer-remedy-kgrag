"""extract_edges 규칙 회귀 테스트 (실제 코퍼스 사용, DB 불필요).  python -m pytest kg -q"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_edges import DATA, Corpus, extract_unit, skip  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    return Corpus(json.loads((DATA / "legal_corpus.json").read_text(encoding="utf-8")))


def edges(corpus, uid):
    return {(e["src"], e["rel"], e["dst"]) for e in extract_unit(uid, corpus)[0]}


def test_range_expands_articles(corpus):
    got = edges(corpus, "door-52")
    for d in ["door-8", "door-9", "door-10", "door-31", "door-32"]:
        assert any(dst == d for _, _, dst in got), d


def test_override_chain(corpus):
    assert edges(corpus, "ecom-17-3") == {("ecom-17-3", "OVERRIDES", "ecom-17-1"),
                                          ("ecom-17-3", "OVERRIDES", "ecom-17-2")}


def test_delegation_reversed_and_plain_reference(corpus):
    got = edges(corpus, "enf-21")
    assert ("ecom-17-2-6", "DELEGATES_TO", "enf-21") in got
    assert ("enf-21", "REFERS_TO", "ecom-13-2-5") in got


def test_delegation_to_proviso(corpus):
    assert ("ecom-5-1-proviso", "DELEGATES_TO", "enf-4-0") in edges(corpus, "enf-4-0")


def test_law_alias_and_mok(corpus):
    assert ("ecom-2-5-2", "DELEGATES_TO", "enf-2-0") in edges(corpus, "enf-2-0")


def test_bare_paragraph_is_same_article(corpus):
    assert edges(corpus, "door-32-4") == {("door-32-4", "REFERS_TO", "door-32-1"),
                                          ("door-32-4", "REFERS_TO", "door-32-2")}


def test_naeji_range(corpus):  # "제55조제1항 내지 제3항"
    got = edges(corpus, "cons-58-1")
    assert {("cons-58-1", "REFERS_TO", f"cons-55-{i}") for i in (1, 2, 3)} <= got
    assert ("cons-58-1", "REFERS_TO", "cons-58") not in got


def test_chain_after_range(corpus):  # "제7조제1항부터 제3항까지 및 제5항"
    assert ("door-49-1-1", "REFERS_TO", "door-7-5") in edges(corpus, "door-49-1-1")


def test_same_article_inherits_law(corpus):  # "법 제47조 … 같은 조 제2항"
    got = edges(corpus, "consenf-68-1-2")
    assert ("consenf-68-1-2", "REFERS_TO", "consenf-47") not in got
    assert any(dst.startswith("cons-47-2") for _, _, dst in got)


def test_admrul_law_alias(corpus):  # contfee: 「방문판매 등에 관한 법률」(이하 "법") → "법 제N조"는 방문판매법
    got = extract_unit("contfee-1", corpus)[0]
    assert got and all(e["dst"].startswith("door-") for e in got)


def test_installment_delegation(corpus):
    assert any(e["rel"] == "DELEGATES_TO" and e["src"].startswith("inst-")
               for u in corpus.units if u.startswith("instenf-") for e in extract_unit(u, corpus)[0])


def test_uses_term_longest_match_and_hub_cutoff(corpus):
    from extract_edges import extract_terms
    rows, freq = extract_terms(corpus)
    got = {(r["src"], r["dst"]) for r in rows}
    assert ("door-31", "door-2-10") in got                     # '계속거래' → 정의
    assert not any(d == "door-2-1" and "후원방문판매" in corpus.units[s]["text"]
                   and "방문판매" not in corpus.units[s]["text"].replace("후원방문판매", "")
                   for s, d in got)                             # '후원방문판매' 안의 '방문판매'는 제외
    assert not any(d == "ecom-2-5" for _, d in got)            # '소비자'(고빈도)는 제외


def test_no_self_edges(corpus):
    for uid in corpus.units:
        if not skip(uid, corpus):
            assert all(e["dst"] != uid for e in extract_unit(uid, corpus)[0] if e["rel"] != "DELEGATES_TO")

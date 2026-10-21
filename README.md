# Naive RAG 베이스라인 실험

소비자 피해구제(환불·위약금) 도메인에서 **단순 벡터검색 RAG가 어떻게 실패하는지**를
측정하는 대조군 실험이다. 교수님 피드백("Naive RAG로 구현했을 때 실제로 성능이 낮은지,
환각이 발생하는지 현상을 확인해보라")에 대한 답을 만드는 것이 목적이다.

## 구조

```
naive_rag_baseline/
├── evaluate.py            실험 실행 + 자동 채점
├── inspect_chunks.py      청킹 진단 (LLM 호출 없이 실행 가능)
├── fetch_corpus.py        법제처 Open API 수집 (본인 PC에서 실행)
├── merge_corpus.py        수집본 + 분쟁해결기준 병합 및 gold_citation 검증
├── requirements.txt
├── rag/
│   ├── chunking.py        고정 길이 / 항 단위 청킹, 단위별 구간 추적
│   ├── backends.py        임베더(tfidf·openai·local), LLM(mock·openai·anthropic)
│   ├── index.py           코사인 유사도 단순 벡터 인덱스
│   └── pipeline.py        검색 1회 + 생성 1회 (의도적으로 자가검증 없음)
├── data/
│   ├── legal_corpus.json    전자상거래법 제13·15·17·18·19·20조, 시행령 제21조,
│   │                        소비자분쟁해결기준 3개 품목
│   └── test_questions.json  16문항 (1hop 5 / 2hop 6 / 3hop 3 / 범위밖 2)
└── results/               실행 결과 (gitignore 권장)
```

## 실행

```bash
pip install -r requirements.txt

# 0) 코퍼스 수집 — 본인 PC에서 실행 (법제처는 외부 프록시에서 막힘)
export LAW_OC=<본인 이메일의 @ 앞부분>
python fetch_corpus.py --inspect        # 응답 구조 먼저 확인
python fetch_corpus.py                  # data/legal_corpus_fetched.json 생성
python merge_corpus.py                  # 병합 + gold_citation 검증

# 1) 청킹 진단 — API 키 불필요
python inspect_chunks.py
python inspect_chunks.py --chunk-size 150 --overlap 30

# 2) 파이프라인 배관 확인 — API 키 불필요
python evaluate.py --embedder tfidf --llm mock --tag smoke

# 3) 실제 측정 (보고서용) — 아래 3개를 모두 돌려야 비교가 성립한다
export OPENAI_API_KEY=...        # 플레이스홀더가 아니라 실제 키
#
# 크레딧 없이 로컬 모델로 돌리려면 (Ollama 등 OpenAI 호환 API):
#   export OPENAI_BASE_URL=http://localhost:11434/v1
#   export OPENAI_API_KEY=ollama
#   export OPENAI_MODEL=qwen2.5:7b
#   그리고 --embedder local 사용 (pip install sentence-transformers)
python evaluate.py --embedder openai --llm openai --tag naive_fixed
python evaluate.py --embedder openai --llm openai --chunking article --tag naive_article
python evaluate.py --embedder openai --llm openai --tag oracle --oracle
```

세 번째 실행(`--oracle`)이 중요하다. 정답 조문을 강제로 컨텍스트에 넣으므로
검색 재현율이 1.0으로 고정된다. **이 조건에서도 답이 틀린다면 그 실패는
검색이 아니라 추론에서 비롯된 것**이고, GraphRAG가 필요한 가장 강한 근거가 된다.

출력 4종이 `results/`에 생긴다.

| 파일 | 내용 |
|---|---|
| `run_<tag>.json` | 질문별 검색 결과·프롬프트·답변 원본 |
| `metrics_<tag>.csv` | 질문별 자동 지표 |
| `summary_<tag>.json` | 전체 요약 |
| `scoring_<tag>.csv` | **수동 채점 시트** — 정확성·환각을 직접 기입 |

## 측정 항목

자동으로 계산되는 것:

- `recall_strict` — 정답 근거 조문이 **온전히** 검색된 비율
- `recall_loose` — 청크 경계에 걸쳐서라도 검색된 비율
- `precision@k` — 검색된 청크 중 정답 근거를 포함한 비율
- `cited_unknown_articles` — 코퍼스에 존재하지 않는 조문을 인용 → **환각**
- `out_of_corpus` 질문에서 기권하지 않음 → **환각**

사람이 채점해야 하는 것 (`scoring_<tag>.csv`):

- `answer_correctness` 0/1/2 — 결론이 맞는가
- `hallucination` 0/1/2 — 근거가 사실인가
- 누락된 쟁점 — 어느 참조 단계를 건너뛰었는가

## 핵심 설계: hop 수 통제

이 실험의 성패는 질문셋에 달려 있다. 단일 조문으로 답이 나오는 질문만 넣으면
Naive RAG도 잘 맞춰서 "성능이 낮다"를 보일 수 없다. 그래서 **정답 도달에 필요한
참조 단계 수(hops)를 통제**했다.

- **1hop (5문항)** — 대조군. Naive RAG도 맞춰야 정상
- **2hop (6문항)** — 본문과 단서, 또는 법과 기준을 연결해야 함
- **3hop (3문항)** — 결론이 뒤집히는 케이스
- **범위밖 (2문항)** — 기권해야 정상. 답을 지어내면 환각

가장 중요한 문항은 **Q12**다.

> 주문제작 책상에 '환불 불가' 서면 동의까지 했는데, 배송된 제품이 상세페이지와 다르다.

법 제17조제2항제6호 → 시행령 제21조까지만 추적하면 **"환불 불가"**라는 답이 나온다.
제17조제3항("제1항 및 제2항에도 불구하고")까지 도달해야 **"환불 가능"**으로 결론이
뒤집힌다. 관계 추론 없이는 정반대 답을 내놓게 되는 구조다. Q13도 같은 형태다.

## 현재까지 확인된 것 (tfidf + mock, 배관 확인 수준)

청킹 진단 — 고정 500자 기준:

```
다단계 질문 9개 중 단일 청크로 근거가 모두 커버되는 경우: 2개 (22.2%)
다단계 질문 평균 최소 필요 청크 수: 2.11
```

다단계 질문의 **78%는 한 번의 검색으로 근거를 모을 수 없는 구조**라는 뜻이다.
이것이 Naive RAG가 다단계 참조에서 실패하는 구조적 원인이며,
LLM 성능과 무관하게 검색 단계에서 이미 결정된다.

항 단위 청킹에서는 재현율 차이가 더 선명하게 드러난다:

| 청킹 | 1hop 재현율 | 다단계 재현율 |
|---|---|---|
| 고정 500자 | 0.60 | 0.82 |
| 항 단위 | 0.60 | **0.39** |

청크가 작고 많아질수록 다단계 질문의 근거 회수가 급격히 어려워진다.

## 알려진 한계 — 측정 전에 반드시 처리할 것

1. **코퍼스가 작다.** 문서 10개, 청크 12개뿐이라 `top_k=4`면 코퍼스의 33%를 검색하는
   셈이 된다. 재현율이 비현실적으로 높게 나오므로, 2단계에서 법제처 Open API로
   조문을 확충한 뒤 다시 측정해야 한다. 그 전까지 수치를 보고서에 쓰면 안 된다.

2. **조문 텍스트를 검증해야 한다.** 현재 텍스트는 실험용으로 정리한 것이다.
   법제처 원문과 대조하고 `legal_corpus.json`의 `_note`를 지울 것.

3. **TF-IDF는 의미 검색이 아니다.** "임베딩이 나빠서 실패한 것 아니냐"는 반박을
   막으려면 최종 측정은 반드시 `--embedder openai`(또는 `local`)로 해야 한다.

## 진행 상황

| 단계 | 상태 |
|---|---|
| 실험 하네스 (코드) | 완료 |
| 평가 질문셋 설계 (16문항, hop 통제) | 완료 |
| oracle retrieval 조건 | 완료 |
| 법제처 수집 스크립트 | 완료 (실행은 본인 PC에서) |
| 코퍼스 확충 | **미완** — 현재 10개 문서 뼈대뿐 |
| 조문 원문 검증 | **미완** |
| 본 측정 (openai) | **미완** — 지금까지는 배관 확인만 |
| 수동 채점 | **미완** |
| 결과 정리 | **미완** |

현재 README에 적힌 수치는 전부 `tfidf + mock` 조합이라 **배관이 돌아간다는 것만
보여줄 뿐 아무것도 측정하지 않았다.** 보고서나 교수님 보고에 쓰면 안 된다.

## 다음 단계

- [x] `fetch_corpus.py`로 코퍼스 확충 (전자상거래법만으로 조 58개 / 단위 392개 확인)
- [ ] `merge_corpus.py`로 병합하고 gold_citation 검증 통과시키기
- [ ] 소비자분쟁해결기준 텍스트를 공정위 고시 원문과 대조
- [ ] openai 임베더·LLM으로 3개 조건 측정 (fixed / article / oracle)
- [ ] `scoring_*.csv` 수동 채점 (16문항 × 3조건)
- [ ] hop별 정확도 곡선 작성 → 기초조사서 7장 평가 계획에 반영

---

# 질문셋 v2 (2026-10)

v1 문항(14개)은 AI가 작성한 것이라 정답이 검증되지 않았습니다. v2는 정답의 출처를 분리합니다.

| 세트 | 파일 | 정답 출처 | 도구 |
|---|---|---|---|
| A 실사용 | questions_faq.jsonl | 한국소비자원 FAQ 공식 답변 | faq_tool.py |
| B 다중홉 | questions_generated.jsonl | KoBLEX 방식 생성 → 자동검증 → 사람 검수 | generate_questions.py |
| C 라우팅 | questions_routing.jsonl (10) | 초안(verified_by=draft) — 검수 필요 | 직접 작성 |
| D 범위 밖 | questions_oos.jsonl (5) | 거절·안내가 정답 | 직접 작성 |

## 실행 순서 (PowerShell)

```powershell
# 1) 코퍼스 확장 (소비자기본법·방문판매법 추가) → 병합
python fetch_corpus.py --oc debora1628
python merge_corpus.py

# 1-1) 소비자분쟁해결기준(공공데이터 21개 품종) — data/std_kca.json이 이미 들어 있으면 생략
python build_std.py 소비자분쟁해결기준_품목별해결기준.csv
python merge_corpus.py                             # 법령 + 분쟁해결기준 공식본 병합

# 2) A세트: 소비자원 품목별 FAQ — 답변이 인용한 조문을 그대로 정답 근거로 사용
python faq_tool.py cite 품목별FAQ.csv              # → data/v2/kca_cite_sample.csv (검수용 60문항)
python faq_tool.py proc 피해구제분쟁조정FAQ.csv     # → data/v2/questions_proc.jsonl (절차 50문항)
#    kca_cite_sample.csv: contexts_추정 열이 있는 행만 항 번호 확인, keep=Y 표시
python faq_tool.py build data/v2/kca_cite_sample.csv

# 3) B세트: seed 추출 → 소량 시험 생성 → 검수 → 변환
python generate_questions.py seeds --show
python generate_questions.py generate --limit 5      # 먼저 5개로 품질 확인
python generate_questions.py generate                # 괜찮으면 전체
#    data/v2/gen_review.csv 에서 keep=Y 표시, 필요 시 수정_ 열에 고친 문장
python generate_questions.py build data/v2/gen_review.csv

# 4) 측정
python evaluate_v2.py --cond naive  --tag v2_naive
python evaluate_v2.py --cond oracle --tag v2_oracle
python evaluate_v2.py --cond cite   --tag v2_cite
$env:OPENAI_MODEL="gpt-4o"; python evaluate_v2.py --cond full --tag v2_full_4o
```

## evaluate_v2 조건

- **naive / oracle**: v1과 같은 방식 (검색 실패 vs 추론 실패 분리)
- **cite**: 각 조항에 `[unit_id]`를 붙이고 ID로만 인용하게 강제 → 조문 번호 오귀속이 인용 방식만으로 줄어드는지
- **full**: 검색 없이 법령 전체를 상위 모델에 주입 → "법령을 통째로 넣으면 되지 않나"에 대한 반론 근거

출력 `results/v2/summary_<tag>.json`의 `cite_gold_hit_mean`, `answers_with_invalid_ids`(존재하지 않는 ID 인용 = 환각), `answers_citing_outside_context`를 봅니다. 라우팅·범위 밖 문항은 `scoring_<tag>.csv`의 안내적절성 열로 수동 채점합니다.

# 법령 지식그래프 (본 개발)

## 1. Neo4j 실행 (Docker)
```powershell
copy .env.example .env          # 비밀번호를 바꾸고 싶으면 .env에서 수정
docker compose up -d            # 처음 한 번은 이미지 다운로드로 1~2분
docker compose ps               # STATUS가 running(healthy)인지 확인
```
브라우저 http://localhost:7474 → 아이디 neo4j / 비밀번호(.env의 NEO4J_PASSWORD)

## 2. 그래프 적재
```powershell
pip install -r requirements.txt
python kg/build_graph.py --dry-run          # DB 없이 노드·관계 수 확인
python kg/build_graph.py --wipe             # 구조만 적재 (무료, 수 초)
python kg/build_graph.py --wipe --embed     # + 임베딩 (첫 회만 OpenAI 비용, 이후 캐시)
```
의미 관계(REFERS_TO·DELEGATES_TO·OVERRIDES)는 `data/edges.csv`에서 `approved=y`인 것만 적재된다.
```powershell
python kg/extract_edges.py                  # 관계 후보 → data/edges.csv (기존 검토 결과 보존)
python kg/extract_edges.py --show ecom-17-3 # 조항 하나에서 뽑힌 관계 확인
python -m pytest kg -q                      # 추출 규칙 회귀 테스트
python kg/check_paths.py                    # 합격 기준: FAQ 근거 쌍 strict ≤4 hop
```
그래프는 `data/legal_corpus.json`(+ `data/edges.csv`, `data/categories.csv`)에서 언제든 다시 만들 수 있다.

## 3. 검색 (2주차)
```powershell
python kg/retrieve.py "헬스장 1년 계약했는데 3개월 만에 그만두면 위약금은?"   # 라우팅·근거 묶음 출력
python kg/retrieve.py --route continuous "…"                              # 라우팅 지정 (none = 전체)
python kg/eval_retrieval.py --route llm --tag r3                          # 42문항 검색 평가 (생성 없음)
python kg/eval_retrieval.py --route oracle --tag r3                       # 라우팅을 정답으로 준 상한
```
LLM 라우팅·재작성 결과는 `results/kg/route_cache.json`에 캐시된다(같은 질문 재실행 비용 0). DB를 지워도(`docker compose down -v`) 잃는 것은 없다.

## 3. 브라우저에서 확인해 볼 쿼리
```cypher
// 전자상거래법 제17조 전체 구조
MATCH (a:Article {doc_id:'ecom-17'})-[:HAS_UNIT]->(u) OPTIONAL MATCH (u)-[r:SUB_OF|PROVISO_OF]->(p)
RETURN a, u, r, p

// 키워드 검색 (cjk 분석기)
CALL db.index.fulltext.queryNodes('unit_text', '청약철회 7일') YIELD node, score
RETURN node.unit_id, node.path, score LIMIT 10
```

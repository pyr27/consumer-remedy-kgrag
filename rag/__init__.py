"""Naive RAG 베이스라인 실험용 최소 구현.

GraphRAG 도입 전, 단순 벡터검색 RAG가 소비자 피해구제 도메인에서
어떤 방식으로 실패하는지 측정하기 위한 대조군이다.
의도적으로 관계 추론·자가검증 없이 '검색 → 생성' 한 번만 수행한다.
"""

__all__ = ["backends", "chunking", "index", "pipeline"]

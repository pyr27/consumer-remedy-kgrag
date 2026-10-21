"""임베딩 모델과 LLM 백엔드.

API 키 없이도 파이프라인 전체를 돌려볼 수 있도록 ``tfidf`` 임베더와
``mock`` LLM을 기본값으로 둔다. 보고서에 쓸 실제 수치는 ``openai`` /
``anthropic`` 백엔드로 측정한다.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

import numpy as np


# --------------------------------------------------------------------------- #
# 임베더
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """문자 n-gram TF-IDF.

    한국어 형태소 분석기 없이도 동작하므로 스모크 테스트에 적합하다.
    의미 기반 검색이 아니므로 '임베딩이 나빠서 실패했다'는 반박을 피하려면
    최종 측정은 반드시 openai/local 임베더로 한 번 더 돌려야 한다.
    """

    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vec.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")
        return np.asarray(self._vec.transform(texts).todense(), dtype=np.float32)


def _openai_client():
    """OpenAI 호환 클라이언트.

    OPENAI_BASE_URL을 지정하면 Ollama·Groq·LM Studio 등 OpenAI 호환 API를
    그대로 쓸 수 있다. 크레딧 없이 로컬 모델로 실험할 때 사용한다.
    """
    from openai import OpenAI

    base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    key = os.getenv("OPENAI_API_KEY") or "not-needed"
    return OpenAI(base_url=base, api_key=key) if base else OpenAI(api_key=key)


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self._client = _openai_client()
        self._model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    def fit(self, texts: list[str]) -> None:  # 학습 불필요
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):  # 배치
            resp = self._client.embeddings.create(
                model=self._model, input=texts[i : i + 64]
            )
            out.extend(d.embedding for d in resp.data)
        return np.asarray(out, dtype=np.float32)


class LocalEmbedder:
    """sentence-transformers 기반. 한국어 법률 텍스트에는 BGE-m3 계열이 무난하다."""

    name = "local"

    def __init__(self, model: str = "BAAI/bge-m3") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def fit(self, texts: list[str]) -> None:
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=False), dtype=np.float32
        )


def get_embedder(name: str) -> Embedder:
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
    if name == "local":
        return LocalEmbedder()
    raise ValueError(f"알 수 없는 임베더: {name}")


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
class LLM(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


class MockLLM:
    """키 없이 배관만 확인하는 더미.

    검색된 청크에서 조문 번호를 긁어 그대로 인용하는, 가장 단순한 형태의
    '검색 결과 나열형' 답변을 흉내낸다. 모른다고 말하는 기능이 없으므로
    범위 밖 질문에서는 자동 환각 플래그가 켜져야 정상이다.
    """

    name = "mock"
    _CITE = re.compile(r"제\s*\d{1,3}\s*조(?:\s*의\s*\d{1,2})?(?:제\d{1,2}항)?(?:제\d{1,2}호)?")

    def generate(self, prompt: str) -> str:
        body = prompt.split("[참고 자료]", 1)[-1]
        cites: list[str] = []
        for m in self._CITE.finditer(body):
            c = m.group(0)
            if c not in cites:
                cites.append(c)
        head = ", ".join(cites[:3]) if cites else "관련 규정"
        return (
            f"{head}에 따르면 해당 사안은 규정이 정하는 바에 따라 처리됩니다. "
            "구체적인 가부는 계약 조건과 사실관계에 따라 달라질 수 있습니다."
        )


_SYSTEM = (
    "당신은 한국 소비자 피해구제 상담을 돕는 어시스턴트입니다. "
    "아래 [참고 자료]에 실제로 적힌 내용만 근거로 답하십시오. "
    "참고 자료에 근거가 없으면 추측하지 말고 "
    "'제공된 자료만으로는 확인할 수 없습니다'라고 답하십시오. "
    "답변에는 근거가 된 조문 번호를 명시하십시오."
)


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self._client = _openai_client()
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def generate(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def get_llm(name: str) -> LLM:
    if name == "mock":
        return MockLLM()
    if name == "openai":
        return OpenAILLM()
    if name == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"알 수 없는 LLM: {name}")

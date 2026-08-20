"""
벡터 DB 기반 알약 유사도 검색.

SQL 정확 매칭이 NO_MATCH일 때 폴백으로 사용한다.
OpenAI text-embedding-3-small 로 알약 특성을 임베딩하고,
numpy 내적(코사인 유사도)으로 가장 유사한 후보를 찾는다.

인덱스는 최초 1회 빌드 후 pickle 캐시에 저장하여 재기동 시 즉시 로드한다.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

from app.models import PillApiItem, VisionExtractionResult

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "pill_vectors.pkl"
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536
_BATCH_SIZE = 2048
_SIMILARITY_THRESHOLD = 0.75


def _item_to_text(item: PillApiItem) -> str:
    """DB 알약 한 건을 임베딩용 텍스트로 변환한다."""
    parts: list[str] = []

    front = (item.print_front or "").strip()
    parts.append(f"앞면각인: {front}" if front else "앞면각인: 없음")

    back = (item.print_back or "").strip()
    parts.append(f"뒷면각인: {back}" if back else "뒷면각인: 없음")

    shape = (item.drug_shape or "").strip()
    parts.append(f"모양: {shape}" if shape else "모양: 미상")

    color1 = (item.color_class1 or "").strip()
    color2 = (item.color_class2 or "").strip()
    if color1 and color2:
        parts.append(f"색상: {color1}, {color2}")
    elif color1:
        parts.append(f"색상: {color1}")
    else:
        parts.append("색상: 미상")

    has_line = bool(
        (item.line_front and item.line_front.strip() and item.line_front.strip() != "없음")
        or (item.line_back and item.line_back.strip() and item.line_back.strip() != "없음")
    )
    parts.append(f"분할선: {'있음' if has_line else '없음'}")

    return " | ".join(parts)


def vision_to_query_text(vision: VisionExtractionResult) -> str:
    """Vision 추출 결과를 검색 쿼리 텍스트로 변환한다."""
    parts: list[str] = []

    front = (vision.print_front or "").strip()
    parts.append(f"앞면각인: {front}" if front else "앞면각인: 없음")

    back = (vision.print_back or "").strip()
    parts.append(f"뒷면각인: {back}" if back else "뒷면각인: 없음")

    shape = (vision.drug_shape or "").strip()
    parts.append(f"모양: {shape}" if shape else "모양: 미상")

    color = (vision.color_class1 or "").strip()
    parts.append(f"색상: {color}" if color else "색상: 미상")

    if vision.score_line is not None:
        parts.append(f"분할선: {'있음' if vision.score_line else '없음'}")
    else:
        parts.append("분할선: 미상")

    return " | ".join(parts)


class VectorPillStore:
    """알약 임베딩 벡터 스토어."""

    # 모듈 레벨 함수를 클래스 메서드로도 접근 가능하게 노출
    vision_to_query_text = staticmethod(vision_to_query_text)

    def __init__(self, openai_client: AsyncOpenAI):
        self._openai = openai_client
        self._items: list[PillApiItem] = []
        self._matrix: Optional[np.ndarray] = None  # (N, 1536) float32
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def build(self, items: list[PillApiItem]) -> None:
        """알약 목록을 임베딩하여 인덱스를 구축한다. 캐시가 유효하면 로드만 한다."""
        if not items:
            logger.warning("벡터 인덱스 빌드 스킵: 알약 데이터가 비어 있습니다")
            return

        self._items = items
        data_hash = self._compute_hash(items)

        # 캐시 확인
        if self._try_load_cache(data_hash):
            logger.info("벡터 인덱스 캐시 로드 완료: %d건", len(self._items))
            self._ready = True
            return

        # 새로 빌드
        logger.info("벡터 인덱스 빌드 시작: %d건", len(items))
        texts = [_item_to_text(item) for item in items]
        embeddings = await self._batch_embed(texts)
        self._matrix = np.array(embeddings, dtype=np.float32)
        self._save_cache(data_hash)
        self._ready = True
        logger.info("벡터 인덱스 빌드 완료: %d건", len(items))

    async def search(
        self, query_text: str, top_k: int = 5
    ) -> list[tuple[PillApiItem, float]]:
        """쿼리 텍스트와 가장 유사한 알약을 반환한다.

        Returns:
            (PillApiItem, similarity_score) 튜플 리스트. 임계값(0.75) 이상만 포함.
        """
        if not self._ready or self._matrix is None:
            return []

        query_embedding = await self._embed_single(query_text)
        query_vec = np.array(query_embedding, dtype=np.float32)

        # text-embedding-3-small은 L2 정규화되어 있으므로 내적 = 코사인 유사도
        similarities = self._matrix @ query_vec  # (N,)

        # 임계값 필터링 + 상위 top_k
        mask = similarities >= _SIMILARITY_THRESHOLD
        if not mask.any():
            return []

        indices = np.where(mask)[0]
        scores = similarities[indices]

        # 상위 top_k 정렬
        if len(indices) > top_k:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            indices = indices[top_indices]
            scores = scores[top_indices]

        # 점수 내림차순 정렬
        order = np.argsort(-scores)
        indices = indices[order]
        scores = scores[order]

        return [(self._items[idx], float(scores[i])) for i, idx in enumerate(indices)]

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트를 배치로 임베딩한다."""
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            batch_num = start // _BATCH_SIZE + 1
            total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
            logger.info("임베딩 배치 %d/%d (%d건)", batch_num, total_batches, len(batch))

            response = await self._openai.embeddings.create(
                model=_EMBEDDING_MODEL, input=batch
            )
            # API 응답은 index 순서대로 정렬되어 있지 않을 수 있으므로 정렬
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

        return all_embeddings

    async def _embed_single(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩한다."""
        response = await self._openai.embeddings.create(
            model=_EMBEDDING_MODEL, input=[text]
        )
        return response.data[0].embedding

    @staticmethod
    def _compute_hash(items: list[PillApiItem]) -> str:
        """item_seq 목록 기반 MD5 해시."""
        seqs = sorted(item.item_seq or "" for item in items)
        return hashlib.md5("|".join(seqs).encode()).hexdigest()

    def _try_load_cache(self, data_hash: str) -> bool:
        """캐시 파일이 유효하면 로드한다."""
        if not _CACHE_PATH.exists():
            return False
        try:
            with open(_CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            if cache.get("hash") != data_hash:
                logger.info("캐시 해시 불일치 — 재빌드합니다")
                return False
            matrix = cache.get("matrix")
            if matrix is None or matrix.shape[0] != len(self._items):
                logger.info("캐시 행렬 크기 불일치 — 재빌드합니다")
                return False
            self._matrix = matrix
            return True
        except Exception:
            logger.warning("캐시 로드 실패 — 재빌드합니다", exc_info=True)
            return False

    def _save_cache(self, data_hash: str) -> None:
        """임베딩 행렬과 해시를 pickle 캐시로 저장한다."""
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_CACHE_PATH, "wb") as f:
                pickle.dump({"hash": data_hash, "matrix": self._matrix}, f)
            logger.info("벡터 캐시 저장 완료: %s", _CACHE_PATH)
        except Exception:
            logger.warning("벡터 캐시 저장 실패", exc_info=True)

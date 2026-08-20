"""
증상 기반 추천(app/symptom/service.py)과 실제 Gemini 호출(gemini_client.py) 사이의
계약(타입)만 정의한다.

app/vision/types.py와 같은 이유로 이 파일은 google-genai를 import하지 않는다 —
service.py는 이 파일에만 의존하게 해서, 테스트에서 실제 API 키 없이도 가짜
extract_fn/suggest_fn을 주입해 순수 로직(공식 DB 검증, 연령/증상 필터링)만
검증할 수 있게 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Protocol


@dataclass(frozen=True)
class SymptomExtraction:
    """증상/발생시각 STT 텍스트에서 뽑아낸 구조화된 결과."""

    symptoms: list[str] = field(default_factory=list)  # SymptomType enum 값들 (app.recommendation.SYMPTOM_TYPE_MAP 키)
    onset_at: Optional[datetime] = None
    onset_confidence: str = "UNKNOWN"  # HIGH | LOW | UNKNOWN


class ExtractSymptomsFn(Protocol):
    """STT 텍스트(증상 발화 + 발생시각 발화) -> SymptomExtraction."""

    async def __call__(
        self,
        *,
        symptom_text: str,
        onset_text: str,
        reference_time: datetime,
    ) -> SymptomExtraction: ...


class SuggestCandidateDrugsFn(Protocol):
    """증상(+나이/성별) -> 검증되지 않은 OTC 제품명 후보 목록(최대 3개)."""

    async def __call__(
        self,
        *,
        symptoms: list[str],
        gender: Optional[str],
        birth_date: Optional[date],
    ) -> list[str]: ...

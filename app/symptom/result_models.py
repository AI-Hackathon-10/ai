"""증상 기반(사진 없이) 약 추천 API의 요청/응답 wire 모델."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.identify_result import Official, Recommendation
from app.models import UserInfo
from app.symptom.service import SymptomRecommendResult


class SymptomVoiceInput(BaseModel):
    """CLOVA Speech 등으로 이미 텍스트 변환된 결과를 받는다 — 오디오 자체는 이
    엔드포인트의 책임이 아니다. 실시간 스트리밍 인식(gRPC)은 별도 클라이언트가
    처리하고, 최종 확정된 텍스트만 여기로 넘어온다."""

    symptom_text: str = Field(..., alias="symptomText", description="증상 발화를 STT로 변환한 텍스트")
    onset_text: str = Field(..., alias="onsetText", description="증상 발생 시각 발화를 STT로 변환한 텍스트")

    model_config = ConfigDict(populate_by_name=True)


class SymptomRecommendRequest(BaseModel):
    user: UserInfo
    voice: SymptomVoiceInput

    model_config = ConfigDict(populate_by_name=True)


class CandidateDrugResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    item_seq: Optional[str] = Field(default=None, alias="itemSeq")
    item_name: Optional[str] = Field(default=None, alias="itemName")
    image_url: Optional[str] = Field(default=None, alias="imageUrl")
    official: Official
    recommendation: Recommendation
    # 사진 스캔으로 확정된 게 아니라 AI가 증상만으로 제안한 후보임을 프론트가
    # 구분해서 보여주도록 표시한다(예: "AI 추천" 배지).
    source: str = "AI_SUGGESTED"


class SymptomRecommendResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    parsed_symptoms: list[str] = Field(default_factory=list, alias="parsedSymptoms")
    symptom_started_at: Optional[datetime] = Field(default=None, alias="symptomStartedAt")
    onset_confidence: str = Field(default="UNKNOWN", alias="onsetConfidence")
    candidates: list[CandidateDrugResponse] = Field(default_factory=list)
    notice: Optional[str] = None
    disclaimer: str = (
        "AI가 증상을 바탕으로 제안한 후보이며 실제 진단이 아닙니다. "
        "복용 전 약사 또는 의사와 상담하고, 반드시 포장지의 표시사항을 확인하세요."
    )


def build_symptom_recommend_response(result: SymptomRecommendResult) -> SymptomRecommendResponse:
    return SymptomRecommendResponse(
        parsed_symptoms=result.parsed_symptoms,
        symptom_started_at=result.symptom_started_at,
        onset_confidence=result.onset_confidence,
        candidates=[
            CandidateDrugResponse(
                item_seq=c.item_seq,
                item_name=c.item_name,
                image_url=c.image_url,
                official=c.official,
                recommendation=c.recommendation,
            )
            for c in result.candidates
        ],
        notice=result.notice,
    )

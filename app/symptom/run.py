"""FastAPI 라우터에서 호출할 진입점. 실제 Gemini 함수를 서비스에 주입한다.

app/vision/run.py와 같은 역할 — 라우터는 이 모듈만 알면 되고, 테스트는
app.symptom.service.recommend_drugs_from_voice에 가짜 extract_fn/suggest_fn을
직접 주입해서 Gemini 없이 검증한다.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from app.drug_detail_service import DrugDetailService
from app.symptom.gemini_client import extract_symptoms_and_onset, suggest_candidate_drug_names
from app.symptom.service import SymptomRecommendResult, recommend_drugs_from_voice


async def run_symptom_recommendation(
    *,
    symptom_text: str,
    onset_text: str,
    gender: Optional[str],
    birth_date: Optional[date],
    detail_service: DrugDetailService,
) -> SymptomRecommendResult:
    return await recommend_drugs_from_voice(
        symptom_text=symptom_text,
        onset_text=onset_text,
        gender=gender,
        birth_date=birth_date,
        detail_service=detail_service,
        extract_fn=extract_symptoms_and_onset,
        suggest_fn=suggest_candidate_drug_names,
    )

"""
증상 음성 인식(STT) 텍스트 -> 구조화된 증상/발생시각 추출, 그리고 증상 기반
일반의약품(OTC) 후보 이름 제안에 쓰는 Gemini 응답 스키마 + 프롬프트.

⚠️ 두 호출 모두 "여기서 최종 사실을 만들어내지 않는다"는 프로젝트 원칙
(기획서: "AI가 약 자체를 추측하는 대신 Vision AI + 식약처 공식 데이터를 결합")을
그대로 따른다.
- 증상/시각 추출: 발화에 명확히 없는 내용은 절대 만들어내지 않는다(없으면 빈
  배열/ null). symptoms는 반드시 고정된 enum 목록 중에서만 고르게 강제한다.
- 약 후보 제안: 여기서 나온 이름은 최종 결과가 아니라 검색 키워드 후보일
  뿐이며, app/symptom/service.py가 e약은요 공식 API로 실존을 재확인하지
  못한 후보는 전부 버린다.
"""
from __future__ import annotations

from typing import Optional

from app.recommendation import SYMPTOM_TYPE_MAP

SYMPTOM_TYPES = list(SYMPTOM_TYPE_MAP.keys())

# -- 1) 증상/발생시각 추출 -----------------------------------------------------

SYMPTOM_EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "symptoms": {
            "type": "array",
            "items": {"type": "string", "enum": SYMPTOM_TYPES},
        },
        "onset_at": {
            "type": "string",
            "nullable": True,
            "description": "ISO 8601 형식의 절대 시각 (예: 2026-08-20T18:30:00+09:00)",
        },
        "onset_confidence": {"type": "string", "enum": ["HIGH", "LOW", "UNKNOWN"]},
    },
    "required": ["symptoms", "onset_at", "onset_confidence"],
}

SYMPTOM_EXTRACTION_SYSTEM_PROMPT = f"""당신은 사용자의 음성을 텍스트로 변환한 발화에서
증상과 증상이 시작된 시각만 정확히 추출하는 전문가입니다. 절대 추측하지 마세요.

[증상 규칙]
1. 아래 목록에 있는 증상 중, 발화에 명확히 언급되었거나 동의어로 확실히 판단되는
   것만 골라 symptoms 배열에 넣으세요.
   - 목록: {", ".join(SYMPTOM_TYPES)}
2. 목록에 없는 증상이거나 애매하면 절대 만들어 넣지 마세요. 아무 증상도 확실하지
   않으면 빈 배열([])을 반환하세요.

[발생 시각 규칙]
1. 사용자에게 "지금 시각(reference_time)"이 함께 주어집니다. "3시간 전부터",
   "어제 저녁부터" 같은 상대 표현은 reference_time을 기준으로 계산해서 절대
   시각(ISO 8601)으로 변환하세요.
2. 시각을 특정할 단서가 발화에 전혀 없으면(예: "그냥 아파요") onset_at은 null,
   onset_confidence는 "UNKNOWN"으로 답하세요.
3. 상대/절대 표현은 있지만 정확한 시:분까지는 알 수 없는 경우(예: "어제부터")는
   합리적으로 추정 가능한 시각(예: 어제 00:00)으로 채우고 onset_confidence는
   "LOW"로 답하세요.
4. 명확한 시각 표현이 있으면 onset_confidence는 "HIGH"로 답하세요.

스키마에 없는 필드는 절대 추가하지 마세요.
"""


def build_symptom_extraction_prompt(*, symptom_text: str, onset_text: str, reference_time_iso: str) -> str:
    return (
        f"지금 시각(reference_time): {reference_time_iso}\n\n"
        f"[증상 발화 텍스트]\n{symptom_text}\n\n"
        f"[증상 발생 시각 발화 텍스트]\n{onset_text}\n\n"
        "위 두 발화에서 증상과 발생 시각을 규칙에 따라 추출하세요."
    )


# -- 2) 증상 기반 OTC 후보 이름 제안 --------------------------------------------

DRUG_CANDIDATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        }
    },
    "required": ["candidates"],
}

DRUG_CANDIDATE_SYSTEM_PROMPT = """당신은 국내 약국에서 처방전 없이 구매 가능한 일반의약품(OTC)의
정확한 제품명만 제안하는 보조자입니다.

규칙:
1. 실제로 존재하는 국내 일반의약품 제품명만 제안하세요. 존재를 확신할 수 없는
   이름은 절대 만들어내지 마세요.
2. 여기서 제안한 이름은 최종 결과가 아니라, 이후 공식 의약품 데이터베이스에서
   검색해 실존 여부와 상세정보를 검증하기 위한 검색어 후보일 뿐입니다. 확신이
   없으면 후보 개수를 줄이세요(0개도 가능합니다).
3. 최대 5개까지, 증상에 직접적으로 대응하는 성분·효능의 제품만 제안하세요.
4. 사용자의 나이·성별 정보가 주어지면 명백히 부적합한 제품(예: 소아용 시럽을
   성인에게 제안, 또는 그 반대)은 제안하지 마세요. 단, 최종 연령/성별 적합성
   판단은 이후 단계에서 공식 데이터로 다시 검증되므로, 여기서는 명백한
   부적합만 걸러내면 됩니다.

스키마에 없는 필드는 절대 추가하지 마세요.
"""


def build_drug_candidate_prompt(
    *, symptoms_ko: list[str], age: Optional[int], gender_ko: Optional[str]
) -> str:
    who = "정보 없음"
    if age is not None and gender_ko:
        who = f"{age}세 {gender_ko}"
    elif age is not None:
        who = f"{age}세"
    elif gender_ko:
        who = gender_ko
    symptoms_text = ", ".join(symptoms_ko) if symptoms_ko else "정보 없음"
    return f"사용자: {who}\n증상: {symptoms_text}\n위 증상에 맞는 일반의약품 후보를 제안하세요."

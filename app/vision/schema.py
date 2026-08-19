"""
Gemini Vision에 요청할 구조화된 출력 스키마 + 프롬프트.

⚠️ PILL_COLORS / PILL_SHAPES 는 낱알식별 API가 쓰는 고정 코드값과 반드시 일치해야
매칭이 된다(app/pill_matching_service.py 가 이 값을 그대로 색상/모양 쿼리 파라미터로
쓴다). 아래 목록은 통용되는 값 기준의 추정치이니, 활용가이드에서 실제 코드값 전체
목록과 대조해서 필요하면 이 상수만 교체할 것.

⚠️ nullable 필드 표기법 주의: 표준 JSON Schema/OpenAPI 3.1 방식인
`{"type": ["string", "null"]}`는 Gemini의 response_schema(google.genai.types.Schema)가
받아들이지 않는다(`type`은 배열이 아니라 단일 값만 허용). Gemini 쪽은 OpenAPI 3.0 스타일인
`{"type": "string", "nullable": true}`를 써야 한다. 아래 스키마는 이 형식으로 맞춰져 있다.
"""
from __future__ import annotations

PILL_COLORS = [
    "하양", "노랑", "주황", "분홍", "빨강", "갈색", "연두", "초록",
    "청록", "파랑", "남색", "자주", "보라", "회색", "검정", "투명",
]

PILL_SHAPES = [
    "원형", "타원형", "장방형", "삼각형", "사각형", "마름모형",
    "오각형", "육각형", "팔각형", "반원형", "기타",
]

# Gemini generate_content 의 response_schema 로 그대로 전달한다 (Gemini Schema 서브셋 형식 —
# nullable 필드는 "type": [..., "null"] 이 아니라 "nullable": True 로 표기해야 한다).
PILL_VISION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "print_front": {"type": "string", "nullable": True},
        "print_front_confidence": {"type": "number"},
        "print_back": {"type": "string", "nullable": True},
        "print_back_confidence": {"type": "number"},
        "color_class1": {"type": "string", "enum": PILL_COLORS},
        "color_class2": {"type": "string", "nullable": True, "enum": PILL_COLORS},
        "drug_shape": {"type": "string", "enum": PILL_SHAPES},
        "line_front": {"type": "string", "nullable": True},
        "line_back": {"type": "string", "nullable": True},
        "form_code_name": {"type": "string", "nullable": True},
        "overall_confidence": {"type": "number"},
    },
    "required": [
        "print_front_confidence",
        "print_back_confidence",
        "color_class1",
        "drug_shape",
        "overall_confidence",
    ],
}

SYSTEM_PROMPT = f"""당신은 알약 앞/뒤 사진에서 의약품 식별 정보를 추출하는 전문가입니다.

규칙:
1. 각인(print_front, print_back)은 명확하게 읽었을 때만 값을 채우고, 조금이라도
   확신이 없으면 절대 추측하지 말고 null로 답하세요.
2. print_front_confidence / print_back_confidence 는 각인 인식에 대한 확신도를
   0.0~1.0 사이 숫자로 앞면/뒷면 각각 독립적으로 매기세요. 각인을 못 읽었다면(null)
   그에 맞게 낮은 값을 주세요.
3. color_class1, color_class2, drug_shape 는 반드시 아래 목록의 값 중 하나만 쓰세요.
   목록에 없는 표현(예: "베이지색", "직사각형에 가까운 타원")은 절대 쓰지 마세요.
   - 색상 목록: {", ".join(PILL_COLORS)}
   - 모양 목록: {", ".join(PILL_SHAPES)}
4. 부색상이 없으면(단일 색상 알약) color_class2 는 null로 답하세요.
5. overall_confidence 는 전체 추출 결과에 대한 종합적인 확신도입니다.
6. 색상과 모양은 사진에 알약이 보이는 한 가능한 한 반드시 판단하세요. 이 두 값이
   모두 없으면 이후 단계에서 의약품을 검색할 방법이 전혀 없습니다.
"""


def build_user_prompt(retry_hint: str | None) -> str:
    base = (
        "첫 번째 이미지는 알약 앞면, 두 번째 이미지는(있다면) 알약 뒷면입니다. "
        "위 규칙에 맞춰 알약의 각인, 색상, 모양, 분할선, 제형을 추출하세요."
    )
    if retry_hint:
        base += f"\n\n[재시도 안내] {retry_hint}"
    return base

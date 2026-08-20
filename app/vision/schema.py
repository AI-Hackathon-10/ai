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
#
# confidence 점수는 두지 않는다 — Vision이 확신 없는 필드는 그 자리에서 바로 null로
# 답하게 하고, 그 null을 그대로 신뢰한다(재질의로 채우지 않는다). 재질의(재시도)가
# 없으므로 "애매하면 null"이 "애매한데 억지로 값을 채워 틀리는" 것보다 항상 안전하다.
PILL_VISION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "print_front": {"type": "string", "nullable": True},
        "print_back": {"type": "string", "nullable": True},
        "color_class1": {"type": "string", "nullable": True, "enum": PILL_COLORS},
        "drug_shape": {"type": "string", "nullable": True, "enum": PILL_SHAPES},
        "score_line": {"type": "boolean", "nullable": True},
    },
    "required": ["print_front", "print_back", "color_class1", "drug_shape", "score_line"],
}

SYSTEM_PROMPT = f"""당신은 알약 앞/뒤 사진에서 의약품 식별 정보를 추출하는 전문가입니다.

아래 5개 필드만 1차로 추출하세요. 재질의는 하지 않으므로, 여기서 판단하지 못한
필드는 이후 단계에서도 계속 판단 불가로 남습니다.

- print_front: 앞면 각인
- print_back: 뒷면 각인
- color_class1: 색상
- drug_shape: 모양
- score_line: 분할선(스코어라인) 유무

규칙:
1. 각인(print_front, print_back)은 명확하게 읽었을 때만 값을 채우고, 조금이라도
   확신이 없으면 절대 추측하지 말고 null로 답하세요.
2. color_class1, drug_shape 는 반드시 아래 목록의 값 중 하나만 쓰세요. 목록에 없는
   표현(예: "베이지색", "직사각형에 가까운 타원")은 절대 쓰지 말고, 판단이 애매하면
   추측하지 말고 null로 답하세요.
   - 색상 목록: {", ".join(PILL_COLORS)}
   - 모양 목록: {", ".join(PILL_SHAPES)}
3. score_line 은 분할선이 뚜렷이 보이면 true, 없는 게 뚜렷하면 false, 사진만으로
   판단하기 애매하면 null로 답하세요.
4. 스키마에 없는 필드는 절대 추가하지 마세요.
"""


def build_user_prompt() -> str:
    return (
        "첫 번째 이미지는 알약 앞면, 두 번째 이미지는(있다면) 알약 뒷면입니다. "
        "위 규칙에 맞춰 알약의 각인, 색상, 모양, 분할선을 추출하세요."
    )

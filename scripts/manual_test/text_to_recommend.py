"""
음성 인식(CLOVA/마이크) 없이, 증상/발생시각을 "텍스트로 직접 입력"해서
POST /symptoms/recommend 가 실제로 3~5개의 약을 추천하고, 각 약의 효능/주의사항이
e약은요 공식 데이터와 대조되어 채워지는지, 그리고 그 결과가 온전한 JSON
리포트로 출력되는지 확인하는 스크립트.

voice_to_recommend.py와 하는 일은 같지만(사용자 정보 + 증상 + 발생시각 ->
서버 호출 -> 결과 출력), 마이크/gRPC/CLOVA 관련 코드가 전혀 없다 — 그래서
clova_streaming_client.py가 없어도, grpc가 설치되어 있지 않아도 이 스크립트
하나만으로 바로 실행된다. 음성 인식 파이프라인과 추천 파이프라인을 분리해서
디버깅하기 위한 용도.

사전 조건
---------
1) ai 레포에서 `uvicorn app.main:app --reload`로 서버가 떠 있어야 한다.
   (아래 FASTAPI_URL을 서버 주소/포트에 맞게 필요하면 바꿀 것)
2) 서버 쪽 .env에 GEMINI_API_KEY, MFDS_PILL_SERVICE_KEY가 실제 값으로
   채워져 있어야 한다 — 없으면 candidates가 항상 비어 있거나 503이 난다.

실행: python3 text_to_recommend.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

FASTAPI_URL = "http://127.0.0.1:8000/symptoms/recommend"


def call_recommend_api(*, symptom_text: str, onset_text: str, user: dict) -> dict:
    body = {
        "user": user,
        "voice": {"symptomText": symptom_text, "onsetText": onset_text},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        FASTAPI_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"서버가 {e.code}로 응답했습니다: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"서버({FASTAPI_URL})에 연결하지 못했습니다: {e}. "
            "uvicorn이 떠 있는지, 포트가 맞는지 확인해주세요."
        ) from e


def print_human_summary(result: dict) -> None:
    print("\n=== [사람이 보기 좋은 요약] ===")
    print("인식된 증상:", result.get("parsedSymptoms"))
    print("증상 발생 시각:", result.get("symptomStartedAt"), f"(신뢰도: {result.get('onsetConfidence')})")

    candidates = result.get("candidates") or []
    print(f"추천된 약 개수: {len(candidates)}개")

    if not candidates:
        print("\n추천된 약이 없습니다.")
        print("안내:", result.get("notice"))
        return

    for i, c in enumerate(candidates, start=1):
        rec = c.get("recommendation", {})
        official = c.get("official", {})
        print(f"\n[{i}] {c.get('itemName')}  (source={c.get('source')})")
        print(f"    상태: {rec.get('status')} / 점수: {rec.get('score')} / 신뢰도: {rec.get('confidence')}")
        print(f"    이유: {rec.get('reason')}")
        if official.get("efficacy"):
            print(f"    효능(e약은요): {official.get('efficacy')}")
        if official.get("useMethod"):
            print(f"    용법: {official.get('useMethod')}")
        if rec.get("caution") or official.get("caution"):
            print(f"    주의사항: {rec.get('caution') or official.get('caution')}")
        if official.get("sideEffect"):
            print(f"    부작용: {official.get('sideEffect')}")
        if c.get("imageUrl"):
            print(f"    이미지: {c.get('imageUrl')}")

    print("\n※", result.get("disclaimer"))


def print_json_report(result: dict) -> None:
    print("\n=== [JSON 리포트 원문 — 이게 실제로 프론트/백엔드가 받는 응답이다] ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    print("=== (텍스트 입력) 증상 -> 약 추천 통합 테스트 ===")
    print(f"서버 주소: {FASTAPI_URL}")
    print("※ 이 스크립트는 마이크를 쓰지 않는다 — 증상/발생시각을 직접 타이핑해서")
    print("  추천 로직 자체가 제대로 동작하는지만 빠르게 확인하는 용도.\n")

    user_id = input("사용자 ID [1]: ").strip() or "1"
    name = input("이름 [테스트]: ").strip() or "테스트"
    gender = (input("성별 MALE/FEMALE [FEMALE]: ").strip() or "FEMALE").upper()
    birth_date = input("생년월일 YYYY-MM-DD [1995-01-01]: ").strip() or "1995-01-01"
    user = {"userId": int(user_id), "name": name, "gender": gender, "birthDate": birth_date}

    symptom_text = input("\n증상을 입력하세요 (예: 머리가 아프고 열이 나요): ").strip()
    if not symptom_text:
        print("증상 텍스트가 비어 있습니다. 다시 실행해서 입력해주세요.")
        return

    onset_text = input("증상이 언제부터 시작됐는지 입력하세요 (예: 3시간 전부터요): ").strip()

    print("\n서버에 전송 중...")
    print("  symptom_text:", symptom_text)
    print("  onset_text:", onset_text)

    try:
        result = call_recommend_api(symptom_text=symptom_text, onset_text=onset_text, user=user)
    except RuntimeError as e:
        print("\n[오류]", e)
        return

    print_human_summary(result)
    print_json_report(result)


if __name__ == "__main__":
    main()
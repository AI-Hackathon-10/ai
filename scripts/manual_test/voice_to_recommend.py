"""
"증상을 음성으로 말하면 실제로 약 추천까지 나오는지" 끝까지 확인하는 통합 데모.

이 폴더의 clova_streaming_client.py(ClovaStreamingClient)로 마이크 음성을 두 번
녹음해서 텍스트로 바꾸고(① 증상, ② 발생 시각), 그 텍스트를 로컬에서 돌고 있는
FastAPI 서버의 POST /symptoms/recommend로 그대로 보낸 뒤 응답을 보기 좋게 출력한다.

사전 조건
---------
1) clova_streaming_client.py가 같은 폴더에 있고, 그 스크립트 자체가 이미 정상
   동작해야 한다(마이크 인식이 콘솔에 찍히는 것까지 확인된 상태).
2) ai 레포에서 `uvicorn app.main:app --reload` (또는 --port 8001 등)로 서버가
   떠 있어야 한다. 아래 FASTAPI_URL을 그 주소로 맞춰줄 것.
3) 서버 쪽 .env에 GEMINI_API_KEY, MFDS_PILL_SERVICE_KEY가 실제 값으로 채워져
   있어야 한다 — 이게 없으면 서버가 503(Gemini/e약은요 호출 실패)을 돌려준다.

실행: python3 voice_to_recommend.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from clova_streaming_client import SECRET_KEY, ClovaStreamingClient

# 서버가 다른 포트로 떠 있으면 여기만 바꾸면 된다 (예: 8000번이 이미 쓰이고 있어서
# --port 8001로 띄우셨다면 "http://127.0.0.1:8001/symptoms/recommend").
FASTAPI_URL = "http://127.0.0.1:8000/symptoms/recommend"

RECORD_SECONDS = 6  # 한 번 녹음에 몇 초 들을지 (짧은 문장이면 충분)


def record_utterance(prompt: str, seconds: float = RECORD_SECONDS) -> str:
    """마이크로 한 번 녹음해서 인식된 텍스트를 이어붙여 반환한다."""
    pieces: list[str] = []

    def on_transcript(text: str, ep_flag: bool):
        text = text.strip()
        if text:
            pieces.append(text)
            print(f"  └ 인식: {text}")

    print(f"\n{prompt} ({seconds:.0f}초간 듣습니다...)")
    client = ClovaStreamingClient(SECRET_KEY, on_transcript=on_transcript)
    client.run_with_microphone(seconds=seconds)
    return " ".join(pieces).strip()


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


def print_result(result: dict) -> None:
    print("\n=== 추천 결과 ===")
    print("인식된 증상:", result.get("parsedSymptoms"))
    print("증상 발생 시각:", result.get("symptomStartedAt"), f"(신뢰도: {result.get('onsetConfidence')})")

    candidates = result.get("candidates") or []
    if not candidates:
        print("\n추천된 약이 없습니다.")
        print("안내:", result.get("notice"))
        return

    for i, c in enumerate(candidates, start=1):
        rec = c.get("recommendation", {})
        print(f"\n[{i}] {c.get('itemName')}  (source={c.get('source')})")
        print(f"    상태: {rec.get('status')} / 점수: {rec.get('score')} / 신뢰도: {rec.get('confidence')}")
        print(f"    이유: {rec.get('reason')}")
        if rec.get("caution"):
            print(f"    주의사항: {rec.get('caution')}")
        if c.get("imageUrl"):
            print(f"    이미지: {c.get('imageUrl')}")

    print("\n※", result.get("disclaimer"))


def main() -> None:
    print("=== 음성 증상 -> 약 추천 통합 테스트 ===")
    print(f"서버 주소: {FASTAPI_URL}")

    user_id = input("사용자 ID [1]: ").strip() or "1"
    name = input("이름 [테스트]: ").strip() or "테스트"
    gender = (input("성별 MALE/FEMALE [MALE]: ").strip() or "MALE").upper()
    birth_date = input("생년월일 YYYY-MM-DD [1995-01-01]: ").strip() or "1995-01-01"
    user = {"userId": int(user_id), "name": name, "gender": gender, "birthDate": birth_date}

    symptom_text = record_utterance("증상을 말씀해주세요 (예: 머리가 아프고 열이 나요)")
    if not symptom_text:
        print("증상 텍스트가 비어 있습니다. 마이크 인식이 안 된 것 같습니다 — clova_streaming_client.py 단독 테스트로 먼저 확인해주세요.")
        return

    onset_text = record_utterance("증상이 언제부터 시작됐는지 말씀해주세요 (예: 3시간 전부터요)")

    print("\n서버에 전송 중...")
    print("  symptom_text:", symptom_text)
    print("  onset_text:", onset_text)

    try:
        result = call_recommend_api(symptom_text=symptom_text, onset_text=onset_text, user=user)
    except RuntimeError as e:
        print("\n[오류]", e)
        return

    print_result(result)


if __name__ == "__main__":
    main()
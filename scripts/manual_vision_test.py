#!/usr/bin/env python3
"""
실제 Gemini Vision API에 알약 사진을 넣어서 추출 결과를 눈으로 확인하는 수동 테스트
스크립트. FastAPI 서버나 LangGraph 실행 없이, 실제 서비스가 쓸 프롬프트/스키마/호출
코드(app/vision/gemini_client.py 의 call_gemini_vision)를 그대로 재사용해서 결과를
빠르게 확인하기 위한 용도다.

사용법:
    export GEMINI_API_KEY=발급받은_API_키   # https://aistudio.google.com/apikey 에서 발급
    pip install -r requirements.txt
    python scripts/manual_vision_test.py --front capsule_front.jpg

앞/뒤 사진이 따로 있으면:
    python scripts/manual_vision_test.py --front front.jpg --back back.jpg
"""
from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (스크립트를 어디서 실행하든 app 패키지를 찾을 수 있게)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vision.gemini_client import GeminiVisionError, call_gemini_vision  # noqa: E402


def _read_bytes(path: str | None, label: str) -> bytes | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        print(f"❌ {label} 파일을 찾을 수 없습니다: {resolved}")
        sys.exit(1)
    data = resolved.read_bytes()
    print(f"  - {label}: {resolved} ({len(data):,} bytes)")
    return data


async def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini Vision 알약 추출 수동 테스트")
    parser.add_argument("--front", required=True, help="앞면(또는 대표) 이미지 경로")
    parser.add_argument("--back", default=None, help="뒷면 이미지 경로 (없으면 생략 가능)")
    args = parser.parse_args()

    print("이미지 로드:")
    front_bytes = _read_bytes(args.front, "front")
    back_bytes = _read_bytes(args.back, "back") if args.back else None

    # 실제 파일 확장자로 mime type을 추정한다 (PNG인데 무조건 image/jpeg로 보내면
    # Gemini 호출이 실패할 수 있다 — /identify 엔드포인트와 동일한 이유).
    front_mime_type = mimetypes.guess_type(args.front)[0] or "image/jpeg"
    back_mime_type = (mimetypes.guess_type(args.back)[0] or "image/jpeg") if args.back else None
    print(f"  - front mime type: {front_mime_type}")
    if back_mime_type:
        print(f"  - back mime type: {back_mime_type}")

    print("\nGemini Vision 호출 중... (실제 API 호출이라 몇 초 걸릴 수 있습니다)")
    try:
        raw = await call_gemini_vision(
            front_bytes,
            back_bytes,
            retry_hint=None,
            front_mime_type=front_mime_type,
            back_mime_type=back_mime_type,
        )
    except GeminiVisionError as e:
        print(f"\n❌ 호출 실패: {e}")
        print("   GEMINI_API_KEY 환경변수가 설정돼 있는지, MODEL_NAME이 유효한지 확인하세요.")
        sys.exit(1)

    print("\n=== 추출 결과 (원시 JSON) ===")
    print(json.dumps(raw, ensure_ascii=False, indent=2))

    print("\n=== 요약 ===")
    print(f"앞면 각인   : {raw.get('print_front')!r}  (신뢰도 {raw.get('print_front_confidence')})")
    print(f"뒷면 각인   : {raw.get('print_back')!r}  (신뢰도 {raw.get('print_back_confidence')})")
    print(f"색상        : {raw.get('color_class1')} / {raw.get('color_class2')}")
    print(f"모양        : {raw.get('drug_shape')}")
    print(f"제형        : {raw.get('form_code_name')}")
    print(f"종합 신뢰도 : {raw.get('overall_confidence')}")

    print(
        "\n확인 포인트: color_class1/drug_shape 값이 app/vision/schema.py의 PILL_COLORS/"
        "PILL_SHAPES 목록에 있는 값과 정확히 일치하는지, 각인이 실제 사진과 맞는지, "
        "확신 없을 때 null을 잘 내는지(추측해서 틀린 값을 채우지 않는지)를 눈으로 확인하세요."
    )


if __name__ == "__main__":
    asyncio.run(main())

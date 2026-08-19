#!/usr/bin/env python3
"""
로컬에 있는 알약 사진 파일(앞/뒤)을 base64 data URI로 인코딩해서 실행 중인 FastAPI
서버의 POST /identify 에 JSON body로 요청을 보내는 스크립트. JSON을 손으로 만들다가
따옴표/이스케이프 실수하는 걸 피하려고 만들었습니다.

사용법:
    # 서버가 로컬에서 떠 있는 상태에서 바로 호출 (기본 URL: http://127.0.0.1:8000/identify)
    python scripts/call_identify.py --front sc250_front.png --back sc250_back.png

    # 서버를 호출하지 않고 요청 JSON만 출력 (Swagger UI 'Request body'에 복붙용)
    python scripts/call_identify.py --front sc250_front.png --back sc250_back.png --print-only

    # 서버 주소를 바꾸고 싶으면
    python scripts/call_identify.py --front f.jpg --url http://127.0.0.1:8000/identify
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path

import httpx


def _to_data_uri(path: str) -> str:
    resolved = Path(path)
    if not resolved.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {resolved}")
        sys.exit(1)

    mime_type = mimetypes.guess_type(str(resolved))[0]
    if mime_type is None:
        print(f"⚠️  {resolved.name}: 확장자로 mime type을 추측하지 못해 image/jpeg로 처리합니다.")
        mime_type = "image/jpeg"

    raw = resolved.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    print(f"  - {resolved.name}: {len(raw):,} bytes -> base64 {len(encoded):,}자, mime type {mime_type}")
    return f"data:{mime_type};base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser(description="이미지 파일을 base64로 인코딩해 /identify 호출")
    parser.add_argument("--front", required=True, help="앞면(각인 있는 쪽) 이미지 경로")
    parser.add_argument("--back", default=None, help="뒷면 이미지 경로 (없으면 생략 가능)")
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/identify", help="호출할 /identify 엔드포인트 URL"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="서버를 호출하지 않고 요청 JSON만 출력한다 (Swagger UI에 복붙용)",
    )
    args = parser.parse_args()

    print("이미지 인코딩:")
    body = {"front_image": _to_data_uri(args.front)}
    if args.back:
        body["back_image"] = _to_data_uri(args.back)

    if args.print_only:
        print("\n=== Swagger UI 'Request body'에 붙여넣을 JSON ===")
        print(json.dumps(body, ensure_ascii=False))
        return

    print(f"\n{args.url} 호출 중...")
    try:
        response = httpx.post(args.url, json=body, timeout=60.0)
    except httpx.ConnectError:
        print(f"❌ {args.url} 에 연결할 수 없습니다. 서버가 떠 있는지 확인하세요 (uvicorn app.main:app --reload).")
        sys.exit(1)

    print(f"\n=== 응답 (status {response.status_code}) ===")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()

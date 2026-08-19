#!/usr/bin/env python3
"""
Vision AI(Gemini)를 완전히 건너뛰고, 낱알식별 API와 e약은요 API "연결 자체"만 직접
확인하는 수동 테스트 스크립트.

지금까지 겪은 문제(모양/각인 추측이 섞이는 매칭 결과)와 분리해서, "서비스키가 유효한가",
"쿼리 파라미터가 실제로 먹히는가", "e약은요로 상세정보까지 이어지는가"를 순수하게
확인하기 위한 용도다.

동작 방식:
1) 낱알식별 API에 필터 없이(서비스키/페이징만) 검색 요청을 보낸다.
   -> header.resultCode가 "00"이고 total_count > 0이면 연결/인증 자체는 정상이라는 뜻.
2) 응답으로 받은 첫 번째 항목의 실제 ITEM_SEQ를 그대로 e약은요 API에 넘겨서 상세조회한다.
   -> 낱알식별 DB에 실제로 존재하는 ITEM_SEQ이므로, 두 API가 서로 값을 주고받는
      전체 사슬(chain)이 맞물려 동작하는지까지 확인할 수 있다.
3) (선택) --print-front / --drug-shape / --color 로 실제 필터를 걸어서, 값이 얼마나
   걸러지는지 눈으로 비교해볼 수 있다. 대소문자(PRINT_FRONT 등) 수정이 실제로 먹히는지
   확인할 때 특히 유용하다.

사용법:
    python scripts/manual_api_test.py
    python scripts/manual_api_test.py --print-front TYLENOL
    python scripts/manual_api_test.py --drug-shape 장방형 --color 하양
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import httpx

from app.config import settings  # noqa: E402
from app.drug_detail_client import DrugDetailApiClient, DrugDetailApiError  # noqa: E402
from app.pill_identification_api_client import (  # noqa: E402
    PillIdentificationApiClient,
    PillApiError,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="낱알식별/e약은요 API 연결 직접 확인")
    parser.add_argument("--print-front", default=None, help="앞면 각인으로 필터링해서 검색")
    parser.add_argument("--print-back", default=None, help="뒷면 각인으로 필터링해서 검색")
    parser.add_argument("--drug-shape", default=None, help="모양 코드값으로 필터링해서 검색")
    parser.add_argument("--color", default=None, help="주색상(COLOR_CLASS1) 코드값으로 필터링해서 검색")
    args = parser.parse_args()

    if not settings.mfds_pill_service_key:
        print("❌ MFDS_PILL_SERVICE_KEY가 비어있습니다. .env를 확인하세요.")
        sys.exit(1)

    http_client = httpx.AsyncClient(timeout=10.0)
    pill_client = PillIdentificationApiClient(
        service_key=settings.mfds_pill_service_key, http_client=http_client
    )
    detail_client = DrugDetailApiClient(
        service_key=settings.mfds_pill_service_key, http_client=http_client
    )

    params: dict[str, str] = {}
    if args.print_front:
        params["PRINT_FRONT"] = args.print_front
    if args.print_back:
        params["PRINT_BACK"] = args.print_back
    if args.drug_shape:
        params["DRUG_SHAPE"] = args.drug_shape
    if args.color:
        params["COLOR_CLASS1"] = args.color

    print("=== 1) 낱알식별 API 호출 ===")
    print(f"보내는 파라미터: {params if params else '(없음 — 필터 없이 전체 조회)'}")

    try:
        pill_response = await pill_client.search(params)
    except PillApiError as e:
        print(f"\n❌ 낱알식별 API 호출 실패: {e}")
        print(
            "   -> SERVICE_KEY_IS_NOT_REGISTERED_ERROR면 서비스키/승인 상태를,\n"
            "      INVALID REQUEST PARAMETER ERROR면 파라미터명(대소문자 포함)을 의심하세요."
        )
        await http_client.aclose()
        sys.exit(1)

    header = pill_response.header
    print(f"resultCode  : {header.result_code if header else None}")
    print(f"resultMsg   : {header.result_msg if header else None}")
    print(f"total_count : {pill_response.total_count}")

    items = pill_response.body.items if pill_response.body else []
    if not items:
        print("\n⚠️  항목이 0건입니다. 필터를 걸었다면 조건을 완화해서 다시 시도해보세요.")
        print("   필터 없이 실행했는데도 0건이면 연결/인증 자체를 의심해야 합니다.")
        await pill_client.aclose()
        await detail_client.aclose()
        await http_client.aclose()
        return

    print(f"\n첫 {min(3, len(items))}건 미리보기:")
    for item in items[:3]:
        print(
            f"  - item_seq={item.item_seq!r}  item_name={item.item_name!r}  "
            f"entp_name={item.entp_name!r}  print_front={item.print_front!r}  "
            f"print_back={item.print_back!r}  drug_shape={item.drug_shape!r}  "
            f"color_class1={item.color_class1!r}"
        )

    first_item_seq = items[0].item_seq
    if not first_item_seq:
        print("\n⚠️  첫 항목에 item_seq가 비어있어 e약은요 조회를 건너뜁니다.")
        await pill_client.aclose()
        await detail_client.aclose()
        await http_client.aclose()
        return

    print(f"\n=== 2) e약은요 API 호출 (item_seq={first_item_seq}) ===")
    try:
        detail_response = await detail_client.get_by_item_seq(first_item_seq)
    except DrugDetailApiError as e:
        print(f"\n❌ e약은요 API 호출 실패: {e}")
        print(
            "   -> 낱알식별 API는 됐는데 여기서 실패하면, e약은요는 별도 활용신청이 "
            "필요한 API일 수 있으니 data.go.kr에서 승인 상태를 확인하세요."
        )
        await pill_client.aclose()
        await detail_client.aclose()
        await http_client.aclose()
        sys.exit(1)

    detail_header = detail_response.header
    print(f"resultCode  : {detail_header.result_code if detail_header else None}")
    print(f"resultMsg   : {detail_header.result_msg if detail_header else None}")
    print(f"total_count : {detail_response.total_count}")

    detail_items = detail_response.body.items if detail_response.body else []
    if detail_items:
        d = detail_items[0]
        print(f"\nitem_name : {d.item_name}")
        print(f"entp_name : {d.entp_name}")
        efficacy = (d.efficacy or "")[:80]
        print(f"efficacy  : {efficacy}{'...' if d.efficacy and len(d.efficacy) > 80 else ''}")
        print("\n✅ 낱알식별 -> e약은요로 이어지는 전체 연결 체인이 정상 동작합니다.")
    else:
        print("\n⚠️  낱알식별 API는 됐지만 e약은요에서 이 item_seq로 0건입니다.")
        print("   두 API가 서로 다른 ITEM_SEQ 체계를 쓸 가능성을 의심해볼 수 있습니다.")

    await pill_client.aclose()
    await detail_client.aclose()
    await http_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())

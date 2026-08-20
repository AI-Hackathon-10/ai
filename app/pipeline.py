"""
Step 1~4 전체를 잇는 오케스트레이션.

Step 1(알약 앞/뒤 촬영)은 프론트엔드/업로드 단계라 여기서 다루지 않는다. 이 모듈은
Step 2(Vision AI로 검색 조건 추출) -> Step 3(DB 매칭) -> Step 4(e약은요
상세정보 조회)를 순서대로 연결한다.

Step 4는 Step 3에서 SINGLE_MATCH 로 정확히 1건이 확정됐을 때만 자동으로 실행한다.
MULTIPLE_MATCHES 인 경우 사용자가 후보 중 하나를 직접 고른 뒤 그 item_seq로
fetch_detail_for_selected_candidate() 를 별도로 호출해야 한다 — 여러 후보 중 하나를
서버가 임의로 골라 상세정보를 보여주면 안 되기 때문이다("신뢰도가 낮으면 결과를
확정하지 않는다" 원칙과 같은 이유).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.drug_detail_models import DrugDetailApiItem
from app.drug_detail_service import DrugDetailService
from app.models import MatchStatus, PillMatchResult, VisionExtractionResult
from app.pill_matching_service import PillMatchingService
from app.vision.run import run_vision_extraction


class PillIdentificationOutcome(BaseModel):
    """FastAPI 응답으로 그대로 내보낼 최종 결과."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    match_result: Optional[PillMatchResult] = None
    detail: Optional[DrugDetailApiItem] = None
    vision_failed: bool = False  # True면 프론트에서 "재촬영 요청"으로 안내해야 함


async def identify_pill(
    front_image_bytes: bytes,
    back_image_bytes: Optional[bytes],
    matching_service: PillMatchingService,
    detail_service: DrugDetailService,
    front_mime_type: str = "image/jpeg",
    back_mime_type: Optional[str] = None,
) -> PillIdentificationOutcome:
    """Step 2 -> 3 -> (조건부) 4 를 한 번에 실행한다."""

    # Step 2: Vision AI로 검색 조건 추출
    vision_result = await run_vision_extraction(
        front_image_bytes,
        back_image_bytes,
        front_mime_type=front_mime_type,
        back_mime_type=back_mime_type,
    )
    if vision_result is None:
        return PillIdentificationOutcome(vision_failed=True)

    # Step 3: DB 매칭
    match_result: PillMatchResult = await matching_service.match(vision_result)

    # Step 4: 정확히 1건으로 확정됐을 때만 자동으로 상세정보까지 조회
    detail: Optional[DrugDetailApiItem] = None
    if match_result.status == MatchStatus.SINGLE_MATCH:
        item_seq = match_result.candidates[0].item_seq
        if item_seq:
            detail = await detail_service.get_detail(item_seq)

    return PillIdentificationOutcome(match_result=match_result, detail=detail)


class IdentifyFromDbOutcome(BaseModel):
    """Vision 추출 + pill_identification 테이블 매칭 결과."""

    vision_failed: bool = False
    vision_result: Optional[VisionExtractionResult] = None
    match_result: Optional[PillMatchResult] = None


async def identify_from_db(
    front_image_bytes: bytes,
    back_image_bytes: Optional[bytes],
    matching_service: PillMatchingService,
    front_mime_type: str = "image/jpeg",
    back_mime_type: Optional[str] = None,
) -> IdentifyFromDbOutcome:
    """Step 2(Vision 추출) 후 pill_identification 테이블만 조회한다."""

    vision_result = await run_vision_extraction(
        front_image_bytes,
        back_image_bytes,
        front_mime_type=front_mime_type,
        back_mime_type=back_mime_type,
    )
    if vision_result is None:
        return IdentifyFromDbOutcome(vision_failed=True)

    match_result = await matching_service.match(vision_result)
    return IdentifyFromDbOutcome(vision_result=vision_result, match_result=match_result)


async def identify_from_db_batch(
    items: list[dict],
    matching_service: PillMatchingService,
) -> list[IdentifyFromDbOutcome]:
    """여러 알약 이미지 세트를 동시에 Vision 추출 + DB 매칭한다.

    items 는 각각 {front_bytes, back_bytes, front_mime_type, back_mime_type}
    키를 가진 dict 리스트다. asyncio.gather 로 병렬 실행하되, 개별 항목의
    예외가 전체를 중단시키지 않도록 한다.
    """

    async def _single(item: dict) -> IdentifyFromDbOutcome:
        return await identify_from_db(
            front_image_bytes=item["front_bytes"],
            back_image_bytes=item["back_bytes"],
            matching_service=matching_service,
            front_mime_type=item["front_mime_type"],
            back_mime_type=item["back_mime_type"],
        )

    import asyncio

    results = await asyncio.gather(
        *[_single(item) for item in items],
        return_exceptions=True,
    )

    outcomes: list[IdentifyFromDbOutcome] = []
    for r in results:
        if isinstance(r, Exception):
            outcomes.append(IdentifyFromDbOutcome(vision_failed=True))
        else:
            outcomes.append(r)
    return outcomes


async def fetch_detail_for_selected_candidate(
    item_seq: str,
    detail_service: DrugDetailService,
) -> Optional[DrugDetailApiItem]:
    """MULTIPLE_MATCHES 후보 목록에서 사용자가 하나를 선택했을 때 프론트에서 호출하는
    Step 4 전용 엔드포인트가 사용할 함수."""
    return await detail_service.get_detail(item_seq)

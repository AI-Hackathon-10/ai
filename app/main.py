"""
FastAPI 앱 진입점. Step 2(Vision AI 추출) -> Step 3(DB 매칭) -> Step 4(e약은요
상세정보 조회)를 실제 HTTP 엔드포인트로 연결한다.

로컬 실행 (uv 기준):
    uv run uvicorn app.main:app --reload
    # 또는 (fastapi-cli 필요: uv add "fastapi[standard]")
    uv run fastapi dev app/main.py

실행 후 http://127.0.0.1:8000/docs 에서 Swagger UI로 바로 테스트할 수 있다.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, Field

from app.base64_images import decode_base64_image_or_400
from app.config import settings
from app.drug_detail_client import DrugDetailApiClient
from app.drug_detail_service import DrugDetailService
from app.drug_detail_models import DrugDetailApiItem
from app.identify_result import IdentifyResultItem, build_identify_result, needs_official_detail
from app.models import PillMatchResult, VisionExtractionResult
from app.pill_data_loader import load_pill_data
from app.pill_identification_repository import MysqlPillCatalogClient, PillIdentificationRepository
from app.pill_matching_service import PillMatchingService
from app.pipeline import fetch_detail_for_selected_candidate, identify_from_db, identify_from_db_batch, identify_pill

from app.vision.gemini_client import GeminiVisionError
from app.vision.run import run_vision_extraction

logger = logging.getLogger(__name__)

_SWAGGER_PLACEHOLDERS = {"", "string", "null"}


class IdentifyRequest(BaseModel):
    """이미지를 파일 업로드가 아니라 base64 문자열로 받는다.

    front_image / back_image 는 아래 둘 중 하나의 형태를 받을 수 있다.
    1) data URI: "data:image/png;base64,iVBORw0KGgo..." (mime type이 자동으로 추출됨, 권장)
    2) 순수 base64 문자열: "iVBORw0KGgo..." (이 경우 front_mime_type/back_mime_type을
       실제 이미지 형식에 맞게 함께 보내야 한다. 안 보내면 기본값 "image/jpeg"로 처리되는데,
       실제로 PNG인데 이 기본값을 그대로 쓰면 Gemini 호출이 실패할 수 있다.)
    """

    front_image: str = Field(..., description="알약 앞면 이미지 (base64 또는 data URI)")
    back_image: Optional[str] = Field(
        None,
        description="알약 뒷면 이미지 (base64 또는 data URI, 선택). 없으면 생략하거나 null.",
        examples=[None],
    )
    front_mime_type: str = Field(
        "image/jpeg", description="front_image가 순수 base64일 때만 사용되는 mime type"
    )
    back_mime_type: Optional[str] = Field(
        None,
        description="back_image가 순수 base64일 때만 사용되는 mime type. 없으면 생략하거나 null.",
        examples=[None],
    )


class IdentifyResponse(BaseModel):
    """/identify 응답. Vision 추출이 실패하면 match_result/detail은 둘 다 None이고
    vision_failed=True다 — 프론트에서는 이 경우 "재촬영 요청"으로 안내하면 된다."""

    vision_failed: bool = False
    match_result: Optional[PillMatchResult] = None
    detail: Optional[DrugDetailApiItem] = None


class SelectCandidateResponse(BaseModel):
    detail: Optional[DrugDetailApiItem] = None


class VisionExtractResponse(BaseModel):
    """Vision AI(Step 2) 추출 결과만 담는다. 낱알식별 매칭(Step 3)이나 e약은요 상세조회
    (Step 4)는 하지 않는다 — Vision 단계 하나만 따로 확인하고 싶을 때 쓰는 디버깅용
    엔드포인트다. (동료가 보여준 /pills/features와 같은 역할: 겉모습만 추출하고 제품명은
    추측하지 않는다.)"""

    vision_failed: bool = False
    result: Optional[VisionExtractionResult] = None


class IdentifyDbResponse(BaseModel):
    """/vision/extract 와 같은 이미지 요청으로 Vision 추출 후, 그 값으로
    pill_identification 테이블을 조회한 결과."""

    vision_failed: bool = False
    vision_result: Optional[VisionExtractionResult] = None
    match_result: Optional[PillMatchResult] = None


class UserInfo(BaseModel):
    """배치 요청 시 백엔드가 함께 보내주는 유저 정보."""

    user_id: int = Field(..., alias="userId", description="사용자 고유 ID (PK)")
    name: str = Field(..., description="사용자 이름")
    gender: str = Field(..., description="성별 (MALE / FEMALE)")
    birth_date: date = Field(..., alias="birthDate", description="생년월일 (YYYY-MM-DD)")

    model_config = {"populate_by_name": True}


class BatchIdentifyDbItem(BaseModel):
    """배치 요청의 개별 항목. id 가 최상단에 위치하도록 필드 순서를 직접 정의한다."""

    id: str = Field(..., description="클라이언트가 부여한 알약 식별 ID (응답에서 그대로 돌려줌)")
    front_image: str = Field(..., description="알약 앞면 이미지 (base64 또는 data URI)")
    back_image: Optional[str] = Field(
        None,
        description="알약 뒷면 이미지 (base64 또는 data URI, 선택). 없으면 생략하거나 null.",
        examples=[None],
    )
    front_mime_type: str = Field(
        "image/jpeg", description="front_image가 순수 base64일 때만 사용되는 mime type"
    )
    back_mime_type: Optional[str] = Field(
        None,
        description="back_image가 순수 base64일 때만 사용되는 mime type. 없으면 생략하거나 null.",
        examples=[None],
    )


class BatchIdentifyDbRequest(BaseModel):
    """여러 알약 이미지를 한 번에 보내서 Vision 추출 + DB 매칭을 일괄 수행한다.

    items 의 각 항목은 기존 IdentifyRequest 와 동일한 이미지 필드에
    id 필드가 추가된 형태다.
    """

    user: UserInfo = Field(..., description="요청 유저 정보")
    symptoms: list[str] = Field(
        default_factory=list,
        description="사용자가 선택한 증상 (SymptomType enum 값: HEADACHE, FEVER 등). 여러 개 선택 가능",
    )
    symptom_started_at: Optional[AwareDatetime] = Field(
        default=None,
        alias="symptomStartedAt",
        description="증상 발생 시각 (ISO 8601, 예: 2026-08-20T14:00:00+09:00)",
        examples=["2026-08-20T14:00:00+09:00"],
    )
    items: list[BatchIdentifyDbItem] = Field(
        ..., description="알약 이미지 세트 리스트", min_length=1
    )

    model_config = {"populate_by_name": True}


class BatchIdentifyDbResponse(BaseModel):
    """/identify/db/batch 응답. 요청한 순서대로 result 리스트를 담는다."""

    result: list[IdentifyResultItem] = Field(default_factory=list)


def _optional_body_value(raw: Optional[str]) -> Optional[str]:
    """Swagger Try it out 기본값('string')과 빈 값을 없는 필드로 취급한다."""
    if raw is None:
        return None
    stripped = raw.strip()
    if stripped.lower() in _SWAGGER_PLACEHOLDERS:
        return None
    return raw


def _decode_request_images(body: Union["IdentifyRequest", "BatchIdentifyDbItem"]) -> tuple[bytes, str, Optional[bytes], Optional[str]]:
    """IdentifyRequest(base64 JSON body)를 (front_bytes, front_mime_type,
    back_bytes_or_None, back_mime_type_or_None)로 디코딩한다. /identify와
    /vision/extract가 요청 형식을 그대로 공유하므로 디코딩 로직도 공유한다."""
    front_decoded = decode_base64_image_or_400(
        body.front_image, fallback_mime_type=body.front_mime_type, field_name="front_image"
    )
    if front_decoded is None:
        raise HTTPException(status_code=400, detail="front_image는 필수이며 빈 문자열일 수 없습니다.")
    front_bytes, front_mime_type = front_decoded

    back_image = _optional_body_value(body.back_image)
    back_mime = _optional_body_value(body.back_mime_type)
    back_decoded = decode_base64_image_or_400(
        back_image, fallback_mime_type=back_mime or "image/jpeg", field_name="back_image"
    )
    back_bytes, back_mime_type = back_decoded if back_decoded else (None, None)

    return front_bytes, front_mime_type, back_bytes, back_mime_type


_http_client = httpx.AsyncClient(timeout=10.0)


def _mysql_connect():
    import pymysql

    return pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_db,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


_matching_service = PillMatchingService(
    MysqlPillCatalogClient(PillIdentificationRepository(connect=_mysql_connect))
)

# data.go.kr은 계정 하나당 인증키 하나로 여러 API를 함께 쓰는 구조라 같은 키를 재사용한다.
# e약은요를 별도 계정/키로 신청했다면 .env에 키를 하나 더 추가해서 갈라주면 된다.
_detail_api_client = DrugDetailApiClient(
    service_key=settings.mfds_pill_service_key,
    http_client=_http_client,
)
_detail_service = DrugDetailService(_detail_api_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 낱알식별 데이터를 백그라운드에서 DB에 적재한다.
    # 적재가 완료되기 전에도 서버는 요청을 받을 수 있다.
    task = asyncio.create_task(_startup_load_pill_data())
    yield
    task.cancel()
    await _detail_api_client.aclose()
    await _http_client.aclose()


async def _startup_load_pill_data():
    """서버 시작 시 낱알식별 데이터 적재. 실패해도 서버는 정상 동작한다."""
    try:
        count = await load_pill_data()
        logger.info("시작 시 낱알식별 데이터 적재 완료: %d건", count)
    except Exception:
        logger.exception("시작 시 낱알식별 데이터 적재 실패 — 기존 DB 데이터로 서비스합니다.")


app = FastAPI(title="알약 식별 API", lifespan=lifespan)


@app.exception_handler(GeminiVisionError)
async def gemini_vision_error_handler(_request: Request, exc: GeminiVisionError) -> JSONResponse:
    """Gemini 쪽 장애(수요 폭주 503 등)를 문서화되지 않은 500 대신 JSON 503으로 돌려준다."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# @app.post("/identify", response_model=IdentifyResponse)
# async def identify(body: IdentifyRequest):
#     """Step 1(촬영)에서 넘어온 사진(base64 JSON body)으로 Step 2 -> 3 -> (조건부) 4 를
#     한 번에 실행한다. pill_identification DB 테이블로 매칭한다.

#     front_image는 필수이며 빈 문자열이면 400을 반환한다. base64 디코딩에 실패해도
#     500이 아니라 400으로 응답한다(app/base64_images.py 의 decode_base64_image_or_400).
#     """
#     front_bytes, front_mime_type, back_bytes, back_mime_type = _decode_request_images(body)

#     outcome = await identify_pill(
#         front_bytes,
#         back_bytes,
#         _matching_service,
#         _detail_service,
#         front_mime_type=front_mime_type,
#         back_mime_type=back_mime_type,
#     )

#     return IdentifyResponse(
#         vision_failed=outcome.vision_failed,
#         match_result=outcome.match_result,
#         detail=outcome.detail,
#     )


@app.post("/vision/extract", response_model=VisionExtractResponse)
async def vision_extract(body: IdentifyRequest):
    """Step 2(Vision AI 추출)까지만 실행하고 매칭/상세조회 없이 원시 추출 결과를
    그대로 반환한다. 요청 형식은 /identify와 동일한 base64 JSON body를 그대로
    재사용한다. Vision이 실제로 각인/모양/색상을 무엇으로 인식했는지 낱알식별 매칭과
    분리해서 확인하고 싶을 때 이 엔드포인트로 먼저 확인해보면 된다."""
    front_bytes, front_mime_type, back_bytes, back_mime_type = _decode_request_images(body)

    result = await run_vision_extraction(
        front_bytes,
        back_bytes,
        front_mime_type=front_mime_type,
        back_mime_type=back_mime_type,
    )

    if result is None:
        return VisionExtractResponse(vision_failed=True, result=None)
    return VisionExtractResponse(vision_failed=False, result=result)


@app.post("/identify/db", response_model=IdentifyDbResponse)
async def identify_db(body: IdentifyRequest):
    """/vision/extract 를 실행한 뒤, 나온 각인·색·모양으로 pill_identification
    테이블을 조회한다. 요청 형식은 /identify, /vision/extract 와 같다."""
    front_bytes, front_mime_type, back_bytes, back_mime_type = _decode_request_images(body)

    outcome = await identify_from_db(
        front_bytes,
        back_bytes,
        _matching_service,
        front_mime_type=front_mime_type,
        back_mime_type=back_mime_type,
    )
    return IdentifyDbResponse(
        vision_failed=outcome.vision_failed,
        vision_result=outcome.vision_result,
        match_result=outcome.match_result,
    )


@app.post("/identify", response_model=BatchIdentifyDbResponse)
async def identify_db_batch(body: BatchIdentifyDbRequest):
    """여러 알약 이미지를 한 번에 Vision 추출 + DB 매칭하고, 확정된 항목은
    상세정보까지 조회한다. 응답은 요청 순서의 `result` 리스트다.
    """
    decoded_items: list[dict] = []
    for item in body.items:
        front_bytes, front_mime_type, back_bytes, back_mime_type = _decode_request_images(item)
        decoded_items.append({
            "front_bytes": front_bytes,
            "front_mime_type": front_mime_type,
            "back_bytes": back_bytes,
            "back_mime_type": back_mime_type,
        })

    outcomes = await identify_from_db_batch(decoded_items, _matching_service)

    async def _detail_for(outcome):
        if not needs_official_detail(outcome):
            return None
        item_seq = outcome.match_result.candidates[0].item_seq
        try:
            return await _detail_service.get_detail(item_seq)
        except Exception:
            logger.exception("배치 항목 e약은요 상세정보 조회 실패: item_seq=%s", item_seq)
            return None

    details = await asyncio.gather(*[_detail_for(outcome) for outcome in outcomes])

    result = [
        build_identify_result(
            id=body.items[i].id,
            outcome=outcomes[i],
            detail=details[i],
            symptoms=body.symptoms,
            gender=body.user.gender,
            birth_date=body.user.birth_date,
            symptom_started_at=body.symptom_started_at,
        )
        for i in range(len(outcomes))
    ]
    return BatchIdentifyDbResponse(result=result)


@app.post("/identify/select/{item_seq}", response_model=SelectCandidateResponse)
async def select_candidate(item_seq: str):
    """MULTIPLE_MATCHES 후보 목록에서 사용자가 하나를 고른 뒤 호출하는 Step 4 전용 엔드포인트."""
    detail = await fetch_detail_for_selected_candidate(item_seq, _detail_service)
    return SelectCandidateResponse(detail=detail)


@app.post("/admin/reload-pill-data")
async def reload_pill_data():
    """낱알식별 API에서 전체 데이터를 다시 가져와 DB에 적재한다.
    기존 데이터는 삭제하고 새로 넣는다."""
    try:
        count = await load_pill_data()
        return {"status": "ok", "loaded_count": count}
    except Exception as e:
        logger.exception("수동 낱알식별 데이터 적재 실패")
        raise HTTPException(status_code=500, detail=f"데이터 적재 실패: {e}") from e

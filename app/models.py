"""
Vision AI 추출 결과, 낱알식별 API 요청/응답, 매칭 결과를 표현하는 모델 모음.

⚠️ 필드명 검증 필요: data.go.kr 상세페이지가 이 세션에서는 robots.txt로 막혀 있어
최신 명세를 직접 확인하지 못했다. PillApiItem 의 alias(ITEM_SEQ, PRINT_FRONT, DRUG_SHAPE 등)는
이 계열 API에서 통용되는 명칭 기준이며, 서비스키 발급 시 제공되는 활용가이드/OpenAPI 명세의
실제 응답 예시와 반드시 한 번 대조할 것. alias 값만 바꾸면 되고 매칭 로직은 그대로 재사용 가능.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class VisionExtractionResult(BaseModel):
    """FastAPI에서 Gemini Vision 분석 결과로 만들어지는 값.

    color_class1, drug_shape 는 반드시 낱알식별 API가 쓰는 고정 코드값(enum)이어야
    매칭이 된다. Vision AI 프롬프트 단계에서 이 값들로만 답하도록 이미 강제해둔 상태를
    전제로 한다 (이 서비스는 값을 별도로 보정하지 않는다).

    재질의(재시도) 없이 1차 추출값을 그대로 쓴다 — 판단하지 못한 필드는 confidence
    점수 없이 바로 None으로 채워서 돌아오고, 그 None은 매칭(Step 3)에서 조건을
    그만큼 덜 거는 걸로 반영된다.
    """

    print_front: Optional[str] = None       # 앞면 각인 (판단 불가면 None)
    print_back: Optional[str] = None        # 뒷면 각인 (판단 불가면 None)
    color_class1: Optional[str] = None      # 색상 (고정 코드값, 판단 불가면 None)
    drug_shape: Optional[str] = None        # 모양 (고정 코드값, 판단 불가면 None)
    score_line: Optional[bool] = None       # 분할선(스코어라인) 유무, 판단 불가면 None


class PillApiItem(BaseModel):
    """낱알식별 API가 반환하는 개별 의약품 항목."""

    model_config = ConfigDict(populate_by_name=True)

    item_seq: Optional[str] = Field(default=None, alias="ITEM_SEQ")
    item_name: Optional[str] = Field(default=None, alias="ITEM_NAME")
    entp_name: Optional[str] = Field(default=None, alias="ENTP_NAME")
    chart: Optional[str] = Field(default=None, alias="CHART")
    item_image: Optional[str] = Field(default=None, alias="ITEM_IMAGE")
    print_front: Optional[str] = Field(default=None, alias="PRINT_FRONT")
    print_back: Optional[str] = Field(default=None, alias="PRINT_BACK")
    drug_shape: Optional[str] = Field(default=None, alias="DRUG_SHAPE")
    color_class1: Optional[str] = Field(default=None, alias="COLOR_CLASS1")
    color_class2: Optional[str] = Field(default=None, alias="COLOR_CLASS2")
    line_front: Optional[str] = Field(default=None, alias="LINE_FRONT")
    line_back: Optional[str] = Field(default=None, alias="LINE_BACK")
    form_code_name: Optional[str] = Field(default=None, alias="FORM_CODE_NAME")


class PillApiHeader(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    result_code: Optional[str] = Field(default=None, alias="resultCode")
    result_msg: Optional[str] = Field(default=None, alias="resultMsg")

    @property
    def is_success(self) -> bool:
        return self.result_code == "00"


class PillApiBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: list[PillApiItem] = Field(default_factory=list, alias="items")
    total_count: int = Field(default=0, alias="totalCount")
    page_no: Optional[int] = Field(default=None, alias="pageNo")
    num_of_rows: Optional[int] = Field(default=None, alias="numOfRows")


class PillApiResponse(BaseModel):
    """식약처 의약품 낱알식별 정보 API(MdcinGrnIdntfcInfoService03) 응답."""

    model_config = ConfigDict(populate_by_name=True)

    header: Optional[PillApiHeader] = None
    body: Optional[PillApiBody] = None

    @property
    def total_count(self) -> int:
        return self.body.total_count if self.body else 0


class PillCandidate(BaseModel):
    """사용자에게 보여줄 후보 의약품. 테이블 조회 시 각인/색/모양도 함께 내려준다."""

    item_seq: Optional[str] = None
    item_name: Optional[str] = None
    entp_name: Optional[str] = None
    chart: Optional[str] = None
    item_image: Optional[str] = None
    print_front: Optional[str] = None
    print_back: Optional[str] = None
    drug_shape: Optional[str] = None
    color_class1: Optional[str] = None
    color_class2: Optional[str] = None
    line_front: Optional[str] = None
    line_back: Optional[str] = None
    form_code_name: Optional[str] = None

    @classmethod
    def from_item(cls, item: PillApiItem) -> "PillCandidate":
        return cls(
            item_seq=item.item_seq,
            item_name=item.item_name,
            entp_name=item.entp_name,
            chart=item.chart,
            item_image=item.item_image,
            print_front=item.print_front,
            print_back=item.print_back,
            drug_shape=item.drug_shape,
            color_class1=item.color_class1,
            color_class2=item.color_class2,
            line_front=item.line_front,
            line_back=item.line_back,
            form_code_name=item.form_code_name,
        )


class MatchStatus(str, Enum):
    SINGLE_MATCH = "SINGLE_MATCH"            # 정확히 1건. 확정 후보로 넘길 수 있음 (프론트 확인 모달은 그래도 거칠 것)
    MULTIPLE_MATCHES = "MULTIPLE_MATCHES"    # 2건 이상. 사용자가 직접 선택해야 함 — 자동 확정 금지
    NO_MATCH = "NO_MATCH"                    # 모든 완화 단계를 거쳐도 0건. 재촬영/전문가 확인 안내
    TOO_MANY_CANDIDATES = "TOO_MANY_CANDIDATES"  # 가장 넓은 조건에서도 후보 과다. 추가 정보 필요


class PillMatchResult(BaseModel):
    status: MatchStatus
    candidates: list[PillCandidate] = Field(default_factory=list)
    query_level_used: str  # 어느 완화 단계에서 결정됐는지 (디버깅/로그용)

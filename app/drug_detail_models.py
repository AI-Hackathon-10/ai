"""
e약은요(DrbEasyDrugInfoService) API 요청/응답 모델 (Step 4: 의약품 상세정보 조회).

⚠️ 필드명 검증 필요: 이 세션에서는 data.go.kr 상세페이지를 robots.txt 때문에 직접
열람하지 못했다. 아래 alias(efcyQesitm, useMethodQesitm 등)는 e약은요 API에서 널리
통용되는 명칭 기준이며, 서비스키 발급 시 제공되는 활용가이드의 실제 응답 예시와
한 번 대조할 것. alias 값만 바꾸면 되고 나머지 로직은 그대로 재사용 가능하다.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DrugDetailApiItem(BaseModel):
    """e약은요 API가 반환하는 의약품 상세정보 한 건."""

    model_config = ConfigDict(populate_by_name=True)

    item_seq: Optional[str] = Field(default=None, alias="itemSeq")
    item_name: Optional[str] = Field(default=None, alias="itemName")
    entp_name: Optional[str] = Field(default=None, alias="entpName")
    item_image: Optional[str] = Field(default=None, alias="itemImage")
    efficacy: Optional[str] = Field(default=None, alias="efcyQesitm")               # 효능효과
    usage_method: Optional[str] = Field(default=None, alias="useMethodQesitm")       # 사용법(용법용량)
    warning: Optional[str] = Field(default=None, alias="atpnWarnQesitm")             # 경고
    precautions: Optional[str] = Field(default=None, alias="atpnQesitm")             # 주의사항
    interactions: Optional[str] = Field(default=None, alias="intrcQesitm")           # 상호작용
    side_effects: Optional[str] = Field(default=None, alias="seQesitm")              # 부작용
    storage_method: Optional[str] = Field(default=None, alias="depositMethodQesitm")  # 보관법


class DrugDetailApiHeader(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    result_code: Optional[str] = Field(default=None, alias="resultCode")
    result_msg: Optional[str] = Field(default=None, alias="resultMsg")

    @property
    def is_success(self) -> bool:
        return self.result_code == "00"


class DrugDetailApiBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: list[DrugDetailApiItem] = Field(default_factory=list, alias="items")
    total_count: int = Field(default=0, alias="totalCount")


class DrugDetailApiResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    header: Optional[DrugDetailApiHeader] = None
    body: Optional[DrugDetailApiBody] = None

    @property
    def total_count(self) -> int:
        return self.body.total_count if self.body else 0

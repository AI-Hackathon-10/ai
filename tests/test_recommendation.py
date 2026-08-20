"""나이·성별·증상으로 복용 가능 여부와 주의사항을 개인화하는 로직을 검증한다."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.drug_detail_models import DrugDetailApiItem
from app.identify_result import Identification
from app.recommendation import build_recommendation

_KST = timezone(timedelta(hours=9))
_TODAY = date(2026, 8, 20)
_IDENT = Identification(confidence="HIGH", score=1.0)

_TYLENOL_DETAIL = DrugDetailApiItem(
    item_seq="202106092",
    item_name="타이레놀정500밀리그람(아세트아미노펜)",
    efficacy="이 약은 감기로 인한 발열 및 동통(통증), 두통, 신경통, 근육통, 월경통에 사용합니다",
    usage_method="만 12세 이상 소아 및 성인은 1회 1~2정씩 복용합니다.",
    warning="매일 세 잔 이상 정기적 음주자가 이 약을 복용할 때는 의사와 상의하십시오. 간손상을 일으킬 수 있습니다.",
    precautions=(
        "이 약에 과민증, 소화성궤양, 간장애, 신장장애 환자, 알코올을 복용한 사람, "
        "12세 미만의 소아는 이 약을 복용하지 마십시오.\n"
        "임부 또는 수유부, 고령자(노인)는 의사 또는 약사와 상의하십시오.\n"
        "다른 아세트아미노펜 함유 의약품과 함께 복용하지 마십시오."
    ),
    interactions="알코올을 투여한 환자는 의사 또는 약사와 상의하십시오.",
)


def _mock_llm_fail():
    """LLM 호출을 실패시켜 규칙 기반 폴백 로직을 검증한다."""
    return patch(
        "app.recommendation_llm.generate_recommendation_via_llm",
        new_callable=AsyncMock,
        side_effect=Exception("test: LLM disabled"),
    )


@pytest.mark.asyncio
async def test_성인_남성_두통_발열이면_복용_가능하고_소아_임신_주의는_빼다():
    with _mock_llm_fail():
        rec = await build_recommendation(
            identification=_IDENT,
            detail=_TYLENOL_DETAIL,
            symptoms=["HEADACHE", "FEVER"],
            gender="MALE",
            birth_date=date(1995, 12, 30),
            symptom_started_at=datetime(2026, 8, 20, 14, 0, tzinfo=_KST),
            as_of=_TODAY,
        )

    assert rec.status == "RECOMMENDED"
    assert rec.score == 0.94
    assert rec.confidence == "HIGH"
    assert "30세" in rec.reason
    assert "남성" in rec.reason
    assert "두통" in rec.reason
    assert "발열" in rec.reason
    assert "12세 미만" not in rec.caution
    assert "임부" not in rec.caution
    assert "고령자" not in rec.caution
    assert "아세트아미노펜" in rec.caution
    assert "음주" in rec.caution or "알코올" in rec.caution
    assert "사람은 이 약을" in rec.caution


@pytest.mark.asyncio
async def test_12세_미만이면_복용_불가이고_연령_주의를_남긴다():
    with _mock_llm_fail():
        rec = await build_recommendation(
            identification=_IDENT,
            detail=_TYLENOL_DETAIL,
            symptoms=["HEADACHE", "FEVER"],
            gender="MALE",
            birth_date=date(2020, 1, 1),
            as_of=_TODAY,
        )

    assert rec.status == "NOT_RECOMMENDED"
    assert rec.score == 0.2
    assert rec.confidence == "HIGH"
    assert "6세" in rec.reason
    assert "12세" in rec.reason
    assert "12세 미만" in rec.caution


@pytest.mark.asyncio
async def test_증상과_효능이_안_맞으면_복용_불가이고_점수는_낮다():
    with _mock_llm_fail():
        rec = await build_recommendation(
            identification=_IDENT,
            detail=_TYLENOL_DETAIL,
            symptoms=["DIARRHEA"],
            gender="MALE",
            birth_date=date(1995, 12, 30),
            as_of=_TODAY,
        )

    assert rec.status == "NOT_RECOMMENDED"
    assert rec.score == 0.4
    assert rec.confidence == "LOW"
    assert "설사" in rec.reason
    assert "일치하지 않습니다" in rec.reason


@pytest.mark.asyncio
async def test_가임기_여성이면_임부_수유_주의를_남긴다():
    with _mock_llm_fail():
        rec = await build_recommendation(
            identification=_IDENT,
            detail=_TYLENOL_DETAIL,
            symptoms=["HEADACHE"],
            gender="FEMALE",
            birth_date=date(1995, 12, 30),
            as_of=_TODAY,
        )

    assert rec.status == "RECOMMENDED"
    assert rec.score == 0.9
    assert rec.confidence == "MEDIUM"
    assert "여성" in rec.reason
    assert "임부" in rec.caution
    assert "수유" in rec.caution
    assert "12세 미만" not in rec.caution


@pytest.mark.asyncio
async def test_고령자면_고령자_주의를_남긴다():
    with _mock_llm_fail():
        rec = await build_recommendation(
            identification=_IDENT,
            detail=_TYLENOL_DETAIL,
            symptoms=["HEADACHE"],
            gender="MALE",
            birth_date=date(1950, 1, 1),
            as_of=_TODAY,
        )

    assert rec.status == "RECOMMENDED"
    assert rec.score == 0.9
    assert rec.confidence == "MEDIUM"
    assert "76세" in rec.reason
    assert "고령자" in rec.caution


@pytest.mark.asyncio
async def test_증상이_3일_이상이면_지속_주의를_붙인다():
    with _mock_llm_fail():
        rec = await build_recommendation(
            identification=_IDENT,
            detail=_TYLENOL_DETAIL,
            symptoms=["FEVER"],
            gender="MALE",
            birth_date=date(1995, 12, 30),
            symptom_started_at=datetime(2026, 8, 16, 9, 0, tzinfo=_KST),
            as_of=_TODAY,
        )

    assert rec.status == "RECOMMENDED"
    assert rec.confidence == "MEDIUM"
    assert "3일" in rec.caution

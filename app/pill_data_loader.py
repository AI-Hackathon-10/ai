"""
낱알식별 API 전체 데이터를 MySQL pill_identification 테이블에 적재한다.

서버 시작 시 lifespan에서 백그라운드로 호출하거나,
POST /admin/reload-pill-data 엔드포인트로 수동 트리거할 수 있다.
"""
from __future__ import annotations

import asyncio
import logging
import urllib.parse
from typing import Any

import httpx
import pymysql

from app.config import settings

logger = logging.getLogger(__name__)

API_BASE_URL = "https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03"
PAGE_SIZE = 100
CONCURRENCY = 5
TIMEOUT = 10.0

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pill_identification (
    id             BIGINT AUTO_INCREMENT PRIMARY KEY,
    item_seq       VARCHAR(20)   COMMENT '품목코드',
    item_name      VARCHAR(200)  COMMENT '의약품명',
    entp_name      VARCHAR(200)  COMMENT '업체명',
    chart          TEXT          COMMENT '상세설명',
    item_image     VARCHAR(500)  COMMENT '이미지 URL',
    print_front    VARCHAR(100)  COMMENT '앞면 각인',
    print_back     VARCHAR(100)  COMMENT '뒷면 각인',
    drug_shape     VARCHAR(50)   COMMENT '모양',
    color_class1   VARCHAR(50)   COMMENT '주색상',
    color_class2   VARCHAR(50)   COMMENT '부색상',
    line_front     VARCHAR(50)   COMMENT '앞면 분할선',
    line_back      VARCHAR(50)   COMMENT '뒷면 분할선',
    form_code_name VARCHAR(50)   COMMENT '제형',
    UNIQUE KEY uk_item_seq (item_seq)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='식약처 낱알식별 의약품 정보';
"""

INSERT_SQL = """
INSERT INTO pill_identification
    (item_seq, item_name, entp_name, chart, item_image,
     print_front, print_back, drug_shape,
     color_class1, color_class2,
     line_front, line_back, form_code_name)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

FIELDS = [
    "ITEM_SEQ", "ITEM_NAME", "ENTP_NAME", "CHART", "ITEM_IMAGE",
    "PRINT_FRONT", "PRINT_BACK", "DRUG_SHAPE",
    "COLOR_CLASS1", "COLOR_CLASS2",
    "LINE_FRONT", "LINE_BACK", "FORM_CODE_NAME",
]


def _normalize_service_key(raw_key: str) -> str:
    if "%" in raw_key:
        return urllib.parse.unquote(raw_key)
    return raw_key


async def _fetch_page(client: httpx.AsyncClient, key: str, page_no: int) -> dict:
    params = {
        "serviceKey": key,
        "type": "json",
        "pageNo": page_no,
        "numOfRows": PAGE_SIZE,
    }
    resp = await client.get(API_BASE_URL, params=params)
    resp.raise_for_status()
    return resp.json()


async def _fetch_all_items() -> list[dict]:
    key = _normalize_service_key(settings.mfds_pill_service_key)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        data = await _fetch_page(client, key, 1)
        body = data.get("body", {})
        total = body.get("totalCount", 0)
        items = body.get("items", [])

        if not items:
            logger.warning("낱알식별 API 응답에 데이터가 없습니다.")
            return []

        logger.info("낱알식별 API: 전체 %d건, 페이지당 %d건", total, PAGE_SIZE)

        last_page = (total + PAGE_SIZE - 1) // PAGE_SIZE
        if last_page <= 1:
            return items

        sem = asyncio.Semaphore(CONCURRENCY)

        async def fetch_with_sem(page: int) -> list[dict]:
            async with sem:
                d = await _fetch_page(client, key, page)
                return d.get("body", {}).get("items", [])

        tasks = [fetch_with_sem(p) for p in range(2, last_page + 1)]
        results = await asyncio.gather(*tasks)
        for page_items in results:
            items.extend(page_items)

    logger.info("낱알식별 API에서 총 %d건 조회 완료", len(items))
    return items


def _save_to_mysql(items: list[dict[str, Any]]) -> int:
    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_db,
        charset="utf8mb4",
    )

    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()

            # 기존 데이터 전부 삭제 후 새로 적재
            cur.execute("TRUNCATE TABLE pill_identification")
            conn.commit()
            logger.info("기존 pill_identification 데이터 삭제 완료")

            rows = [tuple(item.get(f) for f in FIELDS) for item in items]

            batch_size = 500
            inserted = 0
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                cur.executemany(INSERT_SQL, batch)
                conn.commit()
                inserted += len(batch)

        logger.info("MySQL 적재 완료: %d건", len(rows))
        return len(rows)
    finally:
        conn.close()


async def load_pill_data() -> int:
    """낱알식별 API에서 전체 데이터를 가져와 MySQL에 적재한다.

    Returns:
        적재된 건수. 데이터가 없으면 0.
    """
    logger.info("=== 낱알식별 데이터 적재 시작 ===")
    items = await _fetch_all_items()
    if not items:
        logger.warning("적재할 데이터가 없습니다.")
        return 0
    count = await asyncio.to_thread(_save_to_mysql, items)
    logger.info("=== 낱알식별 데이터 적재 완료: %d건 ===", count)
    return count

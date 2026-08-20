"""
낱알식별 API 전체 데이터를 MySQL에 적재하는 스크립트.

사용법:
    pip install httpx pymysql python-dotenv
    python scripts/load_pill_data.py
"""
import asyncio
import os
import urllib.parse

import httpx
import pymysql
from dotenv import load_dotenv

# .env 로드 (ai/.env → backend/.env 순서)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env"))

# ── API 설정 ──
API_BASE_URL = "https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03"
SERVICE_KEY = os.environ["MFDS_PILL_SERVICE_KEY"]
PAGE_SIZE = 100
CONCURRENCY = 5
TIMEOUT = 10.0

# ── MySQL 설정 ──
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("DB_USERNAME", "root")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "pillcare")

# ── 테이블 DDL ──
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
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    item_name      = VALUES(item_name),
    entp_name      = VALUES(entp_name),
    chart          = VALUES(chart),
    item_image     = VALUES(item_image),
    print_front    = VALUES(print_front),
    print_back     = VALUES(print_back),
    drug_shape     = VALUES(drug_shape),
    color_class1   = VALUES(color_class1),
    color_class2   = VALUES(color_class2),
    line_front     = VALUES(line_front),
    line_back      = VALUES(line_back),
    form_code_name = VALUES(form_code_name);
"""

FIELDS = [
    "ITEM_SEQ", "ITEM_NAME", "ENTP_NAME", "CHART", "ITEM_IMAGE",
    "PRINT_FRONT", "PRINT_BACK", "DRUG_SHAPE",
    "COLOR_CLASS1", "COLOR_CLASS2",
    "LINE_FRONT", "LINE_BACK", "FORM_CODE_NAME",
]


def normalize_service_key(raw_key: str) -> str:
    if "%" in raw_key:
        return urllib.parse.unquote(raw_key)
    return raw_key


async def fetch_page(client: httpx.AsyncClient, key: str, page_no: int) -> dict:
    params = {
        "serviceKey": key,
        "type": "json",
        "pageNo": page_no,
        "numOfRows": PAGE_SIZE,
    }
    resp = await client.get(API_BASE_URL, params=params)
    resp.raise_for_status()
    return resp.json()


async def fetch_all_items() -> list[dict]:
    key = normalize_service_key(SERVICE_KEY)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # 1페이지 조회 → 전체 건수 확인
        data = await fetch_page(client, key, 1)
        body = data.get("body", {})
        total = body.get("totalCount", 0)
        items = body.get("items", [])

        if not items:
            print("API 응답에 데이터가 없습니다.")
            return []

        print(f"전체 {total}건, 페이지당 {PAGE_SIZE}건")

        last_page = (total + PAGE_SIZE - 1) // PAGE_SIZE
        if last_page <= 1:
            return items

        # 나머지 페이지 동시 조회
        sem = asyncio.Semaphore(CONCURRENCY)

        async def fetch_with_sem(page: int) -> list[dict]:
            async with sem:
                d = await fetch_page(client, key, page)
                return d.get("body", {}).get("items", [])

        tasks = [fetch_with_sem(p) for p in range(2, last_page + 1)]
        results = await asyncio.gather(*tasks)
        for page_items in results:
            items.extend(page_items)

    print(f"API에서 총 {len(items)}건 조회 완료")
    return items


def save_to_mysql(items: list[dict]):
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
    )

    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()

            rows = []
            for item in items:
                row = tuple(item.get(f) for f in FIELDS)
                rows.append(row)

            batch_size = 500
            inserted = 0
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                cur.executemany(INSERT_SQL, batch)
                conn.commit()
                inserted += len(batch)
                print(f"  적재 진행: {inserted}/{len(rows)}")

        print(f"MySQL 적재 완료: {len(rows)}건")
    finally:
        conn.close()


async def main():
    print("=== 낱알식별 API → MySQL 적재 시작 ===")
    items = await fetch_all_items()
    if items:
        save_to_mysql(items)
    print("=== 완료 ===")


if __name__ == "__main__":
    asyncio.run(main())

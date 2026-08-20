# PillCare AI Server - API & Architecture Specification

> **Version:** 2.0
> **Base URL:** `http://<host>:8000`
> **Framework:** FastAPI + Uvicorn
> **Last Updated:** 2026-08-20

---

## 1. System Architecture Overview

```
                         +------------------+
                         |   Client (App)   |
                         +--------+---------+
                                  |
                            POST /identify
                         (base64 이미지 전송)
                                  |
                                  v
                    +-------------+-------------+
                    |     FastAPI (main.py)      |
                    |  - 요청 검증 (Pydantic)     |
                    |  - base64 디코딩            |
                    |  - 예외 핸들링              |
                    +-------------+-------------+
                                  |
                                  v
                    +-------------+-------------+
                    |   Pipeline (pipeline.py)   |
                    |   오케스트레이션 계층        |
                    +--+--------+--------+------+
                       |        |        |
              Step 2   |        | Step 3 |  Step 4
                       v        v        v
               +-------+--+ +--+------+ +--+--------+
               |  Vision   | | Matching| |  Detail   |
               |  (Gemini) | | Service | |  Service  |
               +-----------+ +----+----+ +-----------+
                                  |            |
                                  v            v
                              +---+----+ +----+-----+
                              | MySQL  | |e약은요 API|
                              |  DB    | |(data.go. |
                              |        | |  kr)     |
                              +---+----+ +----------+
                                  ^
                                  |
                    +-------------+-------------+
                    | pill_data_loader.py        |
                    | (서버 시작 시 자동 적재      |
                    |  또는 수동 트리거)           |
                    +-------------+-------------+
                                  ^
                                  |
                    +-------------+-------------+
                    | 낱알식별 API (data.go.kr)   |
                    | (데이터 적재 전용)           |
                    +---------------------------+
```

---

## 2. Request/Response Flow (전체 파이프라인)

```
클라이언트 요청
    │
    ▼
[Step 1] base64 디코딩
    │  - data URI → bytes + mime_type 분리
    │  - 순수 base64 → bytes 변환
    │  - 실패 시 → 400 Bad Request
    │
    ▼
[Step 2] Gemini Vision AI 호출
    │  - 알약 이미지 → 각인/색상/모양 추출
    │  - LangGraph 상태 머신으로 재시도 관리 (최대 2회)
    │  - 실패 시 → vision_failed: true
    │
    ▼
[Step 3] DB 매칭
    │  - Vision 결과 → pill_identification 테이블 조회
    │  - 6단계 완화 전략으로 후보 탐색
    │  - 결과: SINGLE_MATCH / MULTIPLE_MATCHES / NO_MATCH
    │
    ▼
[Step 4] e약은요 상세정보 조회 (SINGLE_MATCH일 때만)
    │  - item_seq로 효능/용법/부작용 등 조회
    │
    ▼
클라이언트 응답 (JSON)
```

---

## 3. API Endpoints

### 3.1 `POST /identify` - 알약 식별 (전체 파이프라인)

Vision 추출 → DB 매칭 → 상세정보 조회를 한 번에 수행합니다.
pill_identification MySQL 테이블에서 매칭합니다.

#### Request

```http
POST /identify
Content-Type: application/json
```

```jsonc
{
    "front_image": "data:image/png;base64,iVBORw0KGgo...",  // 필수. base64 또는 data URI
    "back_image": "data:image/png;base64,...",               // 선택. 뒷면 이미지
    "front_mime_type": "image/jpeg",                         // 선택. 순수 base64일 때 MIME (기본: image/jpeg)
    "back_mime_type": "image/jpeg"                           // 선택. 뒷면 MIME
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `front_image` | `string` | **필수** | 앞면 이미지. `data:image/png;base64,...` 형식 또는 순수 base64 문자열 |
| `back_image` | `string` | 선택 | 뒷면 이미지. 형식 동일 |
| `front_mime_type` | `string` | 선택 | 순수 base64일 때 MIME 타입. 기본값: `image/jpeg` |
| `back_mime_type` | `string` | 선택 | 뒷면 순수 base64일 때 MIME 타입 |

> **참고:** `data:image/png;base64,...` 형식으로 보내면 MIME 타입이 자동 추출되므로 `front_mime_type`은 무시됩니다.

#### Response - 성공 (`200 OK`)

```jsonc
{
    "vision_failed": false,
    "match_result": {
        "status": "SINGLE_MATCH",           // 매칭 상태
        "candidates": [                      // 후보 목록
            {
                "item_seq": "199900001",     // 품목일련번호
                "item_name": "타이레놀정500mg",
                "entp_name": "한국얀센",
                "chart": "흰색의 장방형 필름코팅정",
                "item_image": "https://nedrug.mfds.go.kr/pbp/cmn/...",
                "print_front": "TY",
                "print_back": "500",
                "drug_shape": "장방형",
                "color_class1": "하양",
                "color_class2": null,
                "line_front": null,
                "line_back": null,
                "form_code_name": "필름코팅정"
            }
        ],
        "query_level_used": "STRICT_WITH_PRINT_BOTH"  // 사용된 매칭 레벨
    },
    "detail": {                              // SINGLE_MATCH일 때만 포함
        "item_seq": "199900001",
        "item_name": "타이레놀정500mg",
        "entp_name": "한국얀센",
        "item_image": "https://nedrug.mfds.go.kr/...",
        "efficacy": "감기로 인한 발열 및 동통...",
        "usage_method": "만 12세 이상 소아 및 성인: 1회 1~2정...",
        "warning": "매일 세잔 이상 정기적으로 술을 마시는 사람...",
        "precautions": "다음과 같은 사람은 이 약을 복용하지 말 것...",
        "interactions": "이 약을 복용하는 동안 다른 해열진통제...",
        "side_effects": "쇽, 아나필락시스 유사 증상...",
        "storage_method": "실온(1~30도)에서 보관"
    }
}
```

#### Response - Vision 실패

```json
{
    "vision_failed": true,
    "match_result": null,
    "detail": null
}
```

#### Response - 후보 복수 (`MULTIPLE_MATCHES`)

```jsonc
{
    "vision_failed": false,
    "match_result": {
        "status": "MULTIPLE_MATCHES",
        "candidates": [
            { "item_seq": "199900001", "item_name": "..." },
            { "item_seq": "199900002", "item_name": "..." }
            // 최대 10건
        ],
        "query_level_used": "STRICT_WITH_PRINT_FRONT_ONLY"
    },
    "detail": null   // MULTIPLE_MATCHES이면 detail 없음
}
```

#### Response - 매칭 실패 (`NO_MATCH`)

```json
{
    "vision_failed": false,
    "match_result": {
        "status": "NO_MATCH",
        "candidates": [],
        "query_level_used": "COLOR_ONLY"
    },
    "detail": null
}
```

#### Error Responses

| 상태 코드 | 조건 | 응답 |
|-----------|------|------|
| `400` | `front_image` 누락 또는 빈 문자열 | `{"detail": "front_image는 필수이며 빈 문자열일 수 없습니다."}` |
| `400` | base64 디코딩 실패 | `{"detail": "Invalid base64 in field 'front_image'"}` |
| `503` | Gemini Vision API 오류 | `{"detail": "Gemini vision error: ..."}` |
| `500` | DB 조회 오류 | `{"detail": "..."}` |

---

### 3.2 `POST /vision/extract` - Vision 추출만

Gemini Vision AI로 이미지를 분석하여 각인/색상/모양만 추출합니다. 매칭은 수행하지 않습니다.

#### Request

`POST /identify`와 동일합니다.

#### Response (`200 OK`)

```jsonc
{
    "vision_failed": false,
    "result": {
        "print_front": "TY",               // 앞면 각인 (null 가능)
        "print_front_confidence": 0.95,     // 각인 신뢰도 0.0~1.0
        "print_back": "500",                // 뒷면 각인 (null 가능)
        "print_back_confidence": 0.90,
        "color_class1": "하양",             // 주색상
        "color_class2": null,               // 부색상 (없으면 null)
        "drug_shape": "장방형",             // 모양
        "line_front": null,                 // 앞면 분할선 (null 가능)
        "line_back": null,                  // 뒷면 분할선 (null 가능)
        "form_code_name": "정제",           // 제형 (null 가능)
        "overall_confidence": 0.92          // 전체 신뢰도
    }
}
```

---

### 3.3 `POST /identify/db` - Vision + DB 매칭 (Vision 결과 포함)

Vision 추출 후 DB 매칭 결과와 함께 **Vision 원시 추출 결과도** 응답에 포함합니다.
`/identify`와 동일한 DB 매칭을 사용하지만 상세정보(Step 4)는 수행하지 않고, 대신 Vision 결과를 확인할 수 있습니다.

#### Request

`POST /identify`와 동일합니다.

#### Response (`200 OK`)

```jsonc
{
    "vision_failed": false,
    "vision_result": {
        "print_front": "TY",
        "print_front_confidence": 0.95,
        "print_back": "500",
        "print_back_confidence": 0.90,
        "color_class1": "하양",
        "color_class2": null,
        "drug_shape": "장방형",
        "line_front": null,
        "line_back": null,
        "form_code_name": "정제",
        "overall_confidence": 0.92
    },
    "match_result": {
        "status": "SINGLE_MATCH",
        "candidates": [ /* PillCandidate 객체들 */ ],
        "query_level_used": "STRICT_WITH_PRINT_BOTH"
    }
}
```

---

### 3.4 `POST /identify/select/{item_seq}` - 후보 선택 (상세정보 조회)

`MULTIPLE_MATCHES`일 때, 사용자가 선택한 후보의 상세정보를 조회합니다.

#### Request

```http
POST /identify/select/199900001
```

| 파라미터 | 위치 | 타입 | 설명 |
|---------|------|------|------|
| `item_seq` | Path | `string` | 품목일련번호 |

#### Response (`200 OK`)

```jsonc
{
    "detail": {
        "item_seq": "199900001",
        "item_name": "타이레놀정500mg",
        "entp_name": "한국얀센",
        "item_image": "https://nedrug.mfds.go.kr/...",
        "efficacy": "감기로 인한 발열 및 동통...",
        "usage_method": "만 12세 이상 소아 및 성인...",
        "warning": "...",
        "precautions": "...",
        "interactions": "...",
        "side_effects": "...",
        "storage_method": "실온(1~30도)에서 보관"
    }
}
```

#### Response - 해당 약품 없음

```json
{
    "detail": null
}
```

---

### 3.5 `POST /admin/reload-pill-data` - 낱알식별 데이터 수동 적재

낱알식별 API에서 전체 데이터를 다시 가져와 DB에 적재합니다.
기존 데이터를 **TRUNCATE** 후 새로 INSERT합니다.

#### Request

```http
POST /admin/reload-pill-data
```

요청 바디 없음.

#### Response - 성공 (`200 OK`)

```json
{
    "status": "ok",
    "loaded_count": 25847
}
```

#### Response - 실패 (`500`)

```json
{
    "detail": "데이터 적재 실패: ..."
}
```

---

### 3.6 `GET /health` - 헬스 체크

```http
GET /health
```

```json
{"status": "ok"}
```

---

## 4. Data Models

### 4.1 Request Models

#### `IdentifyRequest`

```
+---------------------+------------------+---------+----------------------------+
| Field               | Type             | Required| Description                |
+---------------------+------------------+---------+----------------------------+
| front_image         | string           | Yes     | 앞면 이미지 (base64/data URI)|
| back_image          | string | null    | No      | 뒷면 이미지                 |
| front_mime_type     | string           | No      | MIME 타입 (기본: image/jpeg) |
| back_mime_type      | string | null    | No      | 뒷면 MIME 타입              |
+---------------------+------------------+---------+----------------------------+
```

### 4.2 Response Models

#### `IdentifyResponse` (`POST /identify`)

```
+---------------------+---------------------------+-------------------------------+
| Field               | Type                      | Description                   |
+---------------------+---------------------------+-------------------------------+
| vision_failed       | boolean                   | Vision 추출 실패 여부          |
| match_result        | PillMatchResult | null    | 매칭 결과                     |
| detail              | DrugDetailApiItem | null  | 상세정보 (SINGLE_MATCH만)     |
+---------------------+---------------------------+-------------------------------+
```

#### `VisionExtractResponse` (`POST /vision/extract`)

```
+---------------------+----------------------------------+------------------------+
| Field               | Type                             | Description            |
+---------------------+----------------------------------+------------------------+
| vision_failed       | boolean                          | Vision 추출 실패 여부   |
| result              | VisionExtractionResult | null   | Vision 추출 결과        |
+---------------------+----------------------------------+------------------------+
```

#### `IdentifyDbResponse` (`POST /identify/db`)

```
+---------------------+----------------------------------+------------------------+
| Field               | Type                             | Description            |
+---------------------+----------------------------------+------------------------+
| vision_failed       | boolean                          | Vision 추출 실패 여부   |
| vision_result       | VisionExtractionResult | null   | Vision 추출 결과        |
| match_result        | PillMatchResult | null           | 매칭 결과              |
+---------------------+----------------------------------+------------------------+
```

#### `SelectCandidateResponse` (`POST /identify/select/{item_seq}`)

```
+---------------------+---------------------------+-------------------------------+
| Field               | Type                      | Description                   |
+---------------------+---------------------------+-------------------------------+
| detail              | DrugDetailApiItem | null  | 상세정보 (없으면 null)        |
+---------------------+---------------------------+-------------------------------+
```

### 4.3 Core Domain Models

#### `VisionExtractionResult`

Gemini Vision AI가 이미지에서 추출한 알약 특성입니다.

```
+-------------------------+-----------------+-----------------------------------+
| Field                   | Type            | Description                       |
+-------------------------+-----------------+-----------------------------------+
| print_front             | string | null   | 앞면 각인 문자                     |
| print_front_confidence  | float | null    | 앞면 각인 신뢰도 (0.0~1.0)        |
| print_back              | string | null   | 뒷면 각인 문자                     |
| print_back_confidence   | float | null    | 뒷면 각인 신뢰도 (0.0~1.0)        |
| color_class1            | string | null   | 주색상 (허용값 참조)               |
| color_class2            | string | null   | 부색상 (없으면 null)               |
| drug_shape              | string | null   | 모양 (허용값 참조)                 |
| line_front              | string | null   | 앞면 분할선                        |
| line_back               | string | null   | 뒷면 분할선                        |
| form_code_name          | string | null   | 제형 (정제, 캡슐 등)               |
| overall_confidence      | float | null    | 전체 신뢰도                        |
+-------------------------+-----------------+-----------------------------------+
```

**색상 허용값:** `하양`, `노랑`, `주황`, `분홍`, `빨강`, `갈색`, `연두`, `초록`, `청록`, `파랑`, `남색`, `자주`, `보라`, `회색`, `검정`, `투명`

**모양 허용값:** `원형`, `타원형`, `장방형`, `삼각형`, `사각형`, `마름모형`, `오각형`, `육각형`, `팔각형`, `반원형`, `기타`

#### `PillMatchResult`

```
+---------------------+---------------------+-----------------------------------+
| Field               | Type                | Description                       |
+---------------------+---------------------+-----------------------------------+
| status              | MatchStatus (enum)  | 매칭 결과 상태                     |
| candidates          | PillCandidate[]     | 후보 의약품 목록 (0~10건)          |
| query_level_used    | string              | 최종 사용된 매칭 레벨              |
+---------------------+---------------------+-----------------------------------+
```

**`MatchStatus` enum:**

| 값 | 설명 |
|----|------|
| `SINGLE_MATCH` | 정확히 1건 매칭 → 자동으로 상세정보 조회 |
| `MULTIPLE_MATCHES` | 2~10건 매칭 → 사용자 선택 필요 |
| `NO_MATCH` | 0건 → 모든 레벨에서 매칭 실패 |
| `TOO_MANY_CANDIDATES` | 10건 초과 → 범위가 너무 넓음 |

**`query_level_used` 값:**

| 레벨 | 설명 |
|------|------|
| `STRICT_WITH_PRINT_BOTH` | 앞/뒷면 각인 + 색상 + 모양 |
| `STRICT_WITH_PRINT_FRONT_ONLY` | 앞면 각인 + 색상 + 모양 |
| `STRICT_WITH_PRINT_BACK_ONLY` | 뒷면 각인 + 색상 + 모양 |
| `SHAPE_AND_COLOR` | 모양 + 주색상 + 부색상 |
| `SHAPE_AND_PRIMARY_COLOR` | 모양 + 주색상 |
| `COLOR_ONLY` | 주색상만 |

#### `PillCandidate`

```
+---------------------+------------------+--------------------------------------+
| Field               | Type             | Description                          |
+---------------------+------------------+--------------------------------------+
| item_seq            | string | null    | 품목일련번호 (식약처 고유 코드)       |
| item_name           | string | null    | 의약품명                             |
| entp_name           | string | null    | 제조/수입 업체명                      |
| chart               | string | null    | 성상 설명                            |
| item_image          | string | null    | 알약 이미지 URL                      |
| print_front         | string | null    | 앞면 각인                            |
| print_back          | string | null    | 뒷면 각인                            |
| drug_shape          | string | null    | 모양                                 |
| color_class1        | string | null    | 주색상                               |
| color_class2        | string | null    | 부색상                               |
| line_front          | string | null    | 앞면 분할선                           |
| line_back           | string | null    | 뒷면 분할선                           |
| form_code_name      | string | null    | 제형                                 |
+---------------------+------------------+--------------------------------------+
```

#### `DrugDetailApiItem`

```
+---------------------+------------------+--------------------------------------+
| Field               | Type             | Description                          |
+---------------------+------------------+--------------------------------------+
| item_seq            | string | null    | 품목일련번호                          |
| item_name           | string | null    | 의약품명                             |
| entp_name           | string | null    | 업체명                               |
| item_image          | string | null    | 이미지 URL                           |
| efficacy            | string | null    | 효능효과                             |
| usage_method        | string | null    | 용법용량                             |
| warning             | string | null    | 경고사항                             |
| precautions         | string | null    | 사용상 주의사항                       |
| interactions        | string | null    | 상호작용                             |
| side_effects        | string | null    | 부작용                               |
| storage_method      | string | null    | 보관방법                             |
+---------------------+------------------+--------------------------------------+
```

---

## 5. LangGraph Vision State Machine

Vision 추출은 LangGraph 상태 머신으로 재시도 로직을 관리합니다.

### 5.1 Graph Structure

```
            ┌──────────┐
            │  START    │
            └────┬─────┘
                 │
                 ▼
        ┌────────────────┐
        │  extract_node  │  Gemini Vision API 호출
        │                │  → 원시 dict 응답 저장
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │ validate_node  │  color_class1 & drug_shape 존재 확인
        └───┬────┬───┬───┘
            │    │   │
     valid  │    │   │ invalid & attempts < MAX
            │    │   │
            ▼    │   ▼
  ┌─────────┐   │  ┌───────────────────┐
  │finalize │   │  │   extract_node    │  재시도
  │ success │   │  │ (retry_hint 포함) │
  └────┬────┘   │  └───────────────────┘
       │        │
       │        │ invalid & attempts >= MAX
       │        │
       │        ▼
       │  ┌─────────────┐
       │  │  finalize    │
       │  │  failure     │  → result = None
       │  └──────┬──────┘
       │         │
       ▼         ▼
    ┌──────────────┐
    │     END      │
    └──────────────┘
```

### 5.2 State Schema

```python
{
    "front_image_bytes": bytes,         # 입력 이미지
    "back_image_bytes": bytes | None,   # 뒷면 이미지
    "front_mime_type": str,             # MIME 타입
    "back_mime_type": str | None,
    "raw_response": dict | None,        # Gemini 원시 응답
    "result": VisionExtractionResult | None,  # 최종 결과
    "attempts": int,                    # 현재 시도 횟수
    "retry_hint": str | None           # 재시도 시 추가 힌트
}
```

### 5.3 Constants

| 상수 | 값 | 설명 |
|------|---|------|
| `MAX_ATTEMPTS` | `2` | 최대 재시도 횟수 |
| `PRINT_CONFIDENCE_THRESHOLD` | `0.6` | 각인 신뢰도 임계값 (이하면 각인 무시) |
| `MAX_CANDIDATES` | `10` | 후보 상한선 |

### 5.4 Validation Rules

`validate_node`에서 검증하는 조건:
- `color_class1`이 존재해야 함 (주색상 필수)
- `drug_shape`이 존재해야 함 (모양 필수)
- 둘 중 하나라도 없으면 재시도 (retry_hint에 누락 필드 명시)

---

## 6. Matching Algorithm (6-Level Relaxation)

매칭 서비스는 Vision 결과를 기반으로 pill_identification DB 테이블을 조회하며,
6단계 완화 전략을 사용합니다.

```
Level 1: STRICT_WITH_PRINT_BOTH
   │  앞면각인 + 뒷면각인 + 주색상 + 부색상 + 모양
   │  (각인 신뢰도 >= 0.6인 것만 사용)
   │
   │  0건 → 다음 레벨
   ▼
Level 2: STRICT_WITH_PRINT_FRONT_ONLY
   │  앞면각인 + 주색상 + 부색상 + 모양
   │
   │  0건 → 다음 레벨
   ▼
Level 3: STRICT_WITH_PRINT_BACK_ONLY
   │  뒷면각인 + 주색상 + 부색상 + 모양
   │
   │  0건 → 다음 레벨
   ▼
Level 4: SHAPE_AND_COLOR
   │  모양 + 주색상 + 부색상 (각인 없이)
   │
   │  0건 → 다음 레벨
   ▼
Level 5: SHAPE_AND_PRIMARY_COLOR
   │  모양 + 주색상만
   │
   │  0건 → 다음 레벨
   ▼
Level 6: COLOR_ONLY
      주색상만
```

### 결정 규칙

각 레벨에서:
- **0건** → 다음 레벨로 진행
- **1건** → `SINGLE_MATCH` 반환 → Step 4 자동 실행
- **2~10건** → `MULTIPLE_MATCHES` 반환 → 사용자 선택 대기
- **11건 이상** → `TOO_MANY_CANDIDATES` 반환
- **모든 레벨 통과 후 0건** → `NO_MATCH` 반환

### DB 쿼리 방식

각 레벨의 필터 조건을 SQL WHERE 절로 변환하여 pill_identification 테이블을 직접 조회합니다.

- 각인 비교: `REPLACE(LOWER(IFNULL(column, '')), ' ', '')` — 공백/대소문자 무시
- 색상/모양 비교: `TRIM(column) = ?`
- LIMIT: `MAX_CANDIDATES + 1` (11건) — 초과 여부만 판단

---

## 7. External API Integrations

### 7.1 Gemini Vision API (Google)

| 항목 | 값 |
|------|---|
| **모델** | `gemini-3.7-flash` |
| **인증** | `GEMINI_API_KEY` 환경변수 |
| **입력** | base64 이미지 bytes + 구조화 프롬프트 |
| **출력** | JSON (response_schema로 구조 강제) |
| **재시도** | LangGraph로 최대 2회 |

### 7.2 낱알식별 API (식약처 / data.go.kr) — 데이터 적재 전용

실시간 매칭에는 사용하지 않습니다. 서버 시작 시 또는 수동 트리거로
전체 데이터를 가져와 MySQL에 적재하는 용도로만 사용합니다.

| 항목 | 값 |
|------|---|
| **URL** | `https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03` |
| **메서드** | `GET` |
| **인증** | `serviceKey` 쿼리 파라미터 |
| **다운로드** | 전체 페이지 병렬 다운로드 (동시성=5, 페이지당 100건) |
| **적재 방식** | TRUNCATE 후 INSERT (기존 데이터 삭제 → 새로 적재) |
| **배치 크기** | 500건 |
| **트리거 시점** | 서버 시작 시 자동 (백그라운드) / `POST /admin/reload-pill-data` 수동 |

### 7.3 e약은요 API (식약처 / data.go.kr)

| 항목 | 값 |
|------|---|
| **URL** | `https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList` |
| **메서드** | `GET` |
| **인증** | `serviceKey` 쿼리 파라미터 |
| **요청** | `itemSeq` 파라미터 |
| **필드 매핑** | 아래 표 참조 |

**e약은요 필드 매핑:**

| API 원본 필드 | 변환 필드 | 설명 |
|--------------|----------|------|
| `efcyQesitm` | `efficacy` | 효능효과 |
| `useMethodQesitm` | `usage_method` | 용법용량 |
| `atpnWarnQesitm` | `warning` | 경고사항 |
| `atpnQesitm` | `precautions` | 주의사항 |
| `intrcQesitm` | `interactions` | 상호작용 |
| `seQesitm` | `side_effects` | 부작용 |
| `depositMethodQesitm` | `storage_method` | 보관방법 |

---

## 8. Database Schema

### 8.1 `pill_identification` 테이블

```sql
CREATE TABLE pill_identification (
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 8.2 데이터 적재 방식

- **적재 전략:** TRUNCATE 후 INSERT (기존 데이터 전체 삭제 → 새 데이터 적재)
- **배치 크기:** 500건씩 커밋
- **트리거 시점:**
  1. 서버 시작 시 자동 (백그라운드 — 적재 완료 전에도 서버는 요청 수신 가능)
  2. `POST /admin/reload-pill-data` 수동 트리거

### 8.3 쿼리 특성

- 각인 비교: `REPLACE(LOWER(IFNULL(column, '')), ' ', '')` — 공백 제거 + 대소문자 무시
- 색상/모양: `TRIM()` 후 비교
- LIMIT: 매칭 시 `MAX_CANDIDATES + 1 = 11`건만 조회

---

## 9. Server Startup & Data Loading

### 9.1 시작 시 자동 적재

```
서버 시작 (uvicorn)
    │
    ▼
FastAPI lifespan 진입
    │
    ├── 백그라운드 태스크 생성: load_pill_data()
    │       │
    │       ├── 낱알식별 API에서 전체 데이터 다운로드
    │       ├── pill_identification 테이블 TRUNCATE
    │       └── 새 데이터 INSERT (500건 배치)
    │
    └── 서버 요청 수신 시작 (적재 완료 대기 안 함)
```

- 적재가 완료되기 전에도 서버는 **즉시 요청을 받을 수 있음** (기존 DB 데이터 사용)
- 적재 실패 시 서버는 정상 동작하며, 로그에 경고만 남김

### 9.2 수동 트리거

```bash
curl -X POST http://localhost:8000/admin/reload-pill-data
```

응답: `{"status": "ok", "loaded_count": 25847}`

---

## 10. Configuration

### 환경변수

| 변수명 | 필수 | 기본값 | 설명 |
|-------|------|--------|------|
| `MFDS_PILL_SERVICE_KEY` | **필수** | - | data.go.kr API 서비스 키 |
| `GEMINI_API_KEY` | 선택 | `""` | Google Gemini API 키 |
| `MYSQL_HOST` | 선택 | `127.0.0.1` | MySQL 호스트 |
| `MYSQL_PORT` | 선택 | `3306` | MySQL 포트 |
| `MYSQL_USER` / `DB_USERNAME` | 선택 | `root` | MySQL 사용자 |
| `MYSQL_PASSWORD` / `DB_PASSWORD` | 선택 | `""` | MySQL 비밀번호 |
| `MYSQL_DB` | 선택 | `pillcare` | MySQL 데이터베이스명 |

### `.env` 파일 로드 순서

1. `ai/.env`
2. `backend/.env`

### 서비스 키 정규화

data.go.kr은 URL 인코딩된("Encoding") 키를 제공합니다. httpx가 다시 인코딩하므로, `%` 문자가 포함된 키는 한 번 디코딩하여 사용합니다.

---

## 11. Error Handling

### Custom Exceptions

| Exception | HTTP Status | 발생 시점 |
|-----------|-------------|----------|
| `InvalidBase64ImageError` | `400` | base64 디코딩 실패 |
| `GeminiVisionError` | `503` | Gemini API 호출/파싱 실패 |
| `DrugDetailApiError` | `500` | e약은요 API 호출/파싱 실패 |

### 에러 응답 형식

```json
{
    "detail": "에러 메시지"
}
```

---

## 12. Project File Structure

```
ai/
├── app/
│   ├── __init__.py
│   ├── main.py                             # FastAPI 앱 + 라우터
│   ├── config.py                           # 환경변수 설정
│   ├── models.py                           # Pydantic 요청/응답 모델
│   ├── base64_images.py                    # base64 디코딩/인코딩
│   ├── pipeline.py                         # Step 2→3→4 오케스트레이션
│   ├── pill_matching_service.py            # 6단계 매칭 로직
│   ├── pill_data_loader.py                 # 낱알식별 API→DB 적재 (시작 시/수동)
│   ├── pill_identification_repository.py   # MySQL 저장소 (매칭 쿼리)
│   ├── drug_detail_service.py              # 상세정보 서비스
│   ├── drug_detail_client.py               # e약은요 API 클라이언트
│   ├── drug_detail_models.py               # e약은요 응답 모델
│   └── vision/
│       ├── __init__.py
│       ├── run.py                          # Vision 진입점
│       ├── graph.py                        # LangGraph 상태 머신
│       ├── types.py                        # Protocol 타입
│       ├── schema.py                       # Gemini 응답 스키마 + 프롬프트
│       └── gemini_client.py                # Gemini Vision API 호출
├── tests/                                  # pytest 테스트
├── scripts/
│   ├── load_pill_data.py                   # MySQL 데이터 적재 (독립 실행 스크립트)
│   └── deploy.sh                           # 배포 스크립트
├── .github/workflows/deploy.yml            # GitHub Actions CI/CD
├── .env                                    # 환경변수 (gitignore)
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 13. Client Integration Guide

### 기본 흐름 (프론트엔드 → 백엔드)

```
1. 사용자가 알약 사진 촬영
       │
       ▼
2. 앞면/뒷면 이미지를 base64로 인코딩
       │
       ▼
3. POST /identify 호출
       │
       ├── vision_failed: true  →  "인식 실패" 안내
       │
       ├── status: SINGLE_MATCH  →  detail 필드로 상세정보 즉시 표시
       │
       ├── status: MULTIPLE_MATCHES  →  candidates 목록 표시
       │       │                         사용자가 하나 선택
       │       ▼
       │   POST /identify/select/{item_seq}  →  상세정보 표시
       │
       ├── status: NO_MATCH  →  "매칭 실패" 안내
       │
       └── status: TOO_MANY_CANDIDATES  →  "결과가 너무 많음" 안내
```

### cURL 예시

```bash
# 전체 파이프라인 (Vision → DB 매칭 → 상세정보)
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d '{
    "front_image": "data:image/jpeg;base64,/9j/4AAQ..."
  }'

# 후보 선택 (MULTIPLE_MATCHES 후)
curl -X POST http://localhost:8000/identify/select/199900001

# 낱알식별 데이터 수동 적재
curl -X POST http://localhost:8000/admin/reload-pill-data

# 헬스 체크
curl http://localhost:8000/health
```

---

## 14. Dependencies

```
fastapi>=0.110          # 웹 프레임워크
uvicorn[standard]>=0.30 # ASGI 서버
python-multipart>=0.0.9 # multipart 요청 지원
pydantic>=2.5           # 데이터 검증
pydantic-settings>=2.2  # 환경변수 바인딩
httpx>=0.27             # 비동기 HTTP 클라이언트
langgraph>=0.2          # 상태 머신 (Vision 재시도)
google-genai>=0.3       # Gemini Vision API
pymysql>=1.1            # MySQL 드라이버
pytest>=8.0             # 테스트
pytest-asyncio>=0.23    # 비동기 테스트
```

---

## 15. Deployment

### CI/CD Pipeline

```
GitHub main 브랜치 push
        │
        ▼
GitHub Actions (deploy.yml)
        │
        ▼
SSH → EC2 서버
        │
        ▼
scripts/deploy.sh 실행
  - 환경변수 주입
  - pip install
  - 서버 재시작
  - (서버 시작 시 낱알식별 데이터 자동 적재)
```

### Secrets (GitHub Actions)

| Secret | 용도 |
|--------|------|
| `EC2_HOST` | 배포 대상 EC2 호스트 |
| `EC2_USERNAME` | SSH 사용자명 |
| `EC2_SSH_KEY` | SSH 개인키 |
| `MFDS_PILL_SERVICE_KEY` | 식약처 API 키 |
| `GEMINI_API_KEY` | Google Gemini 키 |
| `MYSQL_HOST` | MySQL 호스트 |
| `MYSQL_PORT` | MySQL 포트 |
| `MYSQL_USER` | MySQL 사용자 |
| `MYSQL_PASSWORD` | MySQL 비밀번호 |
| `MYSQL_DB` | MySQL DB명 |

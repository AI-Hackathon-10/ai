# Vision 멀티 모델 가이드

알약 식별 Vision(Step 2)에서 Gemini와 OpenAI를 환경변수 하나로 전환할 수 있다.

## 지원 프로바이더

| 프로바이더 | `VISION_PROVIDER` 값 | 기본 모델 | 필요한 API Key |
|-----------|---------------------|----------|---------------|
| Google Gemini | `gemini` (기본값) | `gemini-3.5-flash` | `GEMINI_API_KEY` |
| OpenAI | `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |

## 환경변수 설정

`.env` 파일에 아래 변수를 추가한다.

```env
# 프로바이더 선택: "gemini" 또는 "openai"
VISION_PROVIDER=gemini

# 모델명 (비워두면 프로바이더별 기본값 사용)
VISION_MODEL=gemini-3.5-flash

# API Keys (사용할 프로바이더의 키만 있으면 됨)
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
```

### OpenAI로 전환하기

```env
VISION_PROVIDER=openai
VISION_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

### Gemini로 전환하기 (기본값)

```env
VISION_PROVIDER=gemini
VISION_MODEL=gemini-3.5-flash
GEMINI_API_KEY=your-gemini-key
```

`VISION_MODEL`을 비워두면 프로바이더별 기본 모델이 자동 선택된다.

## 새 프로바이더 추가하는 방법

1. `app/vision/` 아래에 `{provider}_client.py` 파일을 만든다.
2. `VisionCallFn` 시그니처를 지키는 async 함수를 구현한다:
   ```python
   async def call_{provider}_vision(
       front_image_bytes: bytes,
       back_image_bytes: Optional[bytes],
       front_mime_type: str = "image/jpeg",
       back_mime_type: Optional[str] = None,
   ) -> dict:
       ...
   ```
3. 반환값은 `app/vision/schema.py`의 `PILL_VISION_RESPONSE_SCHEMA` 형태를 따라야 한다.
4. `app/vision/factory.py`의 `get_vision_call()`에 분기를 추가한다.
5. 필요한 경우 `app/config.py`에 API 키 설정을 추가한다.

## 아키텍처

```
app/vision/
├── types.py          # VisionCallFn Protocol (프로바이더 공통 시그니처)
├── schema.py         # 프롬프트 + JSON 스키마 (프로바이더 공통)
├── gemini_client.py  # Gemini 구현체
├── openai_client.py  # OpenAI 구현체
├── factory.py        # VISION_PROVIDER에 따라 구현체 선택
├── graph.py          # LangGraph 그래프 (프로바이더 무관)
└── run.py            # 진입점 — factory에서 받은 함수로 그래프 빌드
```

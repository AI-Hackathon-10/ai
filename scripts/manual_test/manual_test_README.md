# scripts/manual_test — 음성 증상 추천 파이프라인 수동 테스트

`tests/`(pytest 자동 테스트)와 별개로, **실제 CLOVA Speech / 실제 서버**를 띄워놓고 사람이 직접 확인하는 스크립트 모음입니다. CI에서 자동 실행되지 않으니, 로컬에서 손으로 실행할 때만 씁니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `nest.proto` | CLOVA Speech 실시간 스트리밍 gRPC 스펙 원본 |
| `nest_pb2.py`, `nest_pb2_grpc.py` | `nest.proto`로부터 자동 생성되는 파일 (git에는 안 올라감 — 아래 "최초 1회 설정" 참고) |
| `clova_streaming_client.py` | 마이크로 음성을 녹음해서 CLOVA로 실시간 스트리밍 인식하는 클라이언트 |
| `text_to_recommend.py` | 마이크 없이, 증상/발생시각을 **타이핑**으로 입력해서 `POST /symptoms/recommend`가 제대로 동작하는지 빠르게 확인 |
| `voice_to_recommend.py` | 실제 **마이크**로 증상/발생시각을 말해서 끝까지(STT → 추천) 확인 |

## 최초 1회 설정

```bash
cd ai
source .venv/bin/activate       # 프로젝트 venv 활성화 (없으면 먼저 uv venv로 생성)
uv pip install grpcio grpcio-tools sounddevice python-dotenv

cd scripts/manual_test
python3 -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. nest.proto
```

마지막 명령을 실행하면 `nest_pb2.py`, `nest_pb2_grpc.py`가 이 폴더에 새로 생깁니다(둘 다 `.gitignore`에 등록되어 있어 커밋 대상이 아닙니다 — 사람마다 로컬에서 새로 생성해서 씁니다).

## `.env` 준비

`ai/.env`에 아래 값들이 필요합니다 (이미 프로젝트 루트 `.env`를 쓰고 있다면 그대로 재사용됩니다).

```
CLOVA_SECRET_KEY=...        # voice_to_recommend.py / clova_streaming_client.py 에 필요
GEMINI_API_KEY=...          # 또는 OPENAI_API_KEY (app/symptom/run.py가 어느 쪽을 가리키는지에 따라)
MFDS_PILL_SERVICE_KEY=...   # e약은요 API
```

## 실행 방법

**터미널 A — AI 서버**
```bash
cd ai
source .venv/bin/activate
uvicorn app.main:app --reload
```

**터미널 B — 텍스트 입력으로 추천 로직만 확인** (마이크/CLOVA 불필요)
```bash
cd ai/scripts/manual_test
python3 text_to_recommend.py
```
증상/발생시각을 타이핑으로 입력하면, 사람이 보기 좋은 요약 + 서버가 실제로 반환하는 JSON 원문이 그대로 출력됩니다.

**터미널 B — 실제 마이크로 끝까지 확인**
```bash
cd ai/scripts/manual_test
python3 voice_to_recommend.py
```
증상과 발생시각을 마이크에 대고 말하면, CLOVA로 인식된 텍스트가 그대로 추천 API로 전송됩니다.

## 자주 나는 문제

- `ModuleNotFoundError: No module named 'grpc'` → venv가 활성화 안 된 상태입니다. 새 터미널을 열 때마다 `source ai/.venv/bin/activate`를 다시 해줘야 합니다.
- `.env를 못 찾았습니다` → `clova_streaming_client.py` 상단의 `_ENV_CANDIDATES` 경로 목록에 실제 `.env` 위치를 추가하세요.
- 서버가 `503`으로 응답 → `GEMINI_API_KEY`/`OPENAI_API_KEY` 할당량 소진 또는 키 미설정일 가능성이 높습니다. 서버 로그(터미널 A)에 찍힌 에러 메시지를 확인하세요.

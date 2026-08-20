"""
CLOVA Speech Recognition - 실시간 스트리밍 인식 (gRPC) 클라이언트 데모

사전 준비
---------
1) nest.proto를 컴파일해서 nest_pb2.py / nest_pb2_grpc.py를 같은 폴더에 생성해야 한다.
       python3 -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. nest.proto

2) 패키지 설치
       pip install grpcio grpcio-tools sounddevice

3) CLOVA_SECRET_KEY를 .env 파일에서 읽어온다. 아래 _ENV_CANDIDATES에 있는 경로
   중 하나에 .env가 있고 그 안에 CLOVA_SECRET_KEY=... 줄이 있으면 자동으로
   불러온다. 다른 경로에 있다면 _ENV_CANDIDATES에 그 경로를 추가하거나,
   터미널에서 직접 `export CLOVA_SECRET_KEY=...`로 지정해도 된다.
   (python-dotenv 필요: pip install python-dotenv)

동작 방식(요약)
--------------
- gRPC의 recognize()는 "요청 스트림 -> 응답 스트림"을 주고받는 양방향 스트리밍 RPC다.
- 맨 처음 딱 1번: type=CONFIG 메시지로 언어/키워드부스팅/EPD 등 옵션을 서버에 전달한다.
- 이후 계속: 마이크에서 들어오는 PCM(16kHz, mono, 16bit, 헤더 없음) 조각을
  type=DATA 메시지로 실어 보낸다. 조각마다 extra_contents(JSON)에 seqId/epFlag를 같이 보낸다.
- 서버는 한 문장이 끊길 때마다(묵음/구두점/길이 등 EPD 기준) transcription 결과를
  응답 스트림으로 흘려준다. 별도 스레드/루프에서 이 응답을 계속 읽어서 처리하면 된다.
- 녹음을 끝낼 때는 마지막 DATA 메시지의 epFlag를 true로 보내면 그 순간까지의
  버퍼를 즉시 flush해서 결과를 받을 수 있다.

이 스크립트는 로컬 마이크로 바로 테스트할 수 있는 데모다.
실제 서비스에서는(EC2 서버에는 마이크가 없으므로) 프론트엔드가 마이크로 캡처한
PCM 청크를 WebSocket 등으로 백엔드에 보내고, 백엔드가 그 바이트를
아래 request_queue에 넣어주는 구조로 바꾸면 된다 (맨 아래 "실전 적용 메모" 참고).
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import grpc
import sounddevice as sd

import nest_pb2
import nest_pb2_grpc

# ---------------------------------------------------------------------------
# .env 로드 — CLOVA_SECRET_KEY를 ai 레포의 .env에서 그대로 읽어온다.
# 이 스크립트가 ai 레포와 다른 폴더에 있어도 되도록, 있을 법한 경로 몇 개를
# 순서대로 찾아본다. 본인 환경에 맞는 경로가 목록에 없으면 추가할 것.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    _ENV_CANDIDATES = [
        Path(__file__).resolve().parent / ".env",                                  # 이 스크립트와 같은 폴더
        Path.home() / "Desktop" / "KTB" / "python-weekly" / "pill_ai" / "ai" / ".env",  # ai 레포 경로
    ]
    for _env_path in _ENV_CANDIDATES:
        if _env_path.exists():
            load_dotenv(_env_path)
            print(f".env 로드됨: {_env_path}")
            break
    else:
        print(f".env를 못 찾았습니다 (확인한 경로: {[str(p) for p in _ENV_CANDIDATES]}). "
              "이미 export 해두셨다면 무시해도 됩니다.")
except ImportError:
    print("python-dotenv 미설치 — `pip install python-dotenv` 하거나, "
          "터미널에서 `export CLOVA_SECRET_KEY=...`로 직접 지정해주세요.")

# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
CLOVA_HOST = "clovaspeech-gw.ncloud.com:50051"
SECRET_KEY = os.environ.get("CLOVA_SECRET_KEY", "")
if not SECRET_KEY:
    raise RuntimeError(
        "CLOVA_SECRET_KEY를 찾지 못했습니다. .env 파일 경로가 맞는지, "
        "그 안에 CLOVA_SECRET_KEY=... 줄이 있는지 확인해주세요."
    )

SAMPLE_RATE = 16000          # CLOVA 요구 스펙: 16kHz
CHANNELS = 1                 # mono
CHUNK_DURATION_SEC = 0.2     # 200ms 단위로 전송 (너무 잘게 보내면 오버헤드만 커짐)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)


def build_config_json() -> str:
    """스트림 최초 1회 전송하는 Config JSON.

    keywordBoosting은 도메인 특화 단어(약 이름, 증상 등) 인식률을 높이는 용도다.
    CLOVA 콘솔에서 '도메인'을 만들어 boostingID로 관리하는 방법도 있지만(운영 권장),
    여기서는 코드에서 바로 words를 등록하는 방식을 예시로 사용한다.
    """
    config = {
        "transcription": {
            "language": "ko",
        },
        "keywordBoosting": {
            "useDomainBoostings": True,
            "boostingIDs": ["증상"],
        },
        "semanticEpd": {
            "skipEmptyText": True,
            "useWordEpd": False,
            "usePeriodEpd": True,
            "gapThreshold": 500,      # 0.5초 이상 묵음이면 한 문장으로 끊어서 결과 생성
            "durationThreshold": 20000,
            "syllableThreshold": 0,
        },
    }
    return json.dumps(config, ensure_ascii=False)


@dataclass
class _QueueItem:
    chunk: bytes
    ep_flag: bool


class ClovaStreamingClient:
    """마이크 입력을 CLOVA Speech 실시간 스트리밍 인식 API로 보내고,
    인식 결과를 콜백으로 전달하는 클라이언트."""

    def __init__(self, secret_key: str, on_transcript=None):
        self._secret_key = secret_key
        self._queue: "queue.Queue[Optional[_QueueItem]]" = queue.Queue()
        self._seq_id = 0
        self._stopped = threading.Event()
        # on_transcript(text: str, is_sentence_end: bool) 형태의 콜백
        self._on_transcript = on_transcript or (lambda text, ep: print(f"[{'FINAL' if ep else '...'}] {text}"))

    # -- 1) 요청 스트림 생성기 --------------------------------------------------
    def _request_generator(self):
        # 최초 1회: CONFIG
        yield nest_pb2.NestRequest(
            type=nest_pb2.RequestType.CONFIG,
            config=nest_pb2.NestConfig(config=build_config_json()),
        )
        # 이후: 큐에 쌓이는 PCM 조각을 DATA로 계속 전송
        while True:
            item = self._queue.get()
            if item is None:  # 종료 시그널
                return
            self._seq_id += 1
            extra_contents = json.dumps({"seqId": self._seq_id, "epFlag": item.ep_flag})
            yield nest_pb2.NestRequest(
                type=nest_pb2.RequestType.DATA,
                data=nest_pb2.NestData(chunk=item.chunk, extra_contents=extra_contents),
            )

    # -- 2) 마이크 콜백: 들어오는 PCM을 큐에 적재 --------------------------------
    def _mic_callback(self, indata, frames, time_info, status):
        if status:
            print("mic status:", status)
        if not self._stopped.is_set():
            self._queue.put(_QueueItem(chunk=bytes(indata), ep_flag=False))

    # -- 3) 응답 처리 -----------------------------------------------------------
    def _handle_response(self, response: "nest_pb2.NestResponse") -> None:
        payload = json.loads(response.contents)
        response_types = payload.get("responseType", [])

        if "transcription" in response_types:
            t = payload["transcription"]
            text = t.get("text", "")
            # epFlag=true인 조각이 "이 세그먼트의 마지막 결과"라는 뜻은 아니고,
            # 클라이언트가 epFlag=true로 보낸 요청에 대한 응답인지를 나타낸다.
            # 문장 단위 확정 여부는 durationThreshold/gapThreshold 등 EPD 기준으로
            # 서버가 알아서 끊어 보내주므로, 여기서는 들어오는 text를 그대로 흘려주면 된다.
            self._on_transcript(text, t.get("epFlag", False))
        elif "recognize" in response_types:
            status = payload.get("recognize", {}).get("status")
            if status and status != "Success":
                print("recognize 오류 응답:", payload)
        # config / keywordBoosting / Forbidden / semanticEpd 응답은 최초 설정 단계에서만 온다.
        else:
            print("설정 응답:", payload)

    # -- 4) 실행 -----------------------------------------------------------------
    def run_with_microphone(self, seconds: Optional[float] = None):
        """마이크를 열고 인식을 시작한다. seconds를 지정하면 그 시간 후 자동 종료,
        지정하지 않으면 Ctrl+C로 종료할 때까지 계속 듣는다."""
        channel = grpc.secure_channel(CLOVA_HOST, grpc.ssl_channel_credentials())
        stub = nest_pb2_grpc.NestServiceStub(channel)
        metadata = (("authorization", f"Bearer {self._secret_key}"),)

        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=self._mic_callback,
        )

        responses = None
        with stream:
            responses = stub.recognize(self._request_generator(), metadata=metadata)

            def _reader():
                try:
                    for response in responses:
                        self._handle_response(response)
                except grpc.RpcError as e:
                    if not self._stopped.is_set():
                        print("gRPC 오류:", e)

            reader_thread = threading.Thread(target=_reader, daemon=True)
            reader_thread.start()

            try:
                if seconds is not None:
                    self._stopped.wait(timeout=seconds)
                else:
                    self._stopped.wait()  # Ctrl+C 등 외부에서 stop() 호출 전까지 대기
            except KeyboardInterrupt:
                pass
            finally:
                self.stop()
                reader_thread.join(timeout=5)

        channel.close()

    def stop(self):
        if self._stopped.is_set():
            return
        self._stopped.set()
        # 마지막 조각을 epFlag=true로 보내 즉시 결과를 flush 시키고, 스트림을 닫는다.
        self._queue.put(_QueueItem(chunk=b"", ep_flag=True))
        self._queue.put(None)


if __name__ == "__main__":
    def on_transcript(text: str, ep_flag: bool):
        print(f"[인식 결과] {text}")

    client = ClovaStreamingClient(SECRET_KEY, on_transcript=on_transcript)
    print("녹음을 시작합니다. 10초간 말씀해 주세요 (Ctrl+C로 즉시 종료 가능)...")
    client.run_with_microphone(seconds=10)
    print("종료.")
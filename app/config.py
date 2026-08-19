"""
환경변수/​.env 파일에서 설정값을 읽어온다.

인증키를 코드에 하드코딩하지 않고 여기로 분리하는 이유는 두 가지다.
1) 이 키가 그대로 git에 커밋되면 깃허브에 공개 저장소로 올릴 때 그대로 노출된다.
2) 로컬/서버(EC2)에서 서로 다른 키·설정을 코드 수정 없이 바꿔 끼울 수 있어야 한다.

.env 파일은 반드시 .gitignore 에 추가해서 커밋되지 않게 할 것.
"""
from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # data.go.kr에서 발급받은 인증키. "일반 인증키(Encoding)"를 그대로 붙여넣어도 되고
    # "인증키(Decoding)"를 붙여넣어도 된다 — PillIdentificationApiClient가 알아서 정규화한다.
    mfds_pill_service_key: str

    # Google AI Studio(https://aistudio.google.com/apikey)에서 발급받은 Gemini API 키.
    # 비워두면 Step 2(Vision) 관련 기능만 동작하지 않고, 나머지(Step 3/4)는 정상 동작한다.
    gemini_api_key: str = ""


settings = Settings()

# google-genai SDK(app/vision/gemini_client.py)는 GEMINI_API_KEY를 프로세스 환경변수에서
# 직접 읽는다 — pydantic-settings가 .env를 읽어도 os.environ에는 자동으로 반영되지 않으므로,
# 여기서 한 번 다리를 놓아준다. 이미 쉘에서 export 해둔 값이 있으면 그걸 우선한다
# (setdefault는 이미 키가 있으면 덮어쓰지 않음).
if settings.gemini_api_key:
    os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)

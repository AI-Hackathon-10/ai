# ai

알약 사진 → Vision 추출 → 낱알식별 매칭 FastAPI 서비스.

## 테스트

```bash
python3 -m pytest -q
```

무엇을 검증했는지, `/identify/db` 실DB 확인 결과는 [docs/테스트.md](docs/테스트.md)에 정리했다.

## LangSmith 트레이싱

그래프 실행(`vision_extraction_graph`)과 재시도별 Gemini 호출(`gemini_vision_call`)이
각각 트레이스로 남는다. 이미지 바이트는 크기 정보로만 남고 원본은 트레이스에 올라가지 않는다.

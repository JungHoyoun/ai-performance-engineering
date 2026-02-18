# 챕터 18 - 고급 어텐션 및 디코딩

## 요약
현대적인 디코더 기법인 FlexAttention, FlexDecoding, 추측 및 페이지드 어텐션 워크플로우를 PyTorch와 CUDA/Triton 모두에서 구현하여 실제 하드웨어에서 커널을 검증하면서 빠르게 반복할 수 있습니다.

## 학습 목표
- 커스텀 마스크, 스코어 모드, KV 캐시 통합으로 FlexAttention/FlexDecoding 워크로드를 프로토타입합니다.
- 더 낮은 지연 시간을 위해 추가 계산을 거래하는 추측 디코딩 파이프라인을 평가합니다.
- Blackwell tmem 제한에 맞게 조정된 텐서 코어 최적화 어텐션 커널을 테스트합니다.
- 제공된 러너를 사용하여 서빙 프레임워크(vLLM)와의 통합 지점을 검증합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_flexdecoding.py`, `optimized_flexdecoding.py`, `optimized_flexdecoding_graphs.py`, `v1_engine_loop.py`, `v1_engine_loop_common.py` | FlexDecoding 벤치마크 및 V1 폴링 루프 정확성 도구(벤치마크 쌍 아님). |
| `baseline_tensor_cores.py`, `optimized_tensor_cores.py`, `flashmla_kernel.cu`, `warp_specialized_triton.py` | 빠른 검증을 위한 텐서 코어 어텐션 커널 및 Triton 동등물. |
| `flex_attention_native.py`, `flex_attention_enhanced.py`, `flex_attention_large_model.py`, `kv_cache_integration_example.py` | 소형 크기에서 KV 캐시 재사용이 있는 대형 모델까지의 FlexAttention 예제. |
| `baseline_vllm_v1_integration.py`, `optimized_vllm_v1_integration.py`, `baseline_vllm_decode_graphs.py`, `optimized_vllm_decode_graphs.py`, `configs/`, `spec_configs/`, `workload_config.py` | vLLM 또는 커스텀 하네스를 통해 워크로드를 밀어넣기 위한 서빙 통합 및 구성 프리셋. |
| `speculative_decode/spec_config_sweep.py` | 추측 디코딩 구성을 스윕하고 지연 시간/처리량 트레이드오프를 요약하는 도구. |
| `compare.py`, `expectations_{hardware_key}.json`, `test_flex_attention.py` | 하네스 진입점, 회귀 임계치, FlexAttention API를 위한 pytest 커버리지. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch18/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch18
python -m cli.aisp bench run --targets ch18 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python optimized_flexdecoding.py --profiling`이 디코딩된 토큰을 일치시키면서 베이스라인보다 훨씬 적은 커널과 낮은 지연 시간을 보고합니다.
- `python run_vllm_decoder.py --spec-config spec_configs/draft_and_verify.json`이 네이티브 FlexAttention 경로와 정확도 동등성으로 완료됩니다.
- `python test_flex_attention.py`가 로컬에서 통과하여 마스크/스코어-모드 헬퍼가 올바르게 연결되었음을 확인합니다.

## 참고 사항
- `flex_attention` 스크립트는 코드를 편집하지 않고도 형태를 스윕할 수 있도록 `BLOCK_SIZE`, `DOC_SPAN`, `SEQ_LEN`과 같은 환경 변수를 허용합니다.
- `flashmla_kernel.cu`는 SM121 하드웨어에서 컴파일 상태를 유지하기 위해 Blackwell 특화 텐서 메모리 가드를 포함합니다.

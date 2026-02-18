# 챕터 19 - 저정밀 학습 및 메모리 시스템

## 요약
NVFP4/FP8 워크플로우, KV 캐시 양자화, 메모리 이중 버퍼링, 적응형 할당자를 탐구하여 저정밀 실험이 수치적으로 안전하게 유지되면서 HBM의 모든 바이트를 짜냅니다.

## 학습 목표
- 캘리브레이션 및 검증 훅이 있는 FP4/FP6/FP8 학습 루프를 벤치마크합니다.
- 정밀도 제약을 존중하면서 KV 캐시 프리페치와 계산을 오버랩합니다.
- 드리프트 없이 실행 중에 형식을 전환하는 동적 양자화 캐시를 구현합니다.
- 단편화된 메모리 풀을 모니터링하고 재균형하는 할당자 헬퍼를 설계합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_nvfp4_training.py`, `optimized_nvfp4_training.py`, `native_fp4_quantization.py`, `native_fp6_quantization.py`, `native_fp8_training.py` | 자동 캘리브레이션으로 FP8과 NVFP4 사이를 전환하는 학습 및 양자화 레시피. |
| `baseline_memory_double_buffering.py`, `optimized_memory_double_buffering.py`, `memory_allocator_with_monitoring.py`, `dynamic_memory_allocator.py`, `_allocator_worker.py` | 이중 버퍼링, 계측, 적응형 워커 풀을 포함하는 메모리 관리 헬퍼. |
| `baseline_kv_prefetch_overlap.cu`, `optimized_kv_prefetch_overlap.cu`, `kv_prefetch_overlap_sm121` 바이너리 | cp.async 파이프라인을 사용할 때 양자화된 KV 프리페치가 계산과 오버랩될 수 있음을 증명하는 CUDA 커널. |
| `baseline_dynamic_quantized_cache.py`, `optimized_dynamic_quantized_cache.py`, `dynamic_quantized_cache.py`, `token_precision_switching.py`, `dynamic_precision_switching.py` | 정확도 예산에 따라 정밀도를 동적으로 전환하는 양자화 캐시 관리. |
| `baseline_fp4_hardware_kernel.cu`, `optimized_fp4_hardware_kernel.cu`, `fp8_hardware_kernel.cu`, `custom_allocator_retry.py`, `adaptive_parallelism_strategy.py`, `adaptive_parallelism_worker_pool.py` | 이기종 정밀도 플릿을 위한 하드웨어 수준 커널 및 적응형 스케줄링 헬퍼. |
| `compare.py`, `arch_config.py`, `expectations_{hardware_key}.json` | 하네스 진입점, 아키텍처 토글, 저장된 기대값 데이터. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch19/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch19
python -m cli.aisp bench run --targets ch19 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python optimized_nvfp4_training.py --calibrate`가 FP8로 워밍업한 후 NVFP4로 전환하여 베이스라인의 정확도 임계치와 일치합니다.
- `python optimized_dynamic_quantized_cache.py --trace`가 제한된 오류로 정밀도 전환을 기록하여 토큰 수준 전환의 정확성을 확인합니다.
- `nvcc -o optimized_kv_prefetch_overlap_sm121 optimized_kv_prefetch_overlap.cu`와 베이스라인 바이너리가 Nsight Compute에서 측정 가능한 오버랩 개선을 보여줍니다.

## 참고 사항
- `arch_config.py`는 디바이스별 `ENABLE_NVFP4`/`ENABLE_TF32` 토글을 노출하여 정밀도 레시피를 쉽게 비교할 수 있습니다.
- `validate_quantization_performance.py`는 이익 증명 보고를 위해 정확도 vs 처리량 수치를 CSV 형식으로 집계합니다.

# 챕터 11 - 스트림 및 동시성

## 요약
CUDA 스트림, 순서화된 시퀀스, Hyper-Q, 워프 특화 파이프라인, 적응형 스케줄링을 사용하여 Blackwell에서 계산, 메모리, 통신을 오버랩하는 방법을 설명합니다.

## 학습 목표
- 여러 CUDA 스트림을 사용하여 우선순위 작업을 굶주리게 하지 않고 독립적인 커널을 오버랩합니다.
- KV 캐시 업데이트 및 스트림 순서 메모리 풀에 대한 순서화 제약을 제어합니다.
- DSMEM을 통해 데이터를 공유하는 워프 특화 멀티스트림 커널을 벤치마크합니다.
- 런타임 텔레메트리를 기반으로 스트림 사용을 조정하는 적응형 정책을 도입합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_streams.py`, `optimized_streams.py`, `streams_overlap_demo.cu`, `streams_ordered_demo.cu`, `streams_warp_specialized_demo.cu`, `stream_overlap_base.py` | 직렬화된 실행과 오버랩된 워크로드를 비교하는 핵심 스트림 오버랩 데모. |
| `baseline_stream_ordered.py`, `baseline_stream_ordered_kv_cache.py`, `optimized_stream_ordered.py`, `optimized_stream_ordered_kv_cache.py` | 오버랩을 활성화하면서 결정적인 업데이트를 보장하는 스트림 순서 할당자 및 KV 캐시 예제. |
| `baseline_gemm_streams.py`, `optimized_gemm_streams.py`, `baseline_tensor_cores_streams.py`, `optimized_tensor_cores_streams.py` | 수학 vs I/O 단계를 분리하기 위해 여러 스트림에 걸쳐 텐서 코어 커널을 스케줄링하는 GEMM 파이프라인. |
| `baseline_distributed_streams.py`, `optimized_distributed_streams.py`, `baseline_adaptive_streams.py`, `optimized_adaptive_streams.py` | 대규모 시스템에서 NCCL, 계산, I/O 작업을 균형 있게 조정하는 적응형 스트리밍 컨트롤러. |
| `baseline_warp_specialization_multistream.*`, `optimized_warp_specialized_multistream.*`, `warp_specialized_cluster_pipeline_multistream.cu` | DSMEM 사용 및 스트림별 특화를 시연하는 워프 특화 멀티스트림 커널. |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json` | 동시성 회귀를 위한 하네스 드라이버 및 기대값 데이터. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch11/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch11
python -m cli.aisp bench run --targets ch11 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python optimized_streams.py --trace`가 Nsight Systems에서 오버랩되는 NVTX 범위를 캡처하여 동시성이 활성화되었음을 증명합니다.
- `python optimized_stream_ordered_kv_cache.py --validate`가 캐시 업데이트 간 유휴 갭을 줄이면서 베이스라인 출력과 일치합니다.
- 워프 특화 멀티스트림 커널이 지원되지 않는 하드웨어(DSMEM 없음)를 즉시 표시하여 자동 폴백을 방지합니다.

## 참고 사항
- `warp_specialized_triton.py`는 컴파일러 생성 스케줄을 비교할 수 있도록 CUDA 동시성 데모에 대한 Triton 유사물을 제공합니다.
- `kv_prefetch_pipeline_enhanced_demo.cu`는 이 디렉토리에 번들된 DSMEM 커널을 기반으로 하여 전체 파이프라인을 로컬에서 연구할 수 있습니다.

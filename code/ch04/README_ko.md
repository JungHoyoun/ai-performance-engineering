# 챕터 4 - 다중 GPU 분산

## 요약
NVLink/NVSwitch 패브릭 인식, NCCL 튜닝, NVSHMEM 집합 연산, 대칭 메모리 패턴을 사용하여 여러 Blackwell GPU에 걸쳐 학습 및 추론을 확장하는 방법을 시연합니다.

## 학습 목표
- 오버랩 유무에 따른 데이터 병렬 및 텐서 병렬 학습 루프를 벤치마크합니다.
- 로컬 및 분리된 GPU를 혼합할 때 NVLink 대역폭과 토폴로지 효과를 정량화합니다.
- GPU 동기화에서 호스트 개입을 줄이기 위해 NVSHMEM 파이프라인을 실험합니다.
- KV 캐시 복제 및 옵티마이저 상태 샤딩을 단순화하기 위해 대칭 메모리 풀을 채택합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_dataparallel.py`, `optimized_dataparallel.py` | 단일 GPU DataParallel 안티 패턴 vs 직접 GPU 실행. |
| `baseline_dataparallel_multigpu.py`, `optimized_dataparallel_multigpu.py` | 다중 GPU DataParallel vs 사전 스테이징된 샤드를 사용한 수동 그래디언트 감소. |
| `baseline_no_overlap.py`, `optimized_no_overlap.py` | allreduce 지연 시간을 숨기기 위해 계산/통신 동시성을 스테이징하고 마이크로배치를 파이프라인 처리하는 오버랩 연구. |
| `baseline_nvlink.py`, `optimized_nvlink.py` 외 여러 NVLink 관련 파일 | 피어 대역폭과 토폴로지 효과(단일 및 다중 GPU)를 검증하는 NVLink 실습. |
| `baseline_continuous_batching.py`, `optimized_continuous_batching.py` 외 여러 배칭/분리 파일 | 풀링 및 원격 KV 재사용을 보여주는 연속 배칭 + 분리 추론 데모. |
| `baseline_gradient_compression_fp16.py`, `optimized_gradient_compression_fp16.py` 외 여러 그래디언트 압축 파일 | 소형 버킷 vs 전체 버퍼 압축을 비교하는 그래디언트 압축 all-reduce 벤치마크(단일 GPU 및 다중 GPU FP16/INT8 경로). |
| `baseline_pipeline_parallel.py`, `optimized_pipeline_parallel_1f1b.py` 외 여러 파이프라인/텐서 병렬 파일 | 파이프라인/텐서 병렬 및 torchcomms 오버랩 연구(단일 및 다중 GPU). |
| `baseline_nvshmem_pipeline_parallel_multigpu.py`, `optimized_nvshmem_pipeline_parallel_multigpu.py` 외 | 디바이스 주도 동기화의 이점을 강조하는 NVSHMEM 파이프라인 및 학습 샘플. |
| `baseline_symmetric_memory_perf.py`, `optimized_symmetric_memory_perf.py` 외 | KV 캐시 및 옵티마이저 샤드를 위한 대칭 메모리 유틸리티 및 성능 프로브. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `bandwidth_benchmark_suite_multigpu.py`, `nccl_benchmark.py` | 하네스 드라이버 및 토폴로지 bring-up을 위한 독립형 NCCL/NVLink 스위퍼. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch04/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch04
python -m cli.aisp bench run --targets ch04 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python compare.py --examples dataparallel_multigpu`가 최적화된 쌍이 더 낮은 지연 시간으로 계산과 통신을 오버랩하는 것을 보여줍니다.
- `python bandwidth_benchmark_suite_multigpu.py --profile minimal`이 연결된 GPU 쌍에서 >=250 GB/s 링크를 표면화하고 느린 홉을 강조합니다.
- NVSHMEM 샘플이 `NVSHMEM_SYMMETRIC_SIZE`가 워크로드를 수용할 크기로 설정되면 일관된 출력을 생성합니다. 잘못된 구성은 명확한 오류를 발생시킵니다.

## 참고 사항
- `symmetric_memory_*` 헬퍼는 NVSwitch 패널티 없이 GPU 간 KV 캐시 라인을 풀링하기 위한 사용자 공간 할당자를 보유합니다.
- 다중 노드 테스트를 시작하기 전에 `nccl_blackwell_config.py`를 사용하여 NCCL 환경 변수(최소 NRings, IB 매핑)를 초기화하세요.

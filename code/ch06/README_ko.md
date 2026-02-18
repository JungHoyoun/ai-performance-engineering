# 챕터 6 - CUDA 프로그래밍 기초

## 요약
Python에서 CUDA C++로 이동합니다. 첫 번째 커널 작성, 점유율 분석, 메모리 레이아웃 제어, Blackwell 디바이스에서 ILP, 실행 경계, 통합 메모리 실험을 다룹니다.

## 학습 목표
- 하네스 워크로드를 미러링하는 커스텀 커널을 작성하고 실행합니다.
- 점유율, 실행 경계, 레지스터 압력이 상호 작용하는 방식을 이해합니다.
- ILP와 벡터화된 메모리 연산을 사용하여 스레드당 처리량을 높입니다.
- Blackwell GPU에서 통합 메모리 및 할당자 튜닝을 검증합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `my_first_kernel.cu`, `simple_kernel.cu`, `baseline_add_cuda.cu`, `optimized_add_cuda_parallel.cu` 외 | CUDA 빌드 체인 및 실행 파라미터를 검증하는 Hello-world 커널 및 Python 래퍼. |
| `baseline_add_tensors_cuda.cu`, `optimized_add_tensors_cuda.cu` 외 | 자동 핀 메모리 스테이징 및 정확성 검사가 있는 텐서 지향 덧셈. |
| `baseline_attention_ilp.py`, `baseline_gemm_ilp.py`, `optimized_gemm_ilp.py`, `ilp_low_occupancy_vec4_demo.cu` 외 | 루프 언롤링, 레지스터, 벡터 너비를 조작하는 명령어 수준 병렬성(ILP) 연구. |
| `baseline_bank_conflicts.cu`, `optimized_bank_conflicts.cu`, `baseline_launch_bounds*.{py,cu}`, `optimized_launch_bounds*.{py,cu}` | 공유 메모리 레이아웃과 CTA 크기를 강조하는 뱅크 충돌 및 실행 경계 실습. |
| `baseline_autotuning.py`, `optimized_autotuning.py`, `memory_pool_tuning.cu`, `stream_ordered_allocator/` | 단편화 및 스트림 순서를 제어하는 자동 튜닝 하네스 및 할당자 실험. |
| `unified_memory.cu`, `occupancy_api.cu`, `baseline_quantization_ilp.py`, `optimized_quantization_ilp.py` | 통합 메모리 데모, 점유율 계산기 샘플, 양자화 중심 ILP 워크로드. |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json`, `workload_config.py` | 하네스 진입점, 빌드 스크립트, 기대값 기준선, 워크로드 설정. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch06/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch06
python -m cli.aisp bench run --targets ch06 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `nvcc -o baseline_add_cuda_sm121 baseline_add_cuda.cu` vs 최적화된 벡터화 버전이 Nsight Compute로 검사했을 때 명확한 대역폭 차이를 보입니다.
- `python optimized_autotuning.py --search`가 큐레이션된 프리셋과 동일한 스케줄로 수렴하고 `artifacts/` 아래에 점수 테이블을 기록합니다.
- `python compare.py --examples ilp`가 최적화된 ILP 커널이 동일한 출력으로 더 높은 명령어당 바이트를 달성함을 확인합니다.

## 참고 사항
- `arch_config.py`는 SM별 컴파일 플래그(예: 지원되지 않는 GPU에서 파이프라인 비활성화)를 강제하여 구형 하드웨어에서 우아하게 실패합니다.
- `cuda_extensions/`의 CUDA 확장은 인터랙티브 프로토타이핑을 위해 노트북으로 직접 가져올 수 있습니다.

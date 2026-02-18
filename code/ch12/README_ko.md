# 챕터 12 - CUDA 그래프 및 동적 워크로드

## 요약
현대적인 CUDA 그래프 기능인 조건부 캡처, 그래프 메모리 튜닝, 동적 병렬성, 작업 큐를 다루어 실행당 오버헤드 없이 불규칙적인 워크로드를 성능 있게 유지합니다.

## 학습 목표
- 안정적인 워크로드를 CUDA 그래프로 캡처하고 이거(eager) 실행과의 차이를 연구합니다.
- 적응형 파이프라인을 위해 조건부 노드와 그래프 메모리 풀을 사용합니다.
- CPU 개입을 줄이기 위해 디바이스 측 실행(동적 병렬성)을 실험합니다.
- GPU 상주 작업 큐 및 불균등 파티션 스케줄러를 구현합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_cuda_graphs.py`, `optimized_cuda_graphs.py`, `baseline_cuda_graphs_conditional*.cu`, `optimized_cuda_graphs_conditional*.cu` | 단순 재생에서 조건부 및 DSM 인식 실행으로 발전하는 그래프 캡처 데모. |
| `baseline_graph_bandwidth.{py,cu}`, `optimized_graph_bandwidth.{py,cu}`, `baseline_kernel_launches.py`, `optimized_kernel_launches.py` | 그래프가 CPU 오버헤드를 줄이는 방법을 보여주는 실행 및 대역폭 중심 연구. |
| `baseline_dynamic_parallelism_host.cu`, `baseline_dynamic_parallelism_device.cu`, `optimized_dynamic_parallelism_host.cu`, `optimized_dynamic_parallelism_device.cu` 외 | 동적 병렬성이 도움이 되거나 해가 되는 시기를 보여주는 디바이스 측 실행 샘플. |
| `baseline_work_queue.{py,cu}`, `optimized_work_queue.{py,cu}`, `work_queue_common.cuh` | NVTX 계측을 포함한 불규칙적인 배치 크기를 위한 GPU 작업 큐. |
| `baseline_uneven_partition.cu`, `optimized_uneven_partition.cu`, `baseline_uneven_static.cu`, `optimized_uneven_static.cu` | 런타임에 CTA 할당을 재균형하는 불균등 워크로드 파티셔너. |
| `baseline_kernel_fusion.py`, `optimized_kernel_fusion.py`, `kernel_fusion_cuda_demo.cu` | CPU 동기화를 완전히 제거할 수 있도록 그래프 캡처 내의 커널 퓨전 실습. |
| `compare.py`, `cuda_extensions/`, `expectations_{hardware_key}.json` | 하네스 진입점, 확장 스텁, 기대값 임계치. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch12/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch12
python -m cli.aisp bench run --targets ch12 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python optimized_cuda_graphs.py --iterations 100`이 출력을 일치시키면서 베이스라인보다 낮은 벽 시계 시간을 보고해야 합니다.
- 디바이스 측 동적 병렬성 샘플이 지원되지 않는 하드웨어에서 경고를 내보내어 해당 기능이 있는 GPU의 데이터만 신뢰합니다.
- `python optimized_work_queue.py --trace`가 베이스라인의 뒤처짐과 비교했을 때 CTA 간 균형 잡힌 dequeue 시간을 노출합니다.

## 참고 사항
- `cuda_graphs_workload.cuh`는 자체 커널을 래핑할 때 사용할 수 있는 재사용 가능한 그래프 캡처 헬퍼를 보유합니다.
- `helper_*.cu` 파일은 동적 병렬성 케이스 스터디를 위한 호스트/디바이스 접착 코드를 포함합니다. 새 실험을 부트스트래핑할 때 복사하세요.

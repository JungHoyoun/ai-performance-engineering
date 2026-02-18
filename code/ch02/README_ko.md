# 챕터 2 - GPU 하드웨어 아키텍처

## 요약
Blackwell 시대 시스템을 위한 아키텍처 인식 도구를 제공합니다. SM 및 메모리 사양 쿼리, NVLink 처리량 검증, CPU-GPU 일관성 실험 등을 통해 최적화가 측정된 하드웨어 한계에 기반하게 합니다.

## 학습 목표
- 성능 연구 전에 GPU, CPU, 패브릭 기능을 쿼리하고 기록합니다.
- 전용 마이크로벤치마크를 사용하여 NVLink, PCIe, 메모리 대역폭 상한을 측정합니다.
- 제로 카피 버퍼가 도움이 되는지 해가 되는지 알기 위해 Grace-Blackwell 일관성 경로를 검증합니다.
- 아키텍처별 튜닝 레버를 강조하기 위해 베이스라인 vs 최적화된 cuBLAS 호출을 비교합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `hardware_info.py`, `cpu_gpu_topology_aware.py` | GPU 기능, NUMA 레이아웃, NVLink/NVSwitch 연결, 친화성 힌트를 기록하는 시스템 스캐너. |
| `nvlink_c2c_bandwidth_benchmark.py`, `baseline_memory_transfer.py`, `optimized_memory_transfer.py`, `memory_transfer_pcie_demo.cu`, `memory_transfer_nvlink_demo.cu`, `memory_transfer_zero_copy_demo.cu`, `baseline_memory_transfer_multigpu.cu`, `optimized_memory_transfer_multigpu.cu` | NVLink, PCIe, 일관성 메모리 성능을 정량화하는 피어-투-피어 및 제로 카피 실험. |
| `cpu_gpu_grace_blackwell_coherency.cu`, `cpu_gpu_grace_blackwell_coherency_sm121` | 명시적 전송 vs 공유 매핑을 비교하는 Grace-Blackwell 캐시 일관성 샘플. |
| `baseline_cublas.py`, `optimized_cublas.py` | TF32, 텐서 연산 수학, 스트림 친화성을 토글하여 아키텍처 노브를 강조하는 cuBLAS GEMM 벤치마크 쌍. |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json` | 하네스 드라이버, CUDA 빌드 규칙, 자동화된 합격/불합격 검사를 위한 기대값 파일. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch02/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch02
python -m cli.aisp bench run --targets ch02 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python hardware_info.py`가 시스템의 모든 GPU에 대해 정확한 디바이스 이름, SM 수, HBM 크기를 기록합니다.
- `python nvlink_c2c_bandwidth_benchmark.py --gpus 0 1`이 NVLink 연결 쌍에서 단방향 ~250 GB/s를 유지합니다. 불일치는 토폴로지 또는 드라이버 문제를 나타냅니다.
- 일관성 샘플을 실행하면 제로 카피가 소형 전송(수 MB 미만)에 유리하고, 대형 전송은 명시적 H2D 복사를 선호한다는 것이 문서화된 임계치와 일치함을 보여줍니다.

## 참고 사항
- Grace 전용 일관성 테스트는 GB200/GB300 노드가 필요합니다. PCIe 전용 호스트에서는 바이너리가 no-op입니다.
- `Makefile`은 CUDA 및 CPU 도구를 모두 빌드하므로 챕터를 벗어나지 않고 결과를 비교할 수 있습니다.

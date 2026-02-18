# 챕터 3 - 시스템 튜닝

## 요약
커널 수준 최적화 전에 GPU 워크로드를 지속적으로 공급하는 호스트 수준 변경 사항인 NUMA 피닝, 거버너 조정, 컨테이너 설정, Kubernetes 매니페스트를 다룹니다.

## 학습 목표
- GPU 파이프라인을 스로틀하는 CPU 및 메모리 친화성 문제를 진단합니다.
- 공유 클러스터에서 지속적인 GPU 처리량을 위해 Docker 및 Kubernetes 환경을 강화합니다.
- 실험실 머신이 일관성을 유지하도록 셸 스크립트를 통해 반복 가능한 시스템 튜닝을 자동화합니다.
- 호스트 수준 수정이 GEMM 처리량을 높이고 실행 지연 시간을 줄이는 방법을 정량화합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_numa_unaware.py`, `optimized_numa_unaware.py`, `bind_numa_affinity.py`, `numa_topology_script.sh` | 데이터 로더, NCCL 랭크, GPU 컨텍스트를 올바른 CPU 소켓에 바인딩하기 위한 NUMA 진단 및 피닝 헬퍼. |
| `baseline_docker.py`, `optimized_docker.py`, `docker_gpu_optimized.dockerfile`, `system_tuning.sh`, `gpu_setup_commands.sh` | 영속 모드, 대용량 페이지, IRQ 스티어링, MIG 가시성을 토글하는 컨테이너 구성 및 호스트 설정 스크립트. |
| `baseline_kubernetes.py`, `optimized_kubernetes.py`, `kubernetes_mig_pod.yaml`, `kubernetes_topology_pod.yaml` | 다중 테넌트 플릿을 위한 토폴로지 인식 스케줄링 및 MIG 파티셔닝을 보여주는 Kubernetes 매니페스트. |
| `cpu_gpu_numa_optimizations.sh`, `system_tuning.sh`, `gpu_setup_commands.sh` | CPU 거버너, cgroup 제한, 영속 모드, 드라이버 설정을 벤치마크 하네스와 정렬하는 워크플로우 스크립트. |
| `baseline_gemm.py`, `optimized_gemm.py`, `train.py` | 시스템 튜닝 변경의 영향을 측정 가능한 FLOP/s로 표면화하는 간단한 GEMM + 학습 루프. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | 하네스 진입점, Python 의존성, 회귀 임계치. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch03/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch03
python -m cli.aisp bench run --targets ch03 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `bind_numa_affinity.py` 전후에 `python baseline_numa_unaware.py --diagnostics`를 실행하여 교차 소켓 메모리 트래픽이 거의 제로로 떨어지는지 확인합니다.
- `python optimized_docker.py --image docker_gpu_optimized.dockerfile`이 GPU 클록이 고정된 상태에서 호스트 실행과 동일한 처리량을 유지해야 합니다.
- `python compare.py --examples gemm`이 `system_tuning.sh` 적용 후 optimized_gemm이 측정된 호스트 피크와 일치함을 보여줍니다.

## 참고 사항
- `cpu_gpu_numa_optimizations.sh`는 재부팅 후 안전하게 재실행할 수 있습니다. irqbalance 피닝 및 거버너 설정을 재적용합니다.
- Kubernetes 매니페스트는 외부 저장소를 참조하지 않고 NVLink/NVSwitch 친화성에 필요한 어노테이션을 문서화합니다.

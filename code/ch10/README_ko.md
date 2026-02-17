# 챕터 10 - 텐서 코어 파이프라인 및 클러스터 기능

## 요약
Blackwell에서 텐서 코어 친화적인 스케줄링을 적용합니다. 워프 특화, TMA 기반 파이프라인, 영구 커널, DSMEM 및 NVLink-C2C 인식이 있는 스레드 블록 클러스터를 다룹니다.

## 학습 목표
- 워프 특화와 cp.async/TMA를 사용하여 텐서 코어를 포화 상태로 유지합니다.
- 반복에 걸쳐 실행 오버헤드를 분할 상각하는 영구 행렬 곱셈을 프로토타입합니다.
- DSMEM 유무에 관계없이 스레드 블록 클러스터를 실습하여 하드웨어 한계를 이해합니다.
- PyTorch, Triton, CUDA 커널을 결합하면서 기대값을 동기화합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_attention.py`, `optimized_attention.py`, `baseline_flash_attention.py`, `optimized_flash_attention.py`, `analyze_scaling.py` | 현대 디코더 모델을 위한 이거(eager), 퓨전, `torch.compile` 경로에 걸친 어텐션 워크로드. |
| `baseline_batch.py`, `optimized_batch.py`, `baseline_matmul.py`, `optimized_matmul.py`, `baseline_matmul_tcgen05.py`, `optimized_matmul_tcgen05.py` | tcgen05 낮추기, 레지스터 타일링, PyTorch 통합을 시연하는 텐서 코어 행렬 곱셈 변형. |
| `baseline_tcgen05_warp_specialization.py`, `optimized_tcgen05_warp_specialization.py`, `tcgen05_warp_specialized.cu` | 전용 생산자/소비자 워프가 있는 워프 특화 tcgen05 GEMM. |
| `baseline_tcgen05_warp_specialization_cutlass.py`, `optimized_tcgen05_warp_specialization_cutlass.py` 외 | CUTLASS 워프 특화 메인루프 비교(1-SM 워프 특화 vs 2-SM 워프그룹 타일). |
| `warpgroup_specialization_demo.py`, `tcgen05_warpgroup_specialized.cu` | 2-SM 타일을 사용하는 CUTLASS 워프그룹 배열 메인루프 데모. |
| `baseline_double_buffered_pipeline.{py,cu}`, `optimized_double_buffered_pipeline.{py,cu}` 외 | cp.async, TMA, 수동 이중 버퍼링을 혼합하는 비동기 파이프라인 샘플. |
| `baseline_cluster_group*.{py,cu}`, `optimized_cluster_group*.{py,cu}` 외 | DSMEM 활성화 및 DSMEM 없는 스레드 블록 클러스터를 다루는 클러스터 커널 제품군. |
| `baseline_cluster_multicast.py`, `optimized_cluster_multicast.py` 외 | CUDA 바이너리 하네스 벤치마크로 래핑된 클러스터 멀티캐스트 GEMM 예제. |
| `baseline_cooperative_persistent.{py,cu}`, `optimized_cooperative_persistent.{py,cu}` 외 | 안정적인 처리량을 위해 협력 그룹과 TMA 스트림을 결합하는 영구 커널. |
| `baseline_flash_attn_tma_micro_pipeline.{py,cu}`, `optimized_flash_attn_tma_micro_pipeline.{py,cu}` 외 | Triton, CUDA, 인라인 PTX를 혼합하는 마이크로 파이프라인 및 워프 특화 연구. |
| `compare.py`, `workload_config.py`, `demo_both_examples.sh`, `profile.sh`, `requirements_cufile.txt` | 하네스 진입점, 워크로드 다이얼, 데모 러너, Nsight 자동화, 선택적 cuFile 의존성. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch10/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch10
python -m cli.aisp bench run --targets ch10 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- 클러스터 활성화 커널은 DSMEM 지원이 없는 하드웨어에서 빠르게 실패하고, DSMEM 없는 변형은 여전히 실행됩니다. 이를 사용하여 클러스터 기능 플래그를 확인하세요.
- `python optimized_flash_attn_tma_micro_pipeline.py --profile`이 베이스라인 스크립트보다 더 적은 커널 실행과 더 높은 달성 FLOP/s를 생성합니다.
- `bash demo_both_examples.sh`가 CUDA 메모리 파이프라인 및 GDS 데모를 실행하여 실행 분할 상각 및 I/O 오버랩을 강조합니다.

## 참고 사항
- `cufile_gds_example.py`는 I/O 집약적인 학습 루프를 위한 텐서 코어 파이프라인에 GPUDirect 스토리지를 통합하는 방법을 시연합니다.
- `requirements_cufile.txt`는 선택적 `cufile` 휠을 보유합니다. GPUDirect 스토리지가 활성화된 호스트에만 설치하세요.
- CUTLASS 스타일 워프 특화 쌍은 성능 비교를 위해 `sm100_mma_array_warpspecialized`와 정렬된 참조 구현을 제공합니다.

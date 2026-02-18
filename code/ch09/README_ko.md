# 챕터 9 - 산술 강도 및 커널 퓨전

## 요약
루프라인을 따라 워크로드를 이동하는 방법을 탐구합니다. 타일링으로 산술 강도를 높이고, 메모리 바운드 커널을 퓨전하며, Blackwell 텐서 코어를 위해 구축된 CUTLASS/Triton/인라인 PTX 경로를 배포합니다.

## 학습 목표
- 계산 바운드 vs 메모리 바운드 동작을 분리하고 커널을 그에 맞게 조정합니다.
- 레지스터 압력과 데이터 재사용을 균형 있게 유지하는 마이크로 타일링 스케줄을 설계합니다.
- 커스텀 CUDA 폴백을 유지하면서 빠른 반복을 위해 CUTLASS와 Triton을 활용합니다.
- 중복 메모리 접근을 제거하기 위해 리덕션 집약적인 커널(예: norm + activation)을 퓨전합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_compute_bound.py`, `optimized_compute_bound.py`, `baseline_memory_bound.py`, `optimized_memory_bound.py` | 계산 vs 대역폭 상한을 분리하고 튜닝 전략을 시연하는 참조 커널. |
| `baseline_micro_tiling_matmul.cu`, `optimized_micro_tiling_matmul.cu` 외 | 명시적 레지스터 블로킹 및 cp.async 프리페치가 있는 마이크로 타일링 행렬 곱셈. |
| `baseline_cutlass_gemm.cu`, `optimized_cutlass_gemm.cu` 외 | 수동 튜닝 커널과 벤더 라이브러리를 비교하기 위한 라이브러리 GEMM 기준선. |
| `baseline_cublaslt_gemm.cu`, `optimized_cublaslt_gemm.cu`, `tcgen05_pipelined.cu` 외 | tcgen05 낮추기 및 점유율 튜닝을 보여주는 cuBLASLt 기반 행렬 곱셈 및 tcgen05 파이프라인 커널. |
| `baseline_cute_dsl_nvfp4_gemm.cu`, `optimized_cute_dsl_nvfp4_gemm.cu` 외 | 베이스라인 vs TMA 워프 특화 스케줄이 있는 CuTe-DSL 영감의 NVFP4 GEMM 쌍(대회 형태). |
| `baseline_cutlass_gemm_fp4.cu`, `optimized_cutlass_gemm_fp4.cu` 외 | 스케줄링을 분리하는 CUTLASS NVFP4 GEMM 쌍: 자동 스케줄링 vs 명시적 `KernelTmaWarpSpecialized1SmNvf4Sm100`. |
| `baseline_fused_l2norm.cu`, `optimized_fused_l2norm.cu` 외 | L2 norm + 스케일링을 수치적으로 안정적으로 병합하는 퓨전 예제. |
| `baseline_triton.py`, `optimized_triton.py` | 빠른 프로토타이핑 및 Blackwell에서 컴파일러 생성 PTX 검증을 위한 Triton 대응물. |
| `baseline_tcgen05_tma_pipeline.py`, `optimized_tcgen05_tma_pipeline.py`, `two_stage_pipeline.cu` | 스테이징된 TMA 로드 및 인라인 PTX 훅을 강조하는 생산자/소비자 파이프라인. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | 모든 예제에 대한 하네스 훅 및 회귀 임계치. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch09/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch09
python -m cli.aisp bench run --targets ch09 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python baseline_compute_bound.py --summaries`가 `baseline_memory_bound.py`보다 훨씬 높은 산술 강도를 보고하여 루프라인 플롯과 일치합니다.
- `python optimized_cublaslt_gemm.py --sizes 4096 4096 8192`가 동일한 디바이스에서 `baseline_cublaslt_gemm.py`에 비해 처리량을 개선합니다.
- `python compare.py --examples fused_l2norm`이 퓨전 전후에 수치적으로 동일한 출력을 확인합니다.

## 참고 사항
- `inline_ptx_example.cu`는 아키텍처 가드를 통해 tcgen05 인트린식을 안전하게 래핑하는 방법을 시연합니다.
- `baseline_cute_dsl_nvfp4_gemm.cu` / `optimized_cute_dsl_nvfp4_gemm.cu`는 CuTe DSL NVFP4 커널 워크스루에서 영감을 받았습니다.
- `requirements.txt`는 커널이 PyTorch 2.10-dev 기능을 추적하도록 Triton 나이틀리 피닝을 포함합니다.

# 챕터 8 - 점유율 및 파이프라인 튜닝

## 요약
리소스 균형에 집중합니다. 이중 버퍼링, 루프 언롤링, 비동기 파이프라인을 통해 TMEM 지연 시간을 숨기면서 SM을 가득 채우도록 블록 크기, 레지스터, 공유 메모리를 조정합니다.

## 학습 목표
- 점유율을 명시적으로 튜닝하고 레지스터 수가 상주 CTA를 어떻게 제한하는지 관찰합니다.
- 이중 버퍼링과 비동기 스테이징을 적용하여 DRAM 페치와 계산을 오버랩합니다.
- 타일링, 루프 언롤링, AI별 임계값을 사용하여 지연 시간과 처리량을 제어합니다.
- 공유 하네스를 사용하여 파이프라인된 스케줄이 SM/TMEM 활용률을 어떻게 변화시키는지 측정합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_occupancy_tuning.py`, `optimized_occupancy_tuning.py`, `occupancy_tuning_tool.py`, `occupancy_api_example.cu`, `occupancy_tuning.cu` | CTA 형태, 레지스터 캡, API 계산 한계를 튜닝하는 점유율 연구(빠른 프리셋 탐색을 위한 스윕 도구 포함). |
| `baseline_ai_optimization.py`, `optimized_ai_optimization.py`, `ai_optimization_kernels.cu`, `independent_ops.cu` | 파이프라인 및 점유율 트레이드오프를 강조하기 위해 독립적인 ops를 스테이징하는 AI 커널 스케줄링 샘플. |
| `baseline_hbm_cuda.cu`, `baseline_hbm.py`, `optimized_hbm.py`, `optimized_hbm_cuda_vectorized.cu` 외 | 스칼라, 벡터화, 비동기 페치 패턴을 비교하는 HBM 스트리밍 워크로드. |
| `baseline_loop_unrolling.cu`, `baseline_loop_unrolling.py`, `optimized_loop_unrolling.cu`, `optimized_loop_unrolling.py`, `loop_unrolling_kernels.cu` | 다양한 ILP 레짐을 목표로 하는 루프 언롤링 케이스 스터디. |
| `baseline_threshold.py`, `baseline_thresholdtma.py`, `optimized_threshold.py`, `optimized_thresholdtma.py`, `threshold_kernels.cu` | 스칼라, 벡터화, TMA 기반 파이프라인으로 구현된 임계값 연산자. |
| `baseline_tiling.py`, `baseline_tiling_tcgen05.py`, `optimized_tiling.py`, `optimized_tiling_tcgen05.py`, `tiling_kernels.cu` | tcgen05 행렬 곱셈을 위한 타일 스케줄러(tcgen05가 없을 때의 안전한 폴백 포함). |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | 하네스 진입점, 의존성, 회귀 임계치. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch08/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch08
python -m cli.aisp bench run --targets ch08 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `optimized_thresholdtma.py`에 대한 Nsight Compute 트레이스가 최소한의 유휴 사이클로 TMA 로드가 오버랩되는 것을 보여야 합니다.
- `python -m cli.aisp tools occupancy-tuning`이 점유율 튜닝 마이크로벤치마크에 대한 프리셋 타이밍 + 속도 향상을 출력합니다.
- `python compare.py --examples threshold`가 TMA 기반 커널이 스칼라 참조 구현에 비해 지연 시간을 줄임을 확인합니다.

## 참고 사항
- `arch_config.py`는 GPU별로 tcgen05 낮추기를 활성화/비활성화하는 토글을 노출하여 동일한 스크립트가 SM100과 SM121에서 작동합니다.
- `build/`는 구성별 CUDA 오브젝트 파일을 캐시합니다. 도구 체인을 조정할 때 `python cleanup.py --include-build`로 정리하세요.

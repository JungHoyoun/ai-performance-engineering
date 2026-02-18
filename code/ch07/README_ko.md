# 챕터 7 - 메모리 접근 패턴

## 요약
메모리 레이아웃이 성능을 어떻게 결정하는지 가르칩니다. 병합된 복사, 타일드 행렬 곱셈, 비동기 프리페치, TMA 전송, 룩업 집약적인 워크로드를 위한 공유 메모리 스테이징을 다룹니다.

## 학습 목표
- 스칼라, 병합된, 벡터화된 메모리 이동 간의 성능 차이를 측정합니다.
- 공유 메모리 타일링, TMA, 비동기 복사를 사용하여 텐서 코어를 포화 상태로 유지합니다.
- 룩업 집약적인 워크로드를 분석하고 캐시 스래싱 접근 패턴을 완화합니다.
- 전치 및 gather/scatter 패널티를 정량화하여 레이아웃 변경을 정당화합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_copy_scalar.cu`, `baseline_copy_uncoalesced.cu`, `optimized_copy_uncoalesced_coalesced.cu`, `optimized_copy_scalar_vectorized.cu` 외 | 병합, 벡터 너비, 워프 수준 효율성을 강조하는 복사 커널. |
| `baseline_hbm_copy.cu`, `baseline_hbm_peak.cu`, `optimized_hbm_copy.cu`, `optimized_hbm_peak.cu` 외 | CUDA 및 Python 하네스를 갖춘 HBM 최대 대역폭 프로브. |
| `baseline_async_prefetch.cu`, `optimized_async_prefetch.cu`, `baseline_tma_copy.cu`, `optimized_async_prefetch.py` 외 | 글로벌 메모리 페치와 계산을 오버랩하는 Async/TMA 샘플. |
| `baseline_matmul.cu`, `baseline_matmul.py`, `optimized_matmul_tiled.py`, `optimized_matmul_tiled.cu` | 나이브한 글로벌 메모리 접근과 공유 메모리 타일링 및 워프 수준 재사용을 비교하는 행렬 곱셈 구현. |
| `baseline_lookup.cu`, `baseline_lookup.py`, `optimized_lookup.cu`, `lookup_pytorch.py` | 테이블을 더 나은 지역성을 위해 재구성하는 방법을 보여주는 캐시 민감 룩업 워크로드. |
| `baseline_transpose.cu`, `baseline_transpose.py`, `optimized_transpose_padded.py` 외 | 뱅크 충돌을 최소화하는 방법을 보여주는 전치 및 gather/scatter 실험. |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json`, `memory_access_pytorch.py` | 하네스 진입점, 빌드 레시피, 기대값 임계치, PyTorch 검증 스크립트. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch07/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch07
python -m cli.aisp bench run --targets ch07 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python baseline_hbm_copy.py --bytes 1073741824`가 `optimized_hbm_copy.py`보다 눈에 띄게 낮은 GB/s를 보고하여 벡터화 및 비동기 복사가 작동함을 증명합니다.
- `python compare.py --examples async_prefetch`가 optimized_async_prefetch가 정확도를 유지하면서 총 커널 수를 줄임을 보여줍니다.
- `optimized_matmul_tiled.cu`의 Nsight Compute 캡처가 최소한의 뱅크 충돌로 >80% 공유 메모리 대역폭 활용률을 달성합니다.

## 참고 사항
- Python 행렬 곱셈 래퍼를 사용할 때 `TORCH_COMPILE_MODE`를 토글하여 원시 CUDA 커널과 함께 퓨전 이점을 확인하세요.
- HBM 도구는 현실적인 참조 상한을 제공하기 위해 `benchmark_peak_results_*.json`이 있을 때 실제 최대값을 읽습니다.

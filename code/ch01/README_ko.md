# 챕터 1 - 성능 기초

## 요약
간단한 학습 루프 굿풋 벤치마크와 소규모 CUDA GEMM 케이스 스터디를 통해 벤치마킹 기초를 확립합니다. 이후의 최적화를 반복 가능한 측정, 동등한 워크로드, 검증 가능한 출력에 기반하게 하는 것이 목표입니다.

## 학습 목표
- 공유 하네스로 최소한의 PyTorch 학습 루프를 프로파일링하고 처리량 대 지연 시간을 분석합니다.
- 알고리즘 워크로드를 변경하지 않고 기본 최적화(FP16 + 퓨전 마이크로배치)를 적용합니다.
- 배치형 vs 스트라이드형 GEMM 커널을 비교하여 산술 강도를 이해합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_performance.py`, `optimized_performance.py` | FP32 이거(eager) vs FP16 + 퓨전 마이크로배치(배치 퓨전)를 비교하는 굿풋 중심 학습 루프 쌍. |
| `baseline_gemm.cu`, `optimized_gemm_batched.cu`, `optimized_gemm_strided.cu` | 실행 분할 상각 및 메모리 병합을 설명하는 CUDA GEMM 변형(단일, 배치, 스트라이드). |
| `compare.py`, `workload_config.py`, `arch_config.py`, `expectations_{hardware_key}.json` | 하네스 진입점, 워크로드 형태, 아키텍처 오버라이드, 저장된 기대값 임계치. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch01/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python compare.py`가 기본 마이크로배치 크기에서 optimized_performance가 baseline 대비 토큰/초 >=2배를 달성했다고 보고합니다.
- `make && ./baseline_gemm_sm100` vs `./optimized_gemm_batched_sm100` 실행이 실행 횟수와 총 런타임에서 상당한 감소를 보입니다.

## 참고 사항
- `requirements.txt`는 헬퍼 스크립트에서 사용하는 경량 추가 의존성(Typer, tabulate)을 고정합니다.
- `Makefile`은 빠른 비교를 위해 SM별 접미사를 붙인 CUDA GEMM 바이너리를 빌드합니다.

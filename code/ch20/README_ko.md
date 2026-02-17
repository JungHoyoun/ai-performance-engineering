# 챕터 20 - 종합 케이스 스터디

## 요약
커널, 메모리, 파이프라인, 추론 최적화를 종합적인 케이스 스터디로 결합합니다. 베이스라인 파이프라인을 가져와 단계적인 개선을 적용하고, 모든 주요 서브시스템에 대한 이익 증명 아티팩트를 캡처합니다.

## 학습 목표
- 메모리, 파이프라인, KV 캐시 최적화를 연결하여 누적 영향을 확인합니다.
- 베이스라인 vs 튜닝된 엔드 투 엔드 실행을 비교하는 자동 보고서를 생성합니다.
- AI 커널 생성기를 통해 새 커널을 프로토타입하고 하네스에 슬롯합니다.
- 워크로드별 허용 테스트로 개선 사항을 검증합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_multiple_unoptimized.py`, `optimized_multiple_unoptimized.py`, `ai_kernel_generator.py`, `inductor_guard.py` | 여러 병목을 쌓는 복합 워크로드 및 후보 커널을 안전하게 생성하는 헬퍼. |
| `baseline_pipeline_sequential.py`, `optimized_pipeline_sequential.py`, `baseline_end_to_end_bandwidth.py`, `optimized_end_to_end_bandwidth.py` | 최적화가 스테이지 간에 어떻게 상호 작용하는지 보여주는 파이프라인 및 대역폭 케이스 스터디. |
| `baseline_integrated_kv_cache.py`, `optimized_integrated_kv_cache.py` | 할당자, 오버랩, NVLink 풀링 트릭을 병합하는 통합 KV 캐시 데모. |
| `baseline_memory_standard.py`, `optimized_memory_standard.py` | 시스템 수준에서 할당자 변경을 검증하는 메모리 중심 하네스. |
| `baseline_training_single.py`, `optimized_training_single.py`, `test.cu`, `Makefile` | 단일 디바이스 학습 케이스 스터디 및 최종 보고서에서 사용된 CUDA 커널. |
| `compare.py`, `arch_config.py`, `expectations_{hardware_key}.json` | 하네스 드라이버, 아키텍처 설정, 기대값 기준선. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch20/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch20
python -m cli.aisp bench run --targets ch20 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python compare.py`가 각 최적화된 변형이 저장된 기대값을 충족하거나 초과함을 보여주는 스테이지별 요약을 내보냅니다.
- `python ai_kernel_generator.py --emit test.cu`가 `nvcc`로 컴파일되고 수동 편집 없이 하네스에 통합되는 CUDA 커널을 생성합니다.
- `python optimized_pipeline_sequential.py --trace`가 전체 파이프라인을 아우르는 매끄러운 NVTX 범위를 보여주어 오버랩 성공을 시연합니다.

## 참고 사항
- `inductor_guard.py`는 기능 플래그 뒤에 실험적 커널을 게이팅하기 위한 편의 토글을 제공합니다.
- `ai_kernel_generator.py`는 재현성을 위해 생성된 코드를 `artifacts/`에 기록합니다. 이익 증명 번들과 함께 로그를 캡처하세요.

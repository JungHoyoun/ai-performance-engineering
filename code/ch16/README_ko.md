# 챕터 16 - 프로덕션 추론 최적화

## 요약
실제 추론 서비스에 초점을 맞춥니다. 페이지드 어텐션, Flash SDP, FP8 서빙, 텔레메트리 훅, 스케줄러, Blackwell 친화적 부하 테스트 하네스를 다룹니다.

## 학습 목표
- 모델을 배포하기 전에 대형 디코더 워크로드를 프로파일링하여 핫스팟을 파악합니다.
- 지연 시간 목표를 달성하기 위해 페이지드 어텐션, Flash SDP, 조각 컴파일을 채택합니다.
- 서빙 루프에 FP8 양자화, 대칭 메모리, 캐시 모니터링을 통합합니다.
- 당혹감 검사를 통해 정확도를 검증하면서 프로덕션 부하(다중 노드, MoE)를 시뮬레이션합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `inference_optimizations_blackwell.py`, `inference_profiling.py`, `inference_server_load_test.py`, `inference_serving_multigpu.py` | 다중 GPU 추론 배포를 프로파일링하고 부하 테스트하는 최상위 오케스트레이션 스크립트. |
| `baseline_flash_sdp.py`, `optimized_flash_sdp.py`, `baseline_paged_attention.py`, `optimized_paged_attention.py` | 나이브한 구현과 Flash/페이지드 변형을 비교하는 어텐션 커널. |
| `baseline_piece_graphs.py`, `optimized_piece_graphs.py`, `baseline_regional_compilation.py`, `optimized_regional_compilation.py` | 안정적인 저지연 디코드를 위한 조각별 그래프 캡처 및 지역 컴파일. |
| `fp8_transformer_engine.py`, `test_fp8_quantization_real.py`, `symmetric_memory_inference.py`, `multi_gpu_validation.py` | 정확도와 NVLink 효율성을 보장하기 위한 서빙 시간 FP8 및 대칭 메모리 검증. |
| `moe_performance_benchmark.py`, `synthetic_moe_inference_benchmark.py`, `moe_workload.py` | 라우터 배치 및 전문가별 배칭을 스트레스 테스트하는 MoE 추론 하네스. |
| `cache_monitoring.py`, `dcgm_prometheus_exporter.py`, `scheduler.py`, `perplexity_eval.py` | 추론 파이프라인에 연결된 텔레메트리, 스케줄링, 정확도 유틸리티. |
| `compare.py`, `requirements.txt`, `Makefile`, `expectations_{hardware_key}.json` | 추론 중심 검증을 위한 하네스 진입점 및 의존성. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch16/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch16
python -m cli.aisp bench run --targets ch16 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python optimized_paged_attention.py --profile minimal`이 베이스라인 스크립트에 비해 페이지 폴트가 적고 처리량이 향상됨을 나타냅니다.
- `python symmetric_memory_inference.py --validate`가 NVLink 기반 KV 복제본이 최소한의 왜곡으로 동기화를 유지함을 확인합니다.
- `python inference_server_load_test.py --duration 120`이 스케줄러를 실습하고 워밍업 후 안정적인 TTFT/TPOT 메트릭을 보고해야 합니다.

## 참고 사항
- `dcgm_prometheus_exporter.py`가 추가 설정 없이 Prometheus/Grafana가 소비할 수 있는 GPU별 메트릭을 내보냅니다.
- `cache_monitoring.py`는 실행 간 할당자 상태를 정상 확인하기 위해 독립적으로 실행할 수 있습니다.

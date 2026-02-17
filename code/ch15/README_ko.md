# 챕터 15 - 분리된 추론 및 KV 관리

## 요약
대규모 추론 관련 사항을 다룹니다. 분리된 계산/스토리지, NVLink를 통한 KV 캐시 풀링, 연속 배칭, 전문가 혼합(MoE) 서빙 패턴을 포함합니다.

## 학습 목표
- 단일 vs 분리된 추론 경로를 벤치마크하고 패브릭 비용을 정량화합니다.
- 로컬 및 원격 HBM 풀을 우아하게 아우르는 KV 캐시 관리자를 설계합니다.
- 디코드 처리량을 높게 유지하기 위해 연속 배칭 및 큐잉을 구현합니다.
- 최적화된 통신과 라우팅을 짝지어 MoE 모델을 효율적으로 서빙합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_inference_monolithic.py`, `optimized_inference_monolithic.py` | 분리 전 기준선을 확립하는 단일 박스 추론 루프. |
| `disaggregated_inference_multigpu.py` | 프리필/디코드 풀 위에 추측 디코딩을 레이어링하는 분리 추론 데모. |
| `baseline_disaggregated_inference.py`, `optimized_disaggregated_inference.py` 외 다수의 분리 파일 | 원격 프리필, 디코드 오버랩, NVLink 풀링을 모델링하는 분리 파이프라인(단일 및 다중 GPU). |
| `baseline_kv_cache_management.py`, `optimized_kv_cache_management.py`, `kv_cache_management_math.py` 외 | 로컬 전용, 수학 전용, NVLink 풀링 변형이 있는 KV 캐시 오케스트레이션 유틸리티. |
| `baseline_continuous_batching.py`, `optimized_continuous_batching.py` | TTFT 인식 큐잉을 위한 단일 GPU 연속 배칭 스케줄러. |
| `baseline_continuous_batching_multigpu.py`, `optimized_continuous_batching_multigpu.py` | 확장된 큐잉 처리량을 위한 다중 GPU 연속 배칭 스케줄러. |
| `baseline_moe_inference.py`, `optimized_moe_inference.py` | 라우터 부하와 통신 제어를 짝지우는 추론 특화 MoE 워크로드. |
| `baseline_moe_overlap.py`, `optimized_moe_overlap_shared_expert.py`, `baseline_wide_ep.py`, `optimized_wide_ep.py` 외 | 오버랩, 패킹/언패킹, 토폴로지 인식 라우팅 디스패치를 설명하는 MoE 전문가 병렬 마이크로벤치마크. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `Makefile` | 추론 중심 검증을 위한 하네스 진입점 및 의존성. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch15/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch15
python -m cli.aisp bench run --targets ch15 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python -m cli.aisp bench run --targets ch15:disaggregated_inference_multigpu --profile minimal --ncu-replay-mode kernel`이 정확도 동등성을 유지하면서 베이스라인에 비해 패브릭 지연을 줄임을 보여줍니다.
- `python optimized_kv_cache_management.py --validate`가 제거 및 승격 정책이 디코드 지연 시간을 예산 내로 유지함을 확인합니다.
- `python compare.py --examples continuous_batching`과 `python compare.py --examples continuous_batching_multigpu`가 최적화된 스케줄링이 단순한 큐 드레이닝보다 토큰/초를 높임을 보여줍니다.

## 참고 사항
- `disaggregated_inference_multigpu.py`는 순수 시뮬레이션 모드로 실행할 수 있습니다. 하드웨어가 NVLink 풀링을 위해 배선되지 않은 경우 `--simulate-network`를 설정하세요.
- 원하는 GPU 수에서 분리 파이프라인을 실행하려면 `torchrun --nproc_per_node <num_gpus>`를 사용하세요(기본값은 모든 가시 GPU, 짝수 개).
- `Makefile`은 다중 노드 디코드 실험에 필요한 MPI/UCX 타겟을 래핑합니다.

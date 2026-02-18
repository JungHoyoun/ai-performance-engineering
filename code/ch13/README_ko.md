# 챕터 13 - PyTorch 프로파일링 및 메모리 튜닝

## 요약
PyTorch 중심 최적화에 초점을 맞춥니다. 컴파일된 autograd, 메모리 프로파일링, FSDP/컨텍스트/전문가 병렬성, 동일한 하네스 인프라를 기반으로 하는 FP8/양자화 워크플로우를 다룹니다.

## 학습 목표
- PyTorch 학습 루프를 엔드 투 엔드로 프로파일링하여 굿풋, 메모리, 커널 트레이스를 캡처합니다.
- 오버헤드를 줄이기 위해 `torch.compile`, 지역 컴파일, 커스텀 할당자를 적용합니다.
- 단편화를 제거하기 위해 DataLoader, KV 캐시, 옵티마이저 상태를 튜닝합니다.
- Transformer Engine 통합으로 FP8/양자화 학습 레시피를 실습합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_training_standard.py`, `optimized_training_standard.py`, `train.py`, `train_deepseek_v3.py`, `train_deepseek_coder.py` | 이거(eager) vs 컴파일 경로와 DeepSeek 영감 구성을 보여주는 참조 학습 루프. |
| `baseline_dataloader_default.py`, `optimized_dataloader_default.py`, `baseline_memory_profiling.py`, `optimized_memory_profiling.py`, `memory_profiling.py` | 할당자 통계를 읽고 누수를 수정하는 방법을 설명하는 DataLoader/메모리 연구. |
| `baseline_attention_standard.py`, `optimized_attention_standard.py` 외 다수의 어텐션/행렬 곱셈 파일 | 장문 컨텍스트 Flash SDP를 포함하여 순수하게 PyTorch 내에서 튜닝된 어텐션 및 행렬 곱셈 마이크로벤치마크. |
| `baseline_context_parallel_multigpu.py`, `optimized_context_parallel_multigpu.py`, `context_parallel_benchmark_common.py` | 랭크 간 all-gather vs 링 스타일 스트리밍을 비교하는 컨텍스트 병렬 어텐션 벤치마크. |
| `baseline_expert_parallel_multigpu.py`, `optimized_expert_parallel_multigpu.py`, `expert_parallel_common.py` | 반복당 리스트 할당 vs 사전 할당된 all_to_all_single을 비교하는 전문가 병렬 all-to-all 벤치마크. |
| `context_parallelism.py`, `fsdp_example.py` | 단일 GPU를 넘어 확장하기 위한 컨텍스트 및 FSDP 샤딩 데모(도구, 벤치마크 타겟 아님). |
| `baseline_precisionfp8*.py`, `optimized_precisionfp8*.py`, `baseline_precisionmixed.py`, `optimized_precisionmixed.py`, `compiled_autograd.py` | Transformer Engine 및 컴파일된 autograd 레시피를 다루는 정밀도 관리 제품군. |
| `baseline_quantization.py`, `optimized_quantization.py`, `baseline_kv_cache_naive.py`, `optimized_kv_cache_naive.py`, `optimized_kv_cache_naive_pool.py` | 추론/학습 메모리 절약을 위한 양자화 및 KV 캐시 파이프라인. |
| `compare.py`, `compare_perf.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `workload_config.py` | 하네스 진입점, 성능 비교 헬퍼, 의존성, 회귀 기준선. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch13/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch13
python -m cli.aisp bench run --targets ch13 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python compare.py --examples training_standard`가 최적화된 학습 실행이 동일한 메트릭으로 더 높은 굿풋을 생성함을 보여줍니다.
- `python optimized_precisionfp8_te.py --validate`가 최대 오류 허용치가 적용된 Transformer Engine 캘리브레이션 및 NVFP8 실행을 확인합니다.
- `python memory_profiling.py --dump`와 최적화된 변형이 권장 파라미터 적용 후 할당자 단편화가 감소함을 시연합니다.

## 참고 사항
- `custom_allocator.py`는 단편화를 디버깅할 때 다른 챕터에서 재사용할 수 있는 독립형 torch 할당자 심(shim)을 포함합니다.
- `compiled_autograd.py`는 부분 그래프 캡처에 대한 튜토리얼 역할도 하며, 여기의 README는 이를 직접 참조합니다.

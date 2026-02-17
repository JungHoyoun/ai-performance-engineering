# 챕터 14 - 컴파일러 및 Triton 최적화

## 요약
컴파일러 기반 가속을 강조합니다. `torch.compile` 워크플로우, Triton 커널, CUTLASS/TMA 실험, 양자화 인식 통신을 다루며, 모두 공유 하네스를 통해 검증됩니다.

## 학습 목표
- 컴파일 시간과 안정적인 상태 이득을 추적하면서 대규모 모델에 `torch.compile` 모드를 채택합니다.
- 커스텀 CUDA와 경쟁하는 Triton 커널(TMA 스케줄 포함)을 작성합니다.
- FlexAttention 및 지역 컴파일 전략을 엔드 투 엔드로 프로파일링합니다.
- 회귀 없이 양자화를 NCCL 및 파이프라인 오버랩과 혼합합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_model_eager.py`, `optimized_model_eager.py`, `torch_compile_large_model.py`, `torch_compiler_examples.py`, `training_large_model_1_5x.py` | 컴파일 모드, 가드레일, 대규모 모델 건전성 테스트를 보여주는 모델 규모 예제. |
| `baseline_cutlass.py`, `optimized_cutlass.py`, `triton_examples.py`, `triton_tma_blackwell.py`, `triton_fp8_advanced.py`, `triton_nvshmem_example.py` | CUTLASS vs Triton 비교 및 고급 TMA/NVSHMEM Triton 커널. |
| `baseline_flex_attention.py`, `optimized_flex_attention.py`, `baseline_flex_attention_sparse.py`, `optimized_flex_attention_sparse.py`, `flex_attention_sparse_demo.py` | 커스텀 스코어 모드, 마스크, 희소성, 컴파일 속도 향상을 검증하는 FlexAttention 워크로드. |
| `baseline_nccl_quantization.py`, `optimized_nccl_quantization.py`, `deepseek_innovation_l2_bypass.py` | 양자화 인식 통신 및 DeepSeek 영감의 L2 바이패스 실험. |
| `baseline_regional_triton.py`, `optimized_regional_triton.py`, `inspect_compiled_code.py`, `benchmark_tma_configs.py` | 자동 생성 커널 자동 튜닝을 위한 지역 컴파일 및 TMA 파라미터 스윕. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `train.py`, `transformer.py` | 하네스 진입점, 모델 정의, 의존성 핀. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch14/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch14
python -m cli.aisp bench run --targets ch14 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python optimized_model_eager.py --profile minimal`이 베이스라인 대비 안정적인 상태 처리량 이득에 이어 컴파일 시간 요약을 생성합니다.
- `python triton_tma_blackwell.py --validate`가 Triton과 CUDA 출력을 비교하여 TMA 스케줄링 로직을 재확인합니다.
- `python compare.py --examples flex_attention`이 컴파일된 경로가 정확도를 변경하지 않고 커널 실행 횟수를 크게 줄임을 보여줍니다.

## 참고 사항
- `inspect_compiled_code.py`가 모든 타겟의 Triton/PTX/그래프 캡처를 덤프합니다. 새 워크로드를 내성하기 위해 헬퍼를 편집하세요.
- `requirements.txt`는 컴파일러 기능이 CUDA 13 도구 체인과 정렬되도록 나이틀리 Triton + PyTorch 휠을 포함합니다.

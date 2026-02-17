# 챕터 17 - 동적 라우팅 및 하이브리드 서빙

## 요약
라우터 설계, 분리된 추론, 프로파일링 기율을 혼합하여 Blackwell 클러스터가 활용률을 희생하지 않고 프리필/디코드 풀, MoE 전문가, 파이프라인 스테이지 간에 쿼리를 라우팅할 수 있게 합니다.

## 학습 목표
- TTFT, TPOT, KV 지역성 메트릭에 반응하는 동적 라우터를 구현합니다.
- 현실적인 합성 부하 하에서 완전한 추론 스택(프리필 + 디코드)을 프로파일링합니다.
- 장문 컨텍스트 워크로드를 위해 파이프라인 병렬성과 라우팅 로직을 혼합합니다.
- 라우팅 실험실에 특화된 프로파일링 단계(루프라인, Nsight)를 문서화합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_dynamic_routing.py`, `optimized_dynamic_routing.py`, `dynamic_routing.py`, `early_rejection.py` | 정적 휴리스틱에서 텔레메트리 기반 입장 및 거부 정책으로 발전하는 라우팅 컨트롤러. |
| `baseline_inference_full.py`, `optimized_inference_full.py` 외 다수의 프리필/디코드 분리 파일 | 분리된 프리필 및 디코드 풀을 모델링하는 엔드 투 엔드 추론 흐름(오버랩 중심, 배치 핸드오프, TTFT 중심, 장문 출력 TPOT 중심 다중 GPU 쌍 포함). |
| `baseline_pipeline_parallelism.py`, `optimized_pipeline_parallelism.py` | 계산 및 KV 전송 스케줄링을 결합하는 파이프라인 병렬 워크로드. |
| `baseline_moe_router_uniform.py`, `optimized_moe_router_uniform_topology.py` | 균일 vs 토폴로지 인식 라우팅을 비교하는 MoE 라우터 벤치마크 쌍. |
| `moe_router_uniform_demo.py`, `moe_router_topology_demo.py` | 균일 vs 토폴로지 인식 전문가 선택을 비교하는 MoE 라우팅 데모(비 벤치마크). |
| `baseline_routing_static.py`, `optimized_routing_static.py` | 정적/동적 샤딩 결정을 위한 라우터 변형(비교 가능한 벤치마크). |
| `baseline_memory.py`, `optimized_memory.py`, `blackwell_profiling_guide.py` | 라우팅 워크로드에 맞는 메모리 바운드 케이스 스터디 및 프로파일링 가이드. |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json`, `dynamo_config.yaml` | 하네스 진입점, 빌드 규칙, 기대값 기준선, Dynamo 구성 파라미터. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch17/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch17
python -m cli.aisp bench run --targets ch17 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python optimized_dynamic_routing.py --trace`가 베이스라인의 진동보다 더 빠르게 안정화되는 TTFT/TPOT 추세를 기록합니다.
- `python optimized_pipeline_parallelism.py --profile minimal`이 더 적은 유휴 버블로 프리필/디코드 세그먼트가 오버랩됨을 보여줍니다.
- `python -m cli.aisp tools roofline`이 최신 캡처를 사용하여 문서화된 루프라인 포인트를 재현합니다.

## 참고 사항
- `blackwell_profiling_guide.py`는 라우팅 집약적인 워크로드에 대한 Nsight Systems/Compute 캡처 및 루프라인 vs 점유율 병목 해석을 안내합니다.
- 분리된 프리필/디코드 베이스라인은 나이브한 스케줄링을 모델링하기 위해 요청별 블로킹 핸드오프와 요청별 동기화/배리어를 사용합니다. 최적화된 대응물은 오버랩하거나 처리량을 높이기 위해 그룹별 배치 또는 연속적인 KV/시드 슬랩을 전송합니다.

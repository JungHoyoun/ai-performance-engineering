# 챕터 5 - 스토리지 및 I/O 최적화

## 요약
GPU를 효율적으로 공급하는 것에 초점을 맞춥니다. DataLoader 워커 튜닝, 전처리 벡터화, 계산과 I/O 오버랩, NVMe 트래픽이 병목이 될 때 GPUDirect 스토리지 채택을 다룹니다.

## 학습 목표
- 하네스 메트릭을 통해 I/O 지연을 감지하고 GPU를 바쁘게 유지하도록 파이프라인을 재구성합니다.
- 대규모 배치 학습을 위해 PyTorch DataLoader 파라미터(워커, 프리페치, 핀 메모리)를 튜닝합니다.
- GPUDirect 스토리지 경로 vs 전통적인 CPU 중재 읽기를 평가합니다.
- 원격 스토리지 및 분산 데이터 읽기 전략을 벤치마크합니다.

## 디렉토리 구조
| 경로 | 설명 |
| --- | --- |
| `baseline_storage_cpu.py`, `optimized_storage_cpu.py` | 워커 수, 핀 메모리, 캐싱 전략을 다루는 단일 노드 데이터 로더 비교. |
| `baseline_vectorization.py`, `optimized_vectorization.py` | 전처리에서 Python 루프를 제거하는 벡터화된 파싱 및 메모리 맵 예제. |
| `baseline_ai.py`, `optimized_ai.py`, `storage_io_optimization.py` | 스트리밍 읽기 및 프리페치와 함께 계산을 오버랩하는 LLM 스타일 토큰 파이프라인. |
| `baseline_distributed.py`, `optimized_distributed.py` | 단일 GPU 합산 vs 선택적 분산 all-reduce 폴백. |
| `baseline_distributed_multigpu.py`, `optimized_distributed_multigpu.py` | 다중 GPU 감소 베이스라인(CPU 스테이징) vs GPU 측 reduce_add. |
| `gds_cufile_minimal.py`, `gpudirect_storage_example.py` | cuFile 설정, 버퍼 정렬, 처리량을 검증하는 GPUDirect 스토리지 샘플. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | 하네스 진입점 및 회귀를 감지하기 위한 기대값 기준선. |

## 벤치마크 실행
빠른 비교를 위해 벤치마크 하네스를 사용하거나, 반복 가능한 아티팩트 캡처가 필요할 때 Typer CLI를 사용하세요.
```bash
python ch05/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch05
python -m cli.aisp bench run --targets ch05 --profile minimal
```
- Nsight 트레이스를 캡처할 때 워크로드별로 `--profile` 또는 `--iterations`를 오버라이드하세요.
- 기대값 기준선은 각 챕터 옆에 `expectations_{hardware_key}.json`으로 저장됩니다. 새 하드웨어 검증 후 `--update-expectations`로 갱신하세요.

## 검증 체크리스트
- `python baseline_storage_cpu.py --inspect`가 CPU 대기 시간 > GPU 시간을 노출합니다. `optimized_storage_cpu.py`는 >=80% GPU 활용률로 비율을 역전시킵니다.
- `python gds_cufile_minimal.py --bytes 1073741824`가 `/etc/cufile.json`이 구성되고 NVMe가 GPUDirect 지원을 알릴 때 멀티 GB/s 처리량을 유지합니다.
- `python compare.py --examples ai`가 optimized_ai가 크리티컬 패스에서 CPU 측 전처리를 제거함을 보여줍니다.

## 참고 사항
- `libcufile.so`를 사용할 수 없을 때 GPUDirect 스크립트가 호스트 중재 읽기로 폴백하여 개발 랩톱에서도 안전하게 실행할 수 있습니다.
- `requirements.txt`는 데이터셋 심에 필요한 제한된 추가 의존성(`lmdb` 등)을 캡처합니다.

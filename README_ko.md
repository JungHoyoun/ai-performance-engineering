# AI 성능 엔지니어링

_**업데이트:** 이 내용을 직접 실습하는 강좌에 관심이 있으신가요?_

_관심이 있으시다면, 이 [**양식**](https://docs.google.com/forms/d/e/1FAIpQLSf4TMDLsPcfuoLhaDktXu-hhKIGntQm550BY-ov6bRT_VMJhQ/viewform?usp=sharing&ouid=111382272947765737941)을 작성하여 관심을 표명하고 알림을 받으세요._

## 이 저장소에 대하여

O'Reilly 도서를 위한 AI 시스템 성능 엔지니어링 코드, 도구 및 자료입니다. GPU 최적화, 분산 학습, 추론 스케일링, 그리고 현대 AI 워크로드를 위한 풀스택 성능 튜닝을 다룹니다.

[**채팅**](https://chatgpt.com/g/g-691a6b188d808191b16cdd2b7732cf11-ai-systems-performance-engineering)으로 이 책과 직접 대화하세요!

바로 [**코드**](code/)로 이동하기.

[![O'Reilly 도서](img/ai_sys_perf_engg_cover_cheetah_sm.png)](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

> **O'Reilly 도서 – 2025년 11월**
> [Amazon에서 구매하기](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

### AI 시스템 성능 엔지니어링 도서

현대 AI 시스템은 단순한 FLOP 성능 이상을 요구합니다. 하드웨어, 소프트웨어, 알고리즘 전반에 걸쳐 굿풋(goodput) 기반의 프로파일 우선 엔지니어링이 필요합니다. 이 실습 중심 가이드는 GPU, 인터커넥트, 런타임 스택을 효율적이고 신뢰할 수 있는 학습 및 추론 파이프라인으로 전환하는 방법을 보여줍니다.

Nsight와 PyTorch 프로파일러로 실제 병목을 진단하고, 대역폭과 메모리를 최대한 활용하며, 컴파일러 스택(PyTorch + OpenAI Triton)을 사용하여 고성능 커널을 만드는 방법을 배웁니다. 서빙 측면에서는 vLLM/SGLang, TensorRT-LLM, NVIDIA Dynamo를 활용한 고처리량 추론을 마스터하며, 분리된 프리필/디코드 및 페이지드 KV 캐시를 포함하여 예산 내에서 랙 전체로 확장하는 방법을 익힙니다.

실습 위주의 경험적 방법론과 케이스 스터디, 프로파일링 데이터를 활용하여, AI/ML 엔지니어, 시스템 엔지니어, 연구자, 그리고 대규모 학습/추론을 구축하거나 운영하는 플랫폼 팀에게 유용한 책입니다. 현대 NVIDIA GPU를 위한 수천 줄의 PyTorch 및 CUDA C++ 코드 예제가 포함되어 있습니다.

* 굿풋을 위해 프로파일링하기 – 단순 사용률이 아닌, Nsight Systems/Compute와 PyTorch 프로파일러를 사용하여 실제 병목 지점을 찾습니다.

* 메모리와 대역폭 최대화 – 레이아웃, 캐싱, 데이터 이동을 최적화하여 GPU를 지속적으로 바쁘게 유지합니다.

* 컴파일러로 튜닝하기 – PyTorch 컴파일러 스택과 Triton을 활용하여 C++ 보일러플레이트 없이 고성능 커널을 생성합니다.

* 안정적으로 학습 확장하기 – 병렬화 전략(DP, FSDP, TP, PP, CP, MoE)을 적용하고 계산/통신을 오버랩하여 버블을 최소화합니다.

* 조 단위 파라미터 모델 효율적으로 서빙하기 – vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo를 분리된 프리필/디코드 및 KV 캐시 이동과 함께 활용합니다.

* 토큰당 비용 절감 – 단순 최대 속도가 아닌 성능/전력, 달러당 처리량을 엔지니어링합니다.

* AI 지원 최적화 채택 – AI가 시스템의 수동 조정 한계를 넘어설 때 커널 합성 및 튜닝을 돕도록 합니다.

* 확신을 가지고 출시하기 – 200개 이상의 항목으로 구성된 [체크리스트](docs/appendix.md)를 적용하여 팀 전반에서 성과를 재현하고 회귀를 방지합니다.

### 저자 소개

Chris Fregly는 Netflix, Databricks, Amazon Web Services(AWS)에서 혁신을 이끌어 온 성능 엔지니어이자 AI 제품 리더입니다. 그는 AI/ML 제품 구축, 시장 진출 이니셔티브 확장, 대규모 생성형 AI 및 분석 워크로드의 비용 절감에 집중한 성능 중심 엔지니어링 팀을 이끌었습니다.

Chris는 두 권의 다른 O'Reilly 도서의 저자이기도 합니다: Data Science on AWS와 Generative AI on AWS. 또한 O'Reilly 강좌 "NVIDIA GPU로 프로덕션에서 고성능 AI"와 Andrew Ng과 함께하는 DeepLearning.ai 강좌 "대규모 언어 모델로 생성형 AI"의 제작자이기도 합니다.

그의 작업은 커널 수준 튜닝, 컴파일러 기반 가속, 분산 학습, 고처리량 추론에 걸쳐 있습니다. Chris는 [AI 성능 엔지니어링](https://www.meetup.com/ai-performance-engineering)이라는 월간 밋업을 주최합니다.

### 200개 이상의 항목으로 구성된 성능 [체크리스트](docs/appendix.md)

이 도서는 전체 수명주기를 다루는 현장 검증된 최적화를 담은 **200개 이상의 성능 [체크리스트](docs/appendix.md)**와 함께 제공됩니다. 즉시 적용할 수 있습니다:

- ✅ 성능 튜닝 마인드셋 및 비용 최적화
- ✅ 재현성 및 문서화 모범 사례
- ✅ 시스템 아키텍처 및 하드웨어 계획
- ✅ 운영 체제 및 드라이버 최적화
- ✅ GPU 프로그래밍 및 CUDA 튜닝
- ✅ 분산 학습 및 네트워크 최적화
- ✅ 효율적인 추론 및 서빙
- ✅ 전력 및 열 관리
- ✅ 최신 프로파일링 도구 및 기법
- ✅ 아키텍처별 최적화

### 링크

- **도서**: [Amazon에서 AI 시스템 성능 엔지니어링](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)
- **밋업**: [AI 성능 엔지니어링](https://www.meetup.com/ai-performance-engineering)
- **YouTube**: [AI 성능 엔지니어링 채널](https://www.youtube.com/@AIPerformanceEngineering)

> *AI 성능 엔지니어링 커뮤니티를 위해 샌프란시스코에서 제작*

### 주요 집중 영역

- **GPU 아키텍처, PyTorch, CUDA, OpenAI Triton 프로그래밍**
- **분산 학습 및 추론**
- **메모리 최적화 및 프로파일링**
- **PyTorch 성능 튜닝**
- **다중 노드 확장 전략**

## 도서 챕터

### 챕터 1: 소개 및 AI 시스템 개요

- AI 시스템 성능 엔지니어
- 벤치마킹 및 프로파일링
- 분산 학습 및 추론 확장
- 리소스의 효율적인 관리
- 팀 간 협업
- 투명성과 재현성

### 챕터 2: AI 시스템 하드웨어 개요

- CPU와 GPU "슈퍼칩"
- NVIDIA Grace CPU 및 Blackwell GPU
- NVIDIA GPU 텐서 코어와 트랜스포머 엔진
- 스트리밍 멀티프로세서, 스레드, 워프
- 초대규모 네트워킹
- NVLink와 NVSwitch
- 다중 GPU 프로그래밍

### 챕터 3: OS, Docker, Kubernetes 튜닝

- 운영 체제 구성
- GPU 드라이버 및 소프트웨어 스택
- NUMA 인식 및 CPU 피닝
- 컨테이너 런타임 최적화
- 토폴로지 인식 오케스트레이션을 위한 Kubernetes
- 메모리 격리 및 리소스 관리

### 챕터 4: 분산 네트워크 통신 튜닝

- 통신과 계산 오버랩
- 분산 다중 GPU 통신을 위한 NCCL
- NCCL의 토폴로지 인식
- 분산 데이터 병렬 전략
- NVIDIA 추론 전송 라이브러리 (NIXL)
- In-Network SHARP 집계

### 챕터 5: GPU 기반 스토리지 I/O 최적화

- 빠른 스토리지와 데이터 지역성
- NVIDIA GPUDirect 스토리지
- 분산 병렬 파일 시스템
- NVIDIA DALI를 활용한 멀티모달 데이터 처리
- 고품질 LLM 데이터셋 생성

### 챕터 6: GPU 아키텍처, CUDA 프로그래밍, 점유율 극대화

- GPU 아키텍처 이해
- 스레드, 워프, 블록, 그리드
- CUDA 프로그래밍 리뷰
- GPU 메모리 계층 구조 이해
- 높은 점유율과 GPU 활용률 유지
- 루프라인 모델 분석

### 챕터 7: GPU 메모리 접근 패턴 프로파일링 및 튜닝

- 병합된 vs 비병합 글로벌 메모리 접근
- 벡터화된 메모리 접근
- 공유 메모리를 이용한 타일링 및 데이터 재사용
- 워프 셔플 인트린식
- 비동기 메모리 프리페칭

### 챕터 8: 점유율 튜닝, 워프 효율성, 명령어 수준 병렬성

- GPU 병목 프로파일링 및 진단
- Nsight Systems 및 Compute 분석
- 점유율 튜닝
- 워프 실행 효율성 향상
- 명령어 수준 병렬성 노출

### 챕터 9: CUDA 커널 효율성 및 산술 강도 향상

- 다단계 마이크로 타일링
- 커널 퓨전
- 혼합 정밀도 및 텐서 코어
- 최적 성능을 위한 CUTLASS 활용
- 인라인 PTX 및 SASS 튜닝

### 챕터 10: 커널 내부 파이프라이닝 및 협력 스레드 블록 클러스터

- 커널 내부 파이프라이닝 기법
- 워프 특화 생산자-소비자 모델
- 영구 커널 및 메가커널
- 스레드 블록 클러스터 및 분산 공유 메모리
- 협력 그룹

### 챕터 11: 커널 간 파이프라이닝 및 CUDA 스트림

- 스트림을 사용한 계산과 데이터 전송 오버랩
- 스트림 순서 메모리 할당자
- 이벤트를 통한 세밀한 동기화
- CUDA 그래프를 통한 제로 오버헤드 실행

### 챕터 12: 동적 및 디바이스 측 커널 오케스트레이션

- 원자적 작업 큐를 통한 동적 스케줄링
- CUDA 그래프로 반복 커널 실행 일괄 처리
- 동적 병렬성
- NVSHMEM을 통한 다중 GPU 오케스트레이션

### 챕터 13: PyTorch 프로파일링, 튜닝, 확장

- NVTX 마커 및 프로파일링 도구
- PyTorch 컴파일러 (torch.compile)
- PyTorch에서 메모리 프로파일링 및 튜닝
- PyTorch Distributed로 확장
- HTA를 활용한 다중 GPU 프로파일링

### 챕터 14: PyTorch 컴파일러, XLA, OpenAI Triton 백엔드

- PyTorch 컴파일러 심층 분석
- OpenAI Triton으로 커스텀 커널 작성
- PyTorch XLA 백엔드
- 고급 Triton 커널 구현

### 챕터 15: 다중 노드 추론 병렬성 및 라우팅

- 분리된 프리필 및 디코드 아키텍처
- MoE 모델을 위한 병렬성 전략
- 추측 및 병렬 디코딩 기법
- 동적 라우팅 전략

### 챕터 16: 대규모 추론 프로파일링, 디버깅, 튜닝

- 성능 프로파일링 및 튜닝 워크플로우
- 동적 요청 배칭 및 스케줄링
- 시스템 수준 최적화
- 실시간 추론을 위한 양자화 접근법
- 애플리케이션 수준 최적화

### 챕터 17: 분리된 프리필 및 디코드 확장

- 프리필-디코드 분리의 이점
- 프리필 워커 설계
- 디코드 워커 설계
- 분리된 라우팅 및 스케줄링 정책
- 확장성 고려 사항

### 챕터 18: 고급 프리필-디코드 및 KV 캐시 튜닝

- 최적화된 디코드 커널 (FlashMLA, ThunderMLA, FlexDecoding)
- KV 캐시 활용 및 관리 튜닝
- 이기종 하드웨어 및 병렬성 전략
- SLO 인식 요청 관리

### 챕터 19: 동적 및 적응형 추론 엔진 최적화

- 적응형 병렬성 전략
- 동적 정밀도 변경
- 커널 자동 튜닝
- 런타임 튜닝을 위한 강화학습 에이전트
- 적응형 배칭 및 스케줄링

### 챕터 20: AI 지원 성능 최적화

- AlphaTensor AI 발견 알고리즘
- 자동화된 GPU 커널 최적화
- 자기 개선 AI 에이전트
- 수백만 GPU 클러스터를 향한 확장

## 커뮤니티 리소스

20개 이상의 도시에서 10만 명 이상의 회원을 보유한 월간 밋업:

- [YouTube 채널](https://www.youtube.com/@AIPerformanceEngineering)
- [밋업 그룹](https://www.meetup.com/ai-performance-engineering)

최근 세션:

- [동적 적응형 RL 추론 CUDA 커널 튜닝](resources/Dynamic_Adaptive_RL_Inference_CUDA_Kernel_Tuning.pdf)
- [고성능 에이전틱 AI 추론 시스템](resources/High_Performance_Agentic_AI_Inference_Systems.pdf)
- [PyTorch 모델 최적화](resources/PyTorch_Model_Optimization.pdf)

### 월간 밋업 요약
- **2026년 2월 16일** - [YouTube 공개 예정](https://www.youtube.com/@AIPerformanceEngineering) & 슬라이드: [Verda의 Riccardo Mereu가 발표하는 NVFP4 저정밀 수치](resources/NVFP4_Low_Precision_Numerics_Verda.pdf)
- **2026년 1월 19일** - [YouTube](https://www.youtube.com/watch?v=o-etY6VLHZo) & 슬라이드: [Chaim Rand의 NVIDIA Nsight Systems 프로파일러를 통한 데이터 전송 최적화](resources/Optimizing_Data_Transfer_With_NVIDIA_Nsight_Systems_Profiler_by_Chaim_Rand.pdf)
- **2025년 11월 17일** - [YouTube](https://youtu.be/2EWDG_Dxjs8) & 슬라이드: Abdul Dakkak의 NVIDIA/AMD GPU 및 Modular 플랫폼을 이용한 광속 추론
- **2025년 10월 20일** - [YouTube](https://youtu.be/d3ZLodGTlAo): AI 기반 GPU 커널 최적화 + nbdistributed를 활용한 분산 PyTorch
- **2025년 9월 15일** – [YouTube](https://www.youtube.com/watch?v=eLnHXL1xXfM): 동적 적응형 RL 추론 커널 튜닝 심층 분석
- **2025년 8월 18일** – [YouTube](https://www.youtube.com/watch?v=SBPlOUww57I): 다중 GPU 오케스트레이션 전략 및 Nsight 프로파일링 케이스 스터디
- **2025년 7월 21일** – [YouTube](https://youtu.be/jaiMotxv8ck): FlashMLA, ThunderMLA, FlexDecoding 커널 워크스루와 라이브 Nsight Compute 데모
- **2025년 6월 16일** – 슬라이드: 분리된 추론 라우팅을 다루는 [고성능 에이전틱 AI 추론 시스템](resources/High_Performance_Agentic_AI_Inference_Systems.pdf)
- **2025년 5월 19일** – [YouTube](https://youtu.be/F8jJwI9xHTE) & [PyTorch 데이터 로더 최적화](resources/PyTorch_Model_Optimization_Data_Loader.pdf): Torch.compile 파이프라인, 데이터 로더 처리량 튜닝, 크로스 아키텍처 CUDA/ROCm 커널
- **2025년 4월 21일** – [YouTube](https://youtu.be/XoZcY_fDUKA) & [AI 성능 엔지니어링 밋업 슬라이드](resources/AI_Performance_Engineering_Meetup_Apr_21_2025.pdf): 종합적인 GPU 성능 플레이북 및 [PyTorch 모델 최적화](resources/PyTorch_Model_Optimization.pdf) 워크샵

## 기여하기

기여를 환영합니다! 코드, 문서, 성능 개선에 대한 가이드라인은 `CONTRIBUTING.md`를 참조하세요.

## 라이선스

Apache 2.0 라이선스 – 자세한 내용은 `LICENSE`를 참조하세요.

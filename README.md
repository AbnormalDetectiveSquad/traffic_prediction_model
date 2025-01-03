# 교통 예측 모델 (Traffic_prediction_model)

## 개요

본 프로젝트는 STGCN(Spatio-Temporal Graph Convolutional Networks) 기반의 딥러닝 모델을 사용하여 도시 교통 흐름을 예측하는 것을 목표로 합니다. 교통 흐름은 시간, 공간, 외부 요인(요일, 기상, 휴일 등)에 따라 복잡하게 변화하는 시계열 데이터입니다. 본 프로젝트에서는 이러한 복잡성을 효과적으로 모델링하기 위해 그래프 컨볼루션과 시간 컨볼루션을 결합한 STGCN 아키텍처를 사용합니다. 또한, 요일, 기상, 시간, 휴일 정보를 핫 인코딩하여 모델의 입력으로 제공함으로써 예측 정확도를 향상시키고자 합니다.

## 주요 특징

*   **STGCN 기반 모델:** 교통 네트워크의 공간적, 시간적 특성을 효과적으로 학습합니다.
*   **핫 인코딩된 외부 요인:** 요일, 기상, 시간, 휴일 정보를 활용하여 예측 정확도를 개선합니다.
*   **확장 가능성:** 다양한 교통 데이터(속도, 교통량, 점유율 등)에 적용 가능하며, 다른 외부 요인(사고 정보, 행사 정보 등)을 추가할 수 있습니다.
*   **(추가적인 특징이 있다면 여기에 작성)**

## 아키텍처

본 프로젝트는 다음과 같은 아키텍처를 사용합니다.

*   **입력:**
    *   **교통 데이터:** 각 링크의 교통 정보 5분간격 데이터 (예: 속도, 교통량).
    *   **공간 정보:** 교통 네트워크의 연결 관계를 나타내는 인접 행렬.
    *   **외부 요인:** 요일, 기상, 시간, 휴일 등 외부 정보를 핫 인코딩한 벡터.
*   **모델:** STGCN (Spatio-Temporal Graph Convolutional Networks)
    *   **Graph Convolutional Networks (GCN):** 교통 네트워크의 공간적 의존성을 학습합니다.
    *   **Temporal Convolutional Networks (TCN):** 교통 데이터의 시간적 의존성을 학습합니다.
*   **출력:** 미래 5분 뒤 시점의 각 노드별 교통 정보 예측값.
## 개발 환경

본 프로젝트는 다음 환경에서 개발 및 테스트되었습니다.

*   **하드웨어:**
    *   그래픽 카드: RTX 4090 또는 RTX 4070 Ti
*   **OS:** Ubuntu 24.04 (CUDA 12.4 강제 설치 후 진행)
*   **드라이버 버전:** NVIDIA-SMI 550.120 Driver Version: 550.120 CUDA Version: 12.4
*   **CUDA:** 12.4 ([CUDA 12.4 다운로드](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local))
*   **cuDNN:**
    *   Ubuntu 22.04: [cuDNN for Ubuntu 22.04](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
    *   Ubuntu 24.04: [cuDNN for Ubuntu 24.04](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)
## 주요 사용 기술 및 라이브러리

*   **Python:** 3.12
*   **PyTorch 생태계 (GPU 지원):**
    *   `torch==2.5.1` 
    *   `torchvision==0.20.1` 
    *   `torchaudio==2.5.1` 
    *   `torch-geometric==2.6.1` (그래프 신경망 관련 기능 제공)
    *   CUDA 12 관련 라이브러리 (GPU 가속을 위한 NVIDIA 라이브러리. CUDA 12 환경에서 PyTorch 사용 시 필요)
*   **핵심 과학/수학 라이브러리:**
    *   `numpy==2.2.1` (수치 계산의 기본)
    *   `scipy==1.14.1` (과학 및 공학 계산)
*   **그래프 분석:**
    *   `networkx==3.4.2` 
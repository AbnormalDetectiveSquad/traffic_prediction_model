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
*   **OS:** Ubuntu 22.04 / 24.04 (Ubuntu 24.04에서는 CUDA 12.4 강제 설치 후 진행)
*   **드라이버 버전:** NVIDIA-SMI 550.120 Driver Version: 550.120 CUDA Version: 12.4
*   **CUDA:** 12.4 ([CUDA 12.4 다운로드](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local))
*   **cuDNN:**
    *   Ubuntu 22.04: [cuDNN for Ubuntu 22.04](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
    *   Ubuntu 24.04: [cuDNN for Ubuntu 24.04](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)
*   **Python:** 3.x
*   **PyTorch, torchvision, torchaudio:** `pip3 install torch torchvision torchaudio` (CUDA 12.4 버전용 PyTorch 사용)
*   **PyTorch 관련:**
    *   `torch==2.5.1`
    *   `torchvision==0.20.1`
    *   `torchaudio==2.5.1`
*   **NVIDIA CUDA 관련 (CUDA 12):**
    *   `nvidia-cublas-cu12==12.4.5.8`
    *   `nvidia-cuda-cupti-cu12==12.4.127`
    *   `nvidia-cuda-nvrtc-cu12==12.4.127`
    *   `nvidia-cuda-runtime-cu12==12.4.127`
    *   `nvidia-cudnn-cu12==9.1.0.70`
    *   `nvidia-cufft-cu12==11.2.1.3`
    *   `nvidia-curand-cu12==10.3.5.147`
    *   `nvidia-cusolver-cu12==11.6.1.9`
    *   `nvidia-cusparse-cu12==12.3.1.170`
    *   `nvidia-nccl-cu12==2.21.5`
    *   `nvidia-nvjitlink-cu12==12.4.127`
    *   `nvidia-nvtx-cu12==12.4.127`
*   **수학 및 과학 연산:**
    *   `numpy==2.2.1`
    *   `sympy==1.13.1`
    *   `mpmath==1.3.0`
*   **네트워크 분석:**
    *   `networkx==3.4.2`
*   **이미지 처리:**
    *   `pillow==11.1.0`
*   **기타:**
    *   `filelock==3.16.1`
    *   `fsspec==2024.12.0`
    *   `Jinja2==3.1.5`
    *   `MarkupSafe==3.0.2`
    *   `setuptools==75.6.0`
    *   `triton==3.1.0`
    *   `typing_extensions==4.12.2`

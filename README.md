# InceptionNet

<p align="center">
  <img src="https://raw.githubusercontent.com/pytorch/vision/main/docs/source/_static/img/inception.png" alt="InceptionNet Architecture" width="600">
</p>

[English](#english) | [한국어](#korean)

---

<a name="english"></a>
## English

### Overview
InceptionNet (GoogLeNet) is a convolutional neural network architecture that was introduced in the paper "Going Deeper with Convolutions" by Szegedy et al. in 2014. It was the winner of the ILSVRC 2014 competition, significantly reducing the error rate compared to previous architectures.

The key innovation of InceptionNet is the inception module, which allows the network to extract features at different scales simultaneously by using filters of different sizes in parallel.

### Features
- Implementation of the InceptionNet (GoogLeNet) architecture in PyTorch
- Support for auxiliary classifiers during training to combat vanishing gradients
- Configurable dropout rates for both main and auxiliary classifiers
- Weight initialization options

### Project Structure
```
InceptionNet/
├── data/             # Directory for datasets
├── models/           # Neural network models
│   └── inception_net.py # InceptionNet implementation
└── utils/            # Utility functions
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/InceptionNet.git
cd InceptionNet

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage
```python
import torch
from models.inception_net import InceptionNet

# Create model instance
model = InceptionNet(num_classes=1000, use_aux=True)

# Create a sample input tensor
batch_size = 1
input_tensor = torch.randn(batch_size, 3, 224, 224)

# Forward pass
output, aux1, aux2 = model(input_tensor)

# During training, combine main loss with auxiliary losses
# loss = main_loss + 0.3 * (aux1_loss + aux2_loss)

# During inference, only use the main output
predictions = torch.argmax(output, dim=1)
```

### Key Components
- **BasicConv2d**: Basic convolution block (Conv2d + BatchNorm2d + ReLU)
- **Inception**: Implementation of the inception module with four parallel branches
- **Auxiliary**: Auxiliary classifier for additional gradient flow during training
- **InceptionNet**: Complete implementation of the GoogLeNet architecture

### References
- [Going Deeper with Convolutions (2014)](https://arxiv.org/abs/1409.4842)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

<a name="korean"></a>
## 한국어

### 개요
InceptionNet(GoogLeNet)은 2014년 Szegedy 등이 발표한 "Going Deeper with Convolutions" 논문에서 소개된 컨볼루션 신경망 아키텍처입니다. ILSVRC 2014 대회에서 우승하여 이전 아키텍처에 비해 오류율을 크게 감소시켰습니다.

InceptionNet의 핵심 혁신은 인셉션 모듈로, 다양한 크기의 필터를 병렬로 사용하여 동시에 다른 스케일의 특징을 추출할 수 있게 합니다.

### 특징
- PyTorch로 구현된 InceptionNet(GoogLeNet) 아키텍처
- 그래디언트 소실 문제를 해결하기 위한 보조 분류기 지원
- 메인 및 보조 분류기의 드롭아웃 비율 설정 가능
- 가중치 초기화 옵션 제공

### 프로젝트 구조
```
InceptionNet/
├── data/             # 데이터셋 디렉토리
├── models/           # 신경망 모델
│   └── inception_net.py # InceptionNet 구현
└── utils/            # 유틸리티 함수
```

### 설치
```bash
# 저장소 복제
git clone https://github.com/laplace1009/InceptionNet.git
cd InceptionNet

# 가상 환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows의 경우: venv\Scripts\activate

# 의존성 설치
pip install torch torchvision
```

### 사용법
```python
import torch
from models.inception_net import InceptionNet

# 모델 인스턴스 생성
model = InceptionNet(num_classes=1000, use_aux=True)

# 샘플 입력 텐서 생성
batch_size = 1
input_tensor = torch.randn(batch_size, 3, 224, 224)

# 순방향 전파
output, aux1, aux2 = model(input_tensor)

# 학습 중에는 메인 손실과 보조 손실을 결합
# loss = main_loss + 0.3 * (aux1_loss + aux2_loss)

# 추론 중에는 메인 출력만 사용
predictions = torch.argmax(output, dim=1)
```

### 주요 구성 요소
- **BasicConv2d**: 기본 컨볼루션 블록 (Conv2d + BatchNorm2d + ReLU)
- **Inception**: 네 개의 병렬 브랜치를 가진 인셉션 모듈 구현
- **Auxiliary**: 학습 중 추가 그래디언트 흐름을 위한 보조 분류기
- **InceptionNet**: GoogLeNet 아키텍처의 완전한 구현

### 참고 자료
- [Going Deeper with Convolutions (2014)](https://arxiv.org/abs/1409.4842)
- [PyTorch 문서](https://pytorch.org/docs/stable/index.html)
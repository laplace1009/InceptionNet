import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    """
    기본 컨볼루션 블록: 컨볼루션 + 배치 정규화 + ReLU 활성화 함수
    Basic convolution block: Convolution + Batch Normalization + ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block(x)

class Inception(nn.Module):
    """
    Inception 모듈: 다양한 크기의 필터를 병렬로 사용하여 다양한 스케일의 특징을 추출
    Inception module: Uses filters of different sizes in parallel to extract features at different scales
    """
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        """
        Args:
            in_channels: 입력 채널 수 (number of input channels)
            ch1x1: 1x1 컨볼루션 출력 채널 수 (output channels for 1x1 convolution branch)
            ch3x3_reduce: 3x3 컨볼루션 전 1x1 감소 레이어의 출력 채널 수 (output channels for 1x1 reduction before 3x3 conv)
            ch3x3: 3x3 컨볼루션 출력 채널 수 (output channels for 3x3 convolution)
            ch5x5_reduce: 5x5 컨볼루션 전 1x1 감소 레이어의 출력 채널 수 (output channels for 1x1 reduction before 5x5 conv)
            ch5x5: 5x5 컨볼루션 출력 채널 수 (output channels for 5x5 convolution)
            pool_proj: 풀링 후 1x1 프로젝션의 출력 채널 수 (output channels for 1x1 projection after pooling)
        """
        super().__init__()
        # 1x1 컨볼루션 브랜치 (1x1 convolution branch)
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        
        # 3x3 컨볼루션 브랜치 (3x3 convolution branch)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_reduce, kernel_size=1),
            BasicConv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
        )
        
        # 5x5 컨볼루션 브랜치 (5x5 convolution branch)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5_reduce, kernel_size=1),
            BasicConv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
        )
        
        # 맥스 풀링 브랜치 (Max pooling branch)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        # 각 브랜치의 출력을 계산하고 채널 차원에서 연결 (Compute outputs of each branch and concatenate along channel dimension)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

class Auxiliary(nn.Module):
    """
    보조 분류기: 네트워크의 중간층에서 그래디언트 소실 문제를 줄이기 위한 추가 손실 계산용
    Auxiliary classifier: Used to compute additional loss from intermediate layers to combat vanishing gradients
    """
    def __init__(self, in_channels, num_classes, dropout_rate=0.7):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class InceptionNet(nn.Module):
    """
    InceptionNet (GoogLeNet): 다양한 크기의 Inception 모듈을 사용하는 CNN 아키텍처
    InceptionNet (GoogLeNet): CNN architecture that uses Inception modules of various sizes
    """
    def __init__(self, num_classes = 1000, use_aux=True, init_weights=None, dropout_inception=0.4, dropout_aux=0.7):
        """
        Args:
            num_classes: 분류할 클래스 수 (number of classes for classification)
            use_aux: 보조 분류기 사용 여부 (whether to use auxiliary classifiers)
            init_weights: 가중치 초기화 방법 (weight initialization method)
            dropout_inception: 메인 분류기의 드롭아웃 비율 (dropout rate for main classifier)
            dropout_aux: 보조 분류기의 드롭아웃 비율 (dropout rate for auxiliary classifiers)
        """
        super().__init__()
        self.use_aux = use_aux

        # 입력 스템 (Input stem)
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=7, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv2d(64, 64, kernel_size=1)
        self.conv2b = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception 모듈 (Inception modules)
        # Inception 3 블록 (Inception 3 blocks)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception 4 블록 (Inception 4 blocks)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(512, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception 5 블록 (Inception 5 blocks)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        # 보조 분류기 (Auxiliary classifiers)
        if use_aux:
            self.aux1 = Auxiliary(512, num_classes, dropout_rate=dropout_aux)
            self.aux2 = Auxiliary(528, num_classes, dropout_rate=dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None
            
        # 최종 분류 (Final classification)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout_inception)
        self.fc = nn.Linear(1024, num_classes)

        # 가중치 초기화 (Weight initialization)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
        
    def forward(self, x):
        # 입력 스템 (Input stem)
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.max_pool2(x)
        
        # Inception 3 블록 (Inception 3 blocks)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool3(x)
        
        # Inception 4 블록 (Inception 4 blocks)
        x = self.inception4a(x)
        # 첫 번째 보조 분류기 (First auxiliary classifier)
        if self.aux1 is not None and self.training:
            aux1 = self.aux1(x)
        else:
            aux1 = None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        # 두 번째 보조 분류기 (Second auxiliary classifier)
        if self.aux2 is not None and self.training:
            aux2 = self.aux2(x)
        else:
            aux2 = None
        x = self.inception4e(x)
        x = self.max_pool4(x)
        
        # Inception 5 블록 (Inception 5 blocks)
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # 최종 분류 (Final classification)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x, aux1, aux2

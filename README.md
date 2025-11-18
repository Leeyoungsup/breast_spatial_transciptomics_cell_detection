# Tissue Context를 활용한 Late Fusion YOLO Detection

## 📋 개요

이 프로젝트는 기존 YOLOv11 detection 모델에 **tissue context 정보를 추가 입력**으로 받아 **late fusion**을 통해 **클래스 분류 성능을 향상**시키는 구조입니다.

### 주요 특징

- ✅ **이중 입력 구조**: 메인 detection 이미지 + Tissue context 이미지
- ✅ **Late Fusion**: Context 정보를 detection head의 classification branch에서 융합
- ✅ **독립적인 Context Encoder**: Tissue context를 별도 네트워크로 처리
- ✅ **기존 코드 호환성**: `use_context=False`로 기존 방식도 사용 가능

---

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    Main Detection Branch                 │
│  Input Image → DarkNet → FPN → Head (BBox + Class)      │
└─────────────────────────────┬───────────────────────────┘
                              │
                              │ Late Fusion
                              ↓
┌─────────────────────────────────────────────────────────┐
│                  Tissue Context Branch                   │
│  Context Image → TissueContextEncoder → Global Features │
│                                          ↓               │
│                              Context Fusion Layers       │
│                                          ↓               │
│                         Modulate Class Features          │
└─────────────────────────────────────────────────────────┘
```

### 세부 구조

1. **메인 Detection Branch**
   - `DarkNet`: Feature extraction backbone
   - `DarkFPN`: Feature pyramid network
   - `Head`: Bounding box regression + Classification

2. **Tissue Context Branch** (새로 추가)
   - `TissueContextEncoder`: Context 이미지 인코더
     - 3개의 Conv-CSP 블록으로 feature 추출
     - Global Average Pooling으로 전역 context vector 생성
     - MLP로 context feature 투영
   
3. **Late Fusion Mechanism**
   - Context features → Fusion layers (각 detection scale별)
   - Classification features를 element-wise modulation
   - `cls_feat = cls_feat * (1 + context_weight)`

---

## 📁 파일 구조

```
nets/
  ├── ContextNn.py              # 메인 모델 (Late Fusion 구조 포함)
  └── nn.py                     # 기존 모델 (참고용)

utils/
  ├── dataset.py                # 기존 Dataset
  └── dataset_with_context.py  # Context를 지원하는 Dataset (새로 추가)

train_with_context.py           # Context를 사용한 학습 스크립트
example_context_usage.py        # 사용 예시 및 데모
CONTEXT_FUSION_README.md        # 이 문서
```

---

## 🚀 사용 방법

### 1. 모델 생성

```python
from nets.ContextNn import yolo_v11_n

# Context 없이 사용 (기존 방식)
model = yolo_v11_n(num_classes=5, use_context=False)

# Context와 함께 사용 (Late Fusion)
model_with_context = yolo_v11_n(num_classes=5, use_context=True)
```

### 2. Forward Pass

```python
import torch

# 입력 데이터 준비
batch_size = 4
main_image = torch.randn(batch_size, 3, 640, 640).cuda()
tissue_context = torch.randn(batch_size, 3, 640, 640).cuda()

# Forward
model.eval()
with torch.no_grad():
    # Context 없이
    output = model(main_image)
    
    # Context와 함께 (Late Fusion)
    output = model(main_image, tissue_context)
```

### 3. 학습

#### 방법 1: 제공된 학습 스크립트 사용

```bash
# 기본 사용
python train_with_context.py \
    --data-dir /path/to/dataset \
    --batch-size 16 \
    --epochs 300

# Context 이미지가 별도 폴더에 있는 경우
python train_with_context.py \
    --data-dir /path/to/dataset \
    --context-dir /path/to/context_images \
    --batch-size 16 \
    --epochs 300
```

#### 방법 2: 커스텀 학습 루프

```python
from nets.ContextNn import yolo_v11_n
from utils.dataset_with_context import create_context_dataloader

# 모델 생성
model = yolo_v11_n(num_classes=5, use_context=True)
model.cuda()
model.train()

# DataLoader 생성
loader = create_context_dataloader(
    filenames=train_files,
    input_size=640,
    params=params,
    batch_size=16,
    augment=True,
    context_filenames=None  # 자동 매칭
)

# 학습 루프
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for samples, context_samples, targets in loader:
    samples = samples.cuda().float() / 255
    context_samples = context_samples.cuda().float() / 255
    
    # Forward with context
    outputs = model(samples, context_samples)
    
    # Loss 계산 및 최적화
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 📊 Context 이미지 준비

Context 이미지는 다음 두 가지 방법으로 제공할 수 있습니다:

### 방법 1: 자동 매칭 (권장)

메인 이미지와 같은 폴더에 `_context` suffix를 붙여 저장:

```
dataset/
  images/
    ├── sample001.jpg          # 메인 이미지
    ├── sample001_context.jpg  # Context 이미지
    ├── sample002.jpg
    ├── sample002_context.jpg
    └── ...
```

Dataset 생성 시 `context_filenames=None`으로 설정하면 자동으로 찾습니다.

### 방법 2: 별도 폴더

Context 이미지를 별도 폴더에 보관:

```
dataset/
  main_images/
    ├── sample001.jpg
    ├── sample002.jpg
    └── ...
  context_images/
    ├── sample001.jpg
    ├── sample002.jpg
    └── ...
```

```python
# 별도로 context_filenames 지정
context_files = ['/path/to/context_images/sample001.jpg', ...]
loader = create_context_dataloader(
    filenames=main_files,
    context_filenames=context_files,
    ...
)
```

---

## 🔍 주요 클래스 설명

### `TissueContextEncoder`

Tissue context 이미지를 전역 feature vector로 인코딩합니다.

```python
class TissueContextEncoder(torch.nn.Module):
    def __init__(self, width, depth):
        # Conv layers로 feature 추출
        # Global pooling으로 context vector 생성
        # MLP로 투영
        
    def forward(self, x):
        # Input: [B, 3, H, W]
        # Output: [B, context_dim]
```

### `Head` (수정됨)

Detection head에 late fusion 기능 추가:

```python
class Head(torch.nn.Module):
    def __init__(self, nc=80, filters=(), use_context=False, context_dim=0):
        # use_context=True일 때:
        # - context_fusion: Context feature를 class feature 공간으로 투영
        # - cls_final: Fusion 후 최종 classification
        
    def forward(self, x, context_features=None):
        # Context features로 classification features를 modulate
        # cls_feat = cls_feat * (1 + context_weight)
```

### `YOLO` (수정됨)

전체 모델을 관리:

```python
class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes, use_context=False):
        # use_context=True일 때 TissueContextEncoder 추가
        
    def forward(self, x, tissue_context=None):
        # Main branch: x → net → fpn
        # Context branch: tissue_context → context_encoder
        # Fusion: head(fpn_features, context_features)
```

---

## 💡 데이터 증강 (Augmentation)

### 메인 이미지
- Mosaic, MixUp
- HSV color jittering
- Random perspective
- Horizontal/Vertical flip
- Albumentations (Blur, CLAHE, etc.)

### Context 이미지
- **Color augmentation 제외** (tissue context의 원본 정보 보존)
- Flip은 메인 이미지와 동일하게 적용
- Resize만 적용

---

## 📈 성능 향상 원리

1. **전역 조직 정보 활용**
   - Tissue context는 전체 조직의 구조, 패턴 정보 포함
   - Detection 시 local patch만으로 판단하기 어려운 경우 도움

2. **Late Fusion의 장점**
   - Bounding box detection은 그대로 유지
   - Classification만 context 정보로 보강
   - Context 정보가 없어도 작동 (optional input)

3. **Attention-like Mechanism**
   - Context features가 class features를 modulate
   - 조직 타입에 따라 특정 클래스의 신뢰도 조절

---

## 🔧 커스터마이징

### Context Encoder 수정

더 강력한 context encoder가 필요한 경우:

```python
class TissueContextEncoder(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        # 더 깊은 네트워크 사용
        # Vision Transformer 사용
        # Pretrained backbone 사용 (ResNet, EfficientNet 등)
```

### Fusion 방식 변경

다른 fusion 전략 적용:

```python
# 현재: Multiplicative fusion
cls_feat = cls_feat * (1 + context_weight)

# 대안 1: Additive fusion
cls_feat = cls_feat + context_weight

# 대안 2: Concatenation fusion
cls_feat = torch.cat([cls_feat, context_weight.expand_as(cls_feat)], dim=1)

# 대안 3: Cross-attention
cls_feat = cross_attention(cls_feat, context_features)
```

---

## 📝 예시 실행

```bash
# 데모 실행 (모델 구조 확인 및 테스트)
python example_context_usage.py

# 학습 실행
python train_with_context.py \
    --data-dir ./dataset \
    --batch-size 16 \
    --epochs 100 \
    --input-size 640
```

---

## ⚠️ 주의사항

1. **메모리 사용량**: Context branch 추가로 메모리 사용량 증가
   - Batch size를 적절히 조절하세요

2. **Context 이미지 품질**: 
   - Context 이미지가 메인 이미지와 다른 해상도/배율이어도 OK
   - 자동으로 리사이즈됩니다

3. **학습 시간**: 
   - Context encoder 추가로 학습 시간 약간 증가
   - 전체 파라미터의 ~10-15% 추가

4. **Context 없이도 추론 가능**:
   ```python
   # Context 없이 추론
   output = model(main_image, tissue_context=None)
   ```

---

## 🎯 적용 시나리오

이 구조는 다음과 같은 경우에 특히 유용합니다:

1. **조직병리 이미지 분석**
   - 세포 유형이 주변 조직 구조에 따라 달라지는 경우
   - 예: 종양 미세환경, 면역세포 분포

2. **공간 전사체학 (Spatial Transcriptomics)**
   - 조직 context가 세포 타입 결정에 중요한 역할

3. **멀티스케일 분석**
   - High-resolution patch + Low-resolution context

---

## 📚 참고사항

### 관련 논문 개념
- Late Fusion for Multimodal Learning
- Context-Aware Object Detection
- Attention Mechanisms in Computer Vision

### 코드 베이스
- YOLOv11 PyTorch implementation
- Custom modifications for biomedical imaging

---

## 🤝 기여 및 수정

코드를 프로젝트에 맞게 자유롭게 수정하세요:

- Context encoder 아키텍처 변경
- Fusion 전략 실험
- 데이터 증강 정책 조정
- Loss function 커스터마이징

---

## 📞 문의

구조 개선 제안이나 버그 발견 시 issue를 등록해주세요.

**Happy Training! 🚀**

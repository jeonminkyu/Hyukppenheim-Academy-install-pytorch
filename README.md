# 🔥 PyTorch 학습 노트 — 혁펜하임 아카데미

**혁펜하임 아카데미 - Install PyTorch** 강의를 수강하며 실습한 코드 모음입니다.  
PyTorch 기초부터 CNN, Transfer Learning, 실무 데이터셋 적용까지 단계적으로 학습한 내용을 담고 있습니다.

---

## 📁 파일 목록 및 내용

### 🟦 기초

| 파일 | 내용 |
|------|------|
| `torch_basic.ipynb` | 텐서(Tensor) 개념, 생성/연산/인덱싱/슬라이싱, reshape, broadcasting 등 PyTorch 기초 문법 |
| `mininet_test.ipynb` | `nn.Linear`, `nn.ReLU`, `nn.BatchNorm`, `nn.Dropout`, `nn.Conv2d` 등 주요 레이어 동작 확인용 테스트 코드 |

---

### 🟨 머신러닝 기초

| 파일 | 내용 |
|------|------|
| `Linear_regression.ipynb` | 키-몸무게 예측 선형회귀. 수동 파라미터 탐색 → autograd 직접 구현 → `torch.optim` 활용으로 단계적 학습 |
| `binary_classification.ipynb` | 이진분류(BCE Loss, Sigmoid). MLP 모델 구조 실험, 하이퍼파라미터 튜닝, 선형/비선형 분류경계 비교 |

---

### 🟩 다중분류 (MNIST)

| 파일 | 내용 |
|------|------|
| `multiclass_classification.ipynb` | MNIST 다중분류 기초. torchvision datasets/DataLoader 사용법, Softmax + CrossEntropyLoss |
| `multiclass_classification_short.ipynb` | 커스텀 모듈(`multiclass_functions1`) 사용하여 MLP/CNN 모델 학습·평가·Confusion Matrix 시각화 |
| `multiclass_classification_short_shuffle.ipynb` | 픽셀 셔플 실험 — MLP는 픽셀 순서에 무관하지만 CNN은 공간 정보에 의존함을 실험으로 확인 |

---

### 🟧 CNN 심화 (CIFAR-10 / STL-10)

| 파일 | 내용 |
|------|------|
| `multiclass_classification_CIFAR10.ipynb` | CIFAR-10에 MLP / CNN / CNN_deep 모델 비교 학습 |
| `multiclass_classification_CIFAR10_aug.ipynb` | Data Augmentation 실습 — `torchvision.transforms` 및 `Albumentations` 라이브러리 사용, Segmentation/Detection에서의 차이 설명 |
| `multiclass_classification_STL10.ipynb` | STL-10(96×96 고해상도) 데이터셋 적용. `StepLR` 학습률 스케줄링, train/val 분리 학습 |
| `CNN_Feature_map.ipynb` | CNN 중간 레이어의 Feature Map 직접 시각화 |

---

### 🟥 전이학습 & 실무 데이터

| 파일 | 내용 |
|------|------|
| `VGGnet_test.ipynb` | VGG19-BN 사전학습 모델로 Transfer Learning 실습. ImageNet 클래스 레이블 매핑, `torchinfo` 모델 요약 |
| `Custom_Dataset.ipynb` | `torch.utils.data.Dataset` 상속하여 커스텀 데이터셋 직접 구현 (텍스트: 한국어-영어 번역 데이터 / 이미지: 임의 데이터) |
| `Classification_COVID.ipynb` | Kaggle COVID-19 흉부 X-ray 데이터셋으로 실제 의료 이미지 분류 실습. Grayscale 전처리, Augmentation, LR 스케줄링 |

---

### 🛠️ 유틸리티 모듈

학습이 진행되며 반복 코드를 모듈화한 함수 모음입니다.

| 파일 | 주요 내용 |
|------|-----------|
| `multiclass_functions1.py` | 기본 `Train` / `Test` / `Test_plot` / `get_conf` / `plot_confusion_matrix` 함수. 검증 없이 학습 loss만 추적 |
| `multiclass_functions2.py` | `val_DL` 추가, `acc_history` 추적, best model 자동 저장(`torch.save`), `recall/precision/f1` 계산 함수 추가 |
| `multiclass_functions3.py` | `multiclass_functions2` 기능 + **TensorBoard** 연동 (`SummaryWriter`로 loss/acc 실시간 기록) |

---

## 🧪 주요 실험 기록

- **픽셀 셔플 실험** (`multiclass_classification_short_shuffle.ipynb`)
  - MLP: 섞음 96.0% / 안섞음 96.8% → 거의 차이 없음 (공간 정보 미사용)
  - CNN: 섞음 94.5% / 안섞음 99.0% → CNN은 공간 구조에 크게 의존함을 실험으로 확인

---

## 🛠️ 실행 환경

```bash
pip install torch torchvision tqdm matplotlib numpy pandas albumentations torchinfo tensorboard gdown
```

> Google Colab 또는 로컬 환경 모두 지원하며, 각 노트북 상단의 주석 처리된 코드로 환경을 전환할 수 있습니다.

---

## 📌 참고

- [혁펜하임 아카데미](https://www.youtube.com/@hyukppen)
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [Albumentations 공식 문서](https://albumentations.ai/docs/)
- COVID-19 데이터: [Kaggle - COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

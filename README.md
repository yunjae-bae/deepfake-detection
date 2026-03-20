# Deepfake Detection — EfficientNet-B3

Celeb-DF-v2 데이터셋을 활용한 얼굴 검출 기반 딥페이크 탐지 모델

## 파이프라인

**전처리** → MTCNN으로 비디오 프레임에서 얼굴 영역을 검출·크롭해 이미지 데이터셋 구축

**학습** → EfficientNet-B3 transfer learning, augmentation, early stopping

## 핵심 설계 결정

- **30프레임마다 1장 샘플링**: 인접 프레임 간 중복 정보 제거 + 처리 시간 단축
- **crop margin 30%**: 턱·이마 등 중요한 facial cue가 잘리지 않도록
- **JPEG compression augmentation**: 딥페이크 영상은 압축 시 고유한 아티팩트가 생김. 이를 학습에 반영해 모델 일반화

## 데이터셋

[Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics)

## 기술 스택

Python, PyTorch, facenet-pytorch, albumentations, OpenCV

## 설치
```bash
pip install -r requirements.txt
```

> Python 3.10 또는 3.11 권장 (3.12 이상에서는 일부 패키지 호환성 문제가 발생할 수 있음)

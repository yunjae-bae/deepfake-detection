# Deepfake Detection — EfficientNet-B3

Face-based classification using Celeb-DF-v2 videos.

## Pipeline

1. Sample video frames and extract face crops with MTCNN.
2. Fine-tune EfficientNet-B3 with augmentation and early stopping.

## Dataset

[Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics)

## Installation

```bash
pip install -r requirements.txt
```

The original environment targeted Python 3.10 or 3.11.

# Deepfake Detection

Celeb-DF-v2 video classification using MTCNN face crops and an ImageNet-pretrained EfficientNet-B3. Frames from each source video stay in the same training or validation split.

## Usage

Use Python 3.10 or 3.11:

```sh
pip install -r requirements.txt
```

Place Celeb-DF-v2 under `train_data/Celeb-DF-v2/` and run `deepfake_detection.ipynb` from top to bottom. Each class needs at least two videos with usable faces.

Preprocessing saves crops under `train_data/cropped_faces/Celeb-DF-v2/`, grouped by source video. Regenerate crops if using the old flat directory layout. Training saves the best checkpoint to `best_model.pth`.

Run offline preprocessing checks with `python test_preprocessing.py`.

## Sources

- [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics): Yuezun Li, Xin Yang, Pu Sun, Honggang Qi and Siwei Lyu. *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*. CVPR, 2020.

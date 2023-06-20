# Vesuvius Challenge - Ink Detection

This repository contains the code used to achieve the 34th place in the Vesuvius Challenge - Ink Detection competition, hosted on Kaggle. You can visit the competition page [here](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/).

## Data Acquisition

This section provides information about how to download the necessary datasets from Kaggle.

### Vesuvius Challenge Ink Detection Dataset

This dataset is essential for the main task. It can be downloaded from the following Kaggle competition page:

- [Vesuvius Challenge Ink Detection](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data)

### For Inference
The trained weights for the inference model can be downloaded from:

- [VCID Trained Weight](https://www.kaggle.com/datasets/tmyok1984/vcid-trained-weight)

### For Training
The datasets for training can be downloaded from:

- [VCID validation](https://www.kaggle.com/datasets/tmyok1984/vcid-validation)
- [VCID modified inklabels](https://www.kaggle.com/datasets/tmyok1984/vcid-modify-inklabels)
- [VCID tile iamges 256](https://www.kaggle.com/datasets/tmyok1984/vcid-tile-images-inklabels-256-64)
- [VCID tile iamges 1024](https://www.kaggle.com/datasets/tmyok1984/vcid-tile-images-inklabels-1024-256)

## Usage

### Preparation

1. Clone this repository:
```
git clone --recursive https://github.com/tmyok/kaggle-vesuvius-challenge-ink-detection.git
```

2. Download the datasets into the `input` directory. Refer to [Data Acquisition](#data-acquisition) for more information.

3. If necessary, run a Docker container with the following command:
```
sh docker_container.sh
```

### Training

Ensure that your hardware has at least a total of 40 GB of GPU RAM and then run the following command:

```
python3 train.py model={unet3d, residualunetse3d, mit_b2} fold={0,1,2,3,4}
```

If your hardware has less GPU RAM, adjust the resize_ratio in the configuration file (located at `./working/configs/model/{UNet3D, ResidualUNetSE3D, mit_b2}.yaml`).

The trained weights will be exported to `./output/{UNet3D, ResidualUNetSE3D, mit_b2}`.

### Validation

To perform validation, run the following command:
```
python3 validation.py fold={0,1,2,3,4}
```

The results will be exported to `./output/validation`.

### Inference

To perform inference, run the following command:

```
python3 inference.py --image_path ../input/vesuvius-challenge-ink-detection/test/a/surface_volume/ --mask_path ../input/vesuvius-challenge-ink-detection//test/a/mask.png
```

The results will be exported to `./output/inference`.
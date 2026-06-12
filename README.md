# nuwe.io Image Classification

8-class image classifier built for the [Oracle Spain challenge on nuwe.io](https://nuwe.io/dev/competitions/reto-ensena-oracle-espana/clasificacion-imagenes-reto_1). Uses Ultralytics YOLO for classification with test-time augmentation (TTA) for improved accuracy.

## Results

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | ~78.5% (single model, conservative augmentation) |
| Top-5 Accuracy | ~98.4% |
| Model | YOLOv8x-cls |
| Image Size | 640px |
| Classes | 8 (integer labels 0-7) |

## Architecture

```
preprocessing/          # Dataset preparation
  distribute_dataset.py # Split CSV -> class-labeled directories

nn/                     # Neural network training & inference
  nn_train.py           # Train YOLO classifier with augmentation
  nn_predict.py         # Standard single-pass prediction
  predict_new.py        # TTA prediction (hflip + multi-scale)
  nn_validate.py        # Model validation & metrics

postprocessing/         # Output formatting
  reshape_solution.py   # Map predictions to submission format
```

## Setup

<!-- one-command-install -->
> **One-command install**: clone, configure, and run in a single step:
>
> ```bash
> curl -fsSL https://raw.githubusercontent.com/jasperan/nuwe.io-image-classification/main/install.sh | bash
> ```
>
> <details><summary>Advanced options</summary>
>
> Override install location:
> ```bash
> PROJECT_DIR=/opt/myapp curl -fsSL https://raw.githubusercontent.com/jasperan/nuwe.io-image-classification/main/install.sh | bash
> ```
>
> Or install manually:
> ```bash
> git clone https://github.com/jasperan/nuwe.io-image-classification.git
> cd nuwe.io-image-classification
> # See below for setup instructions
> ```
> </details>


```bash
# Install the project and its dependencies (editable)
pip install -e .

# Or just the runtime dependencies
pip install ultralytics pandas torch numpy Pillow
```

Installing the project makes `nn`, `preprocessing`, and `postprocessing`
importable, so each stage runs as a module with `python -m`.

## Usage

### 1. Prepare dataset

```bash
python -m preprocessing.distribute_dataset \
  --data /path/to/raw/dataset \
  --output /path/to/organized/dataset
```

### 2. Train

```bash
python -m nn.nn_train \
  --data /path/to/organized/dataset \
  --epochs 300 \
  --imgsz 640
```

### 3. Predict (with TTA for best accuracy)

```bash
# Standard prediction
python -m nn.nn_predict \
  --model best.pt \
  --data /path/to/dataset

# TTA prediction (2-5% accuracy boost)
python -m nn.predict_new \
  --model best.pt \
  --data /path/to/dataset \
  --output predictions_tta.json
```

### 4. Format for submission

```bash
python -m postprocessing.reshape_solution \
  --predictions raw_predictions.json \
  --test /path/to/test.csv \
  --output predictions.json
```

## Training Configuration

Augmentation settings (tuned for this dataset):

| Parameter | Value | Purpose |
|-----------|-------|---------|
| hsv_h | 0.015 | Hue variation |
| hsv_s | 0.7 | Saturation variation |
| hsv_v | 0.4 | Value/brightness variation |
| degrees | 10.0 | Rotation range |
| translate | 0.1 | Translation range |
| scale | 0.5 | Scale variation |
| shear | 2.0 | Shear range |
| fliplr | 0.5 | Horizontal flip probability |
| mosaic | 0.5 | Mosaic augmentation |
| mixup | 0.15 | MixUp augmentation |

## License

GPL-3.0. See [LICENSE](LICENSE) for details.

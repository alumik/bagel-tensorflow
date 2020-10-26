# Bagel

![version-1.2.3](https://img.shields.io/badge/version-1.2.3-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/AlumiK/bagel-tensorflow/blob/main/LICENSE)

Bagel is a robust and unsupervised KPI anomaly detection algorithm based on conditional variational autoencoder.

This is an implementation of Bagel in TensorFlow 2. The original PyTorch 0.4 implementation can be found here: [NetManAIOps/Bagel](https://github.com/NetManAIOps/Bagel).

## Dependencies

- Python >=3.6, <3.9
- TensorFlow 2
- CUDA Toolkit 10.1
- cuDNN

Normally, `pip` will automatically install required PyPI dependencies when you install this package:
 
```
pip install -e .
``` 

An `environment.yml` is also provided if you want to use `conda` to manage dependencies:

```
conda env create -f environment.yml
```

### Note

1. `sample/plot_kpi.py` requires `matplotlib`. You can manually install it by `pip install matplotlib`.
2. On Windows, TensorFlow 2 requires [the latest VC runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).

## Run

A sample script can be found at `sample/main.py`:

```
cd sample
python main.py
```

KPI data must be stored in csv files in the following format:

```
timestamp,  value,        label
1469376000,  0.847300274, 0
1469376300, -0.036137314, 0
1469376600,  0.074292384, 0
1469376900,  0.074292384, 0
1469377200, -0.036137314, 0
1469377500,  0.184722083, 0
1469377800, -0.036137314, 0
1469378100,  0.184722083, 0
```

- `timestamp`: timestamps in seconds (10-digit).
- `label`: `0` for normal points, `1` for anomaly points.

## Usage

To prepare the data:

```python
import bagel

kpi = bagel.utils.load_kpi(file_path)
kpi.complete_timestamp()
train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))
train_kpi, mean, std = train_kpi.standardize()
valid_kpi, _, _ = valid_kpi.standardize(mean=mean, std=std)
test_kpi, _, _ = test_kpi.standardize(mean=mean, std=std)
```

To construct a Donut model, train the model, and use the trained model for prediction:

```python
import bagel

model = bagel.models.Bagel()
model.fit(kpi=train_kpi.no_labels(), validation_kpi=valid_kpi, epochs=EPOCHS)
anomaly_scores = model.predict(test_kpi)
```

To save and restore a trained model:

```python
# Save a trained model
model.save(path)

# Load a pre-trained model
import bagel
model = bagel.models.Bagel()
model.load(path)
```

## Citation

```bibtex
@inproceedings{conf/ipccc/LiCP18,
    author    = {Zeyan Li and
                 Wenxiao Chen and
                 Dan Pei},
    title     = {Robust and Unsupervised {KPI} Anomaly Detection Based on Conditional
                 Variational Autoencoder},
    booktitle = {37th {IEEE} International Performance Computing and Communications
                 Conference, {IPCCC} 2018, Orlando, FL, USA, November 17-19, 2018},
    pages     = {1--9},
    publisher = {{IEEE}},
    year      = {2018},
    url       = {https://doi.org/10.1109/PCCC.2018.8710885},
    doi       = {10.1109/PCCC.2018.8710885}
}
```

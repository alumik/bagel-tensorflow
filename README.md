# Bagel

![python-3.8](https://img.shields.io/badge/python-3.8-blue)
![version-1.6.1](https://img.shields.io/badge/version-1.6.1-blue)
[![license-MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/AlumiK/bagel-tensorflow/blob/main/LICENSE)
[![open-in-colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/AlumiK/bagel-tensorflow/blob/main/notebooks/bagel_playground.ipynb)

<img width="140" alt="Bagel Logo" align="right" src="https://www.svgrepo.com/show/275681/bagel.svg"/>

Bagel is a robust and unsupervised KPI anomaly detection algorithm based on conditional variational autoencoder.

This is an implementation of Bagel in TensorFlow 2. The original PyTorch 0.4 implementation can be found at
[NetManAIOps/Bagel](https://github.com/NetManAIOps/Bagel).

## Install

### Dependencies

Normally, `pip` will automatically install required PyPI dependencies when you install this package:

- For development use:

    ```
    git clone https://github.com/AlumiK/bagel-tensorflow.git
    cd bagel-tensorflow
    pip install -e .[dev]
    ```

- For production use:

    ```
    pip install git+https://github.com/AlumiK/bagel-tensorflow.git
    ```

An `environment.yml` is also provided if you prefer `conda` to manage dependencies:

```
conda env create -f environment.yml
```

### Note

- TensorFlow >= 2.4 is required.
- TensorFlow version is tightly coupled to CUDA and cuDNN so they should be selected carefully.
- On Windows, TensorFlow 2 requires
  [the latest VC runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).

## Run

### KPI Format

KPI data must be stored in csv files in the following format:

```
timestamp,   value,       label
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
- Labels are used for evaluation and are not required in the production environment.

### Sample Script

A sample script can be found at `sample/main.py`:

```
cd sample
python main.py
```

### Jupyter Notebook

To help visualize Bagel and to help you get started, we provide a Jupyter notebook found
in `notebooks/bagel_playground.ipynb` that allows one to visualize Bagel in real time.

## Usage

To prepare the data:

```python
import bagel

kpi = bagel.utils.load_kpi('kpi_data.csv')
kpi.complete_timestamp()
train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))
train_kpi, mean, std = train_kpi.standardize()
valid_kpi, _, _ = valid_kpi.standardize(mean=mean, std=std)
test_kpi, _, _ = test_kpi.standardize(mean=mean, std=std)
```

To construct a Bagel model, train the model, and use the trained model for prediction:

```python
import bagel

model = bagel.Bagel()
model.fit(kpi=train_kpi.use_labels(0.), validation_kpi=valid_kpi, epochs=epochs)
anomaly_scores = model.predict(test_kpi)
```

To save and restore a trained model:

```python
# To save a trained model
model.save(prefix)

# To load a pre-trained model
import bagel

model = bagel.Bagel()
model.load(prefix)
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

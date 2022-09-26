# Bagel

![version-2.1.0](https://img.shields.io/badge/version-2.1.0-blue)
![python-3.10](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-FF6F00?logo=tensorflow&logoColor=white)
[![license-MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/alumik/bagel-tensorflow/blob/main/LICENSE)

<img width="140" alt="Bagel Logo" align="right" src="https://www.svgrepo.com/show/275681/bagel.svg"/>

Bagel is a robust and unsupervised KPI anomaly detection algorithm based on conditional variational autoencoder.

This is an implementation of Bagel in TensorFlow 2. The original PyTorch 0.4 implementation can be found at
[NetManAIOps/Bagel](https://github.com/NetManAIOps/Bagel).

## Install

`pip` will automatically install required PyPI dependencies when you install this package:

- For development use:

    ```
    git clone https://github.com/alumik/bagel-tensorflow.git
    cd bagel-tensorflow
    pip install -e .
    ```

- For production use:

    ```
    pip install git+https://github.com/alumik/bagel-tensorflow.git
    ```

An `environment.yml` is also provided if you prefer `conda` to manage dependencies:

```
conda env create -f environment.yml
```

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
- `label` (optional): `0` for normal points, `1` for anomaly points.
- Labels are used only for evaluation and are not required in model training and inference.

### Sample Script

A sample script can be found at `sample/main.py`:

```
cd sample
python main.py
```

## Usage

To prepare the data:

```python
import bagel

kpi = bagel.data.load_kpi('kpi_data.csv')
kpi.complete_timestamp()
train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))
train_kpi, mean, std = train_kpi.standardize()
valid_kpi, _, _ = valid_kpi.standardize(mean=mean, std=std)
test_kpi, _, _ = test_kpi.standardize(mean=mean, std=std)

dataset = bagel.data.KPIDataset(
    train_kpi.use_labels(0.),
    window_size=window_size,
    time_feature=time_feature,
    missing_injection_rate=missing_injection_rate,
)
valid_dataset = bagel.data.KPIDataset(
    valid_kpi,
    window_size=window_size,
    time_feature=time_feature,
)
test_dataset = bagel.data.KPIDataset(
    test_kpi.no_labels(),
    window_size=window_size,
    time_feature=time_feature,
)
```

To build and train a Bagel model:

```python
model = bagel.Bagel(
    window_size=window_size,
    hidden_dims=hidden_dims,
    latent_dim=latent_dim,
    dropout_rate=dropout_rate,
)
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=10 * len(dataset) // batch_size,
    decay_rate=0.75,
    staircase=True,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler, clipnorm=clipnorm)
model.compile(optimizer=optimizer, jit_compile=True)
model.fit(
    x=[dataset.values, dataset.time_code, dataset.normal],
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([valid_dataset.values, valid_dataset.time_code, valid_dataset.normal], None),
    validation_batch_size=batch_size,
)
```

To use the trained model for prediction:

```python
anomaly_scores = model.predict(
    x=[test_dataset.values, test_dataset.time_code, test_dataset.normal],
    batch_size=batch_size,
)
```

Use `tf.keras.Model.save` API to save the model.

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

import bagel
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Sequence, Optional


class AutoencoderLayer(tf.keras.layers.Layer):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence):
        super().__init__()
        self._hidden = tf.keras.Sequential([
            tf.keras.Input(shape=(input_dim,)),
        ])
        for hidden_dim in hidden_dims:
            self._hidden.add(tf.keras.layers.Dense(hidden_dim, activation='relu'))
        self._mean = tf.keras.layers.Dense(output_dim, activation='relu')
        self._std = tf.keras.layers.Dense(output_dim, activation='relu')

    def call(self, inputs, **kwargs):
        x = self._hidden(inputs)
        output_mean = self._mean(x)
        output_std = tf.math.softplus(self._std(x)) + 1e-6

        loss = tf.Variable(0.)
        for weight in self._hidden.weights:
            loss.assign(loss + tf.math.reduce_sum(tf.math.square(weight)))
        self.add_loss(loss)

        return output_mean, output_std


class ConditionalVariationalAutoencoder(tf.keras.Model):

    def __init__(self, encoder: AutoencoderLayer, decoder: AutoencoderLayer):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def call(self, inputs, n_samples: int = 1, **kwargs):
        x, y = inputs[0], inputs[1]
        cat = tf.keras.layers.Concatenate()([x, y])
        z_mean, z_std = self._encoder(cat)
        q_zx = tfp.distributions.Normal(z_mean, z_std)

        p_z = tfp.distributions.Normal(
            tf.zeros(z_mean.shape, dtype='float32'),
            tf.ones(z_std.shape, dtype='float32')
        )
        epsilon = p_z.sample((n_samples,))
        z = epsilon * tf.expand_dims(z_std, 0) + tf.expand_dims(z_mean, 0)

        y = tf.broadcast_to(y, [n_samples, y.shape[0], y.shape[1]])
        cat = tf.keras.layers.Concatenate()([z, y])
        x_mean, x_std = self._decoder(cat)
        p_xz = tfp.distributions.Normal(x_mean, x_std)
        return q_zx, p_xz, z

    def get_config(self):
        raise NotImplementedError


class Bagel:

    def __init__(self,
                 window_size: int = 120,
                 hidden_dims: Optional[Sequence] = None,
                 latent_dim: int = 8,
                 dropout_rate: float = 0.1):
        super().__init__()
        self._hidden_dims = [100, 100] if hidden_dims is None else hidden_dims
        self._latent_dim = latent_dim
        self._window_size = window_size
        self._cond_size = 60 + 24 + 7
        self._dropout_rate = dropout_rate
        self._model = ConditionalVariationalAutoencoder(
            encoder=AutoencoderLayer(
                input_dim=self._window_size + self._cond_size,
                output_dim=self._latent_dim,
                hidden_dims=self._hidden_dims
            ),
            decoder=AutoencoderLayer(
                input_dim=self._latent_dim + self._cond_size,
                output_dim=self._window_size,
                hidden_dims=self._hidden_dims
            ),
        )
        self._p_z = tfp.distributions.Normal(
            tf.zeros(self._latent_dim, dtype='float32'),
            tf.ones(self._latent_dim, dtype='float32')
        )

    @staticmethod
    def _m_elbo(x: tf.Tensor,
                z: tf.Tensor,
                normal: tf.Tensor,
                p_xz: tfp.distributions.Normal,
                q_zx: tfp.distributions.Normal,
                p_z: tfp.distributions.Normal) -> tf.Tensor:
        x = tf.expand_dims(x, 0)
        normal = tf.expand_dims(normal, 0)
        log_p_xz = p_xz.log_prob(x)
        log_q_zx = tf.math.reduce_sum(q_zx.log_prob(z), axis=-1)
        log_p_z = tf.math.reduce_sum(p_z.log_prob(z), axis=-1)
        ratio = (tf.math.reduce_sum(normal, axis=-1) / float(normal.shape[-1]))
        return -tf.math.reduce_mean(tf.math.reduce_sum(log_p_xz * normal, axis=-1) + log_p_z * ratio - log_q_zx)

    def _mcmc_missing_imputation(self, x: tf.Tensor, y: tf.Tensor, normal: tf.Tensor, max_iter: int = 10) -> tf.Tensor:
        for _ in range(max_iter):
            _, p_xz, _ = self._model([x, y])
            reconstruction = p_xz.sample()[0]
            x = tf.where(tf.cast(normal, 'bool'), x, reconstruction)
        return x

    def fit(self,
            kpi: bagel.data.KPI,
            epochs: int,
            validation_kpi: Optional[bagel.data.KPI] = None,
            batch_size: int = 128):
        dataset = bagel.data.KPIDataset(kpi, window_size=self._window_size, missing_injection_rate=0.01)
        dataset = dataset.to_tensorflow()
        dataset = dataset.shuffle(len(dataset)).batch(batch_size, drop_remainder=True)
        validation_dataset = None
        if validation_kpi is not None:
            validation_dataset = bagel.data.KPIDataset(validation_kpi, window_size=self._window_size)
            validation_dataset = validation_dataset.to_tensorflow()
            validation_dataset = validation_dataset.shuffle(len(validation_dataset)).batch(batch_size)

        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=len(dataset) * 10,
            decay_rate=0.75,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler, clipnorm=10.)
        for epoch in range(epochs):
            print(f'Training Epoch: {epoch + 1}/{epochs}')
            progbar = tf.keras.utils.Progbar(len(dataset) + (0 if validation_kpi is None else len(validation_dataset)),
                                             interval=0.5)

            for batch in dataset:
                y, x, normal = batch

                with tf.GradientTape() as tape:
                    y = tf.keras.layers.Dropout(self._dropout_rate)(y)
                    q_zx, p_xz, z = self._model([x, y])
                    loss = self._m_elbo(x=x, z=z, normal=normal, p_xz=p_xz, q_zx=q_zx, p_z=self._p_z)
                    loss += tf.math.reduce_sum(self._model.losses) * 0.001

                grads = tape.gradient(loss, self._model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
                progbar.add(1, values=[('loss', loss)])

            if validation_kpi is None:
                continue

            for batch in validation_dataset:
                y, x, normal = batch
                q_zx, p_xz, z = self._model([x, y])
                val_loss = self._m_elbo(x=x, z=z, normal=normal, p_xz=p_xz, q_zx=q_zx, p_z=self._p_z)
                val_loss += tf.math.reduce_sum(self._model.losses) * 0.001
                progbar.add(1, values=[('val_loss', val_loss)])

    def predict(self, kpi: bagel.data.KPI, batch_size: int = 256) -> np.ndarray:
        kpi = kpi.no_labels()
        dataset = bagel.data.KPIDataset(kpi, window_size=self._window_size)
        dataset = dataset.to_tensorflow()
        dataset = dataset.batch(batch_size)

        print('Testing Epoch')
        progbar = tf.keras.utils.Progbar(len(dataset), interval=0.5)

        anomaly_scores = []
        for batch in dataset:
            y, x, normal = batch
            x = self._mcmc_missing_imputation(x=x, y=y, normal=normal)
            q_zx, p_xz, z = self._model([x, y], n_samples=128)
            test_loss = self._m_elbo(x=x, z=z, normal=normal, p_xz=p_xz, q_zx=q_zx, p_z=self._p_z)
            log_p_xz = p_xz.log_prob(x).numpy()
            anomaly_scores.extend(-np.mean(log_p_xz[:, :, -1], axis=0))
            progbar.add(1, values=[('test_loss', test_loss)])

        anomaly_scores = np.asarray(anomaly_scores, dtype=np.float32)
        return np.concatenate([np.ones(self._window_size - 1) * np.min(anomaly_scores), anomaly_scores])

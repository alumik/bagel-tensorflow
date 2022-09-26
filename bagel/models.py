import tensorflow as tf
import tensorflow_probability as tfp

from typing import *


class AutoencoderLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        self._hidden = tf.keras.Sequential()
        for hidden_dim in self._hidden_dims:
            self._hidden.add(
                tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))
            )
        self._mean = tf.keras.layers.Dense(self._output_dim, kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self._std = tf.keras.layers.Dense(self._output_dim, kernel_regularizer=tf.keras.regularizers.L2(0.001))

    def call(self, inputs, **kwargs):
        x = self._hidden(inputs)
        mean = self._mean(x)
        std = tf.math.softplus(self._std(x)) + 1e-6
        return mean, std

    def get_config(self):
        config = {
            'hidden_dims': self._hidden_dims,
            'output_dim': self._output_dim,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class ConditionalVariationalAutoencoder(tf.keras.layers.Layer):

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 latent_dim: int,
                 dropout_rate: float):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._latent_dim = latent_dim
        self._dropout_rate = dropout_rate
        self._encoder = AutoencoderLayer(self._hidden_dims, self._latent_dim)
        self._decoder = AutoencoderLayer(list(reversed(self._hidden_dims)), self._input_dim)
        self._dropout = tf.keras.layers.Dropout(self._dropout_rate)

    def call(self, inputs, **kwargs):
        x, y = inputs
        n_samples = kwargs.get('n_samples', 1)
        y = self._dropout(y, training=kwargs.get('training', True))
        z_mean, z_std = self._encoder(tf.concat([x, y], axis=-1))
        q_zx = tfp.distributions.Normal(z_mean, z_std)
        p_z = tfp.distributions.Normal(tf.zeros_like(z_mean), tf.ones_like(z_std))
        z = p_z.sample((n_samples,)) * tf.expand_dims(z_std, 0) + tf.expand_dims(z_mean, 0)
        y = tf.tile(tf.expand_dims(y, 0), multiples=[n_samples, 1, 1])
        x_mean, x_std = self._decoder(tf.concat([z, y], axis=-1))
        p_xz = tfp.distributions.Normal(x_mean, x_std)
        return q_zx, p_xz, p_z, z

    def get_config(self):
        config = {
            'input_dim': self._input_dim,
            'hidden_dims': self._hidden_dims,
            'latent_dim': self._latent_dim,
            'dropout_rate': self._dropout_rate,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class Bagel(tf.keras.Model):

    def __init__(self,
                 window_size: int = 120,
                 hidden_dims: Sequence = (100, 100),
                 latent_dim: int = 8,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self._window_size = window_size
        self._hidden_dims = hidden_dims
        self._latent_dim = latent_dim
        self._dropout_rate = dropout_rate
        self._cvae = ConditionalVariationalAutoencoder(
            input_dim=self._window_size,
            hidden_dims=self._hidden_dims,
            latent_dim=self._latent_dim,
            dropout_rate=self._dropout_rate,
        )
        self._loss_tracker = tf.keras.metrics.Mean(name='loss')

    @staticmethod
    def _m_elbo(x: tf.Tensor,
                z: tf.Tensor,
                normal: tf.Tensor,
                q_zx: tfp.distributions.Normal,
                p_z: tfp.distributions.Normal,
                p_xz: tfp.distributions.Normal) -> tf.Tensor:
        x = tf.expand_dims(x, 0)
        normal = tf.expand_dims(normal, 0)
        log_p_xz = p_xz.log_prob(x)
        log_q_zx = tf.reduce_sum(q_zx.log_prob(z), axis=-1)
        log_p_z = tf.reduce_sum(p_z.log_prob(z), axis=-1)
        ratio = (tf.reduce_sum(normal, axis=-1) / float(normal.shape[-1]))
        return tf.reduce_mean(tf.reduce_sum(log_p_xz * normal, axis=-1) + log_p_z * ratio - log_q_zx)

    def _missing_imputation(self, x: tf.Tensor, y: tf.Tensor, normal: tf.Tensor, steps: int = 10) -> tf.Tensor:
        cond = tf.cast(normal, 'bool')
        for _ in range(steps):
            _, p_xz, _, _ = self._cvae([x, y], training=False)
            reconstruction = p_xz.sample()[0]
            x = tf.where(cond, x, reconstruction)
        return x

    def train_step(self, data):
        data, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        x, y, normal = data
        with tf.GradientTape() as tape:
            q_zx, p_xz, p_z, z = self._cvae([x, y])
            loss = -self._m_elbo(x, z, normal, q_zx, p_z, p_xz)
            loss += tf.add_n(self._cvae.losses)
        self.optimizer.minimize(loss, self._cvae.trainable_weights, tape=tape)
        self._loss_tracker.update_state(loss)
        return {'loss': self._loss_tracker.result()}

    def test_step(self, data):
        data, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        x, y, normal = data
        q_zx, p_xz, p_z, z = self._cvae([x, y], training=False)
        loss = -self._m_elbo(x, z, normal, q_zx, p_z, p_xz)
        loss += tf.add_n(self._cvae.losses)
        self._loss_tracker.update_state(loss)
        return {'loss': self._loss_tracker.result()}

    def call(self, inputs, **kwargs):
        x, y, normal = inputs
        x = self._missing_imputation(x, y, normal)
        _, p_xz, _, _ = self._cvae([x, y], n_samples=128, training=False)
        log_p_xz = p_xz.log_prob(x)
        return -tf.reduce_mean(log_p_xz[:, :, -1], axis=0)

    def predict(self, *args, **kwargs):
        result = super().predict(*args, **kwargs)
        result = tf.concat([tf.ones(self._window_size - 1) * tf.reduce_min(result), result], axis=0)
        return result

    @property
    def metrics(self):
        return [self._loss_tracker]

    def get_config(self):
        config = {
            'window_size': self._window_size,
            'hidden_dims': self._hidden_dims,
            'latent_dim': self._latent_dim,
            'dropout_rate': self._dropout_rate,
        }
        base_config = super().get_config()
        return {**base_config, **config}

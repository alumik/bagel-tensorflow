import tensorflow as tf
import tensorflow_probability as tfp

from typing import *


class AutoencoderLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        self._hidden = tf.keras.Sequential()
        for hidden_dim in hidden_dims:
            self._hidden.add(
                tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001))
            )
        self._mean = tf.keras.layers.Dense(output_dim, kernel_regularizer=tf.keras.regularizers.L2(0.001))
        self._std = tf.keras.layers.Dense(output_dim, kernel_regularizer=tf.keras.regularizers.L2(0.001))

    def call(self, inputs, **kwargs):
        x = self._hidden(inputs)
        mean = self._mean(x)
        std = tf.math.softplus(self._std(x)) + 1e-6
        return mean, std

    def get_config(self):
        config = {
            'hidden_dims': [layer.units for layer in self._hidden.layers],
            'output_dim': self._mean.units,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class ConditionalVariationalAutoencoder(tf.keras.layers.Layer):

    def __init__(self, encoder: AutoencoderLayer, decoder: AutoencoderLayer, dropout_rate: float = 0.1):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._dropout_rate = dropout_rate

    def call(self, inputs, **kwargs):
        x, y = tuple(inputs)
        y = tf.keras.layers.Dropout(self._dropout_rate)(y, training=kwargs.get('training', True))
        n_samples = kwargs.get('n_samples', 1)
        x_y = tf.keras.layers.Concatenate()([x, y])
        z_mean, z_std = self._encoder(x_y)
        q_zx = tfp.distributions.Normal(z_mean, z_std)
        p_z = tfp.distributions.Normal(tf.zeros_like(z_mean), tf.ones_like(z_std))
        z = p_z.sample((n_samples,)) * tf.expand_dims(z_std, 0) + tf.expand_dims(z_mean, 0)
        y = tf.broadcast_to(y, [n_samples, tf.shape(y)[0], tf.shape(y)[1]])
        z_y = tf.keras.layers.Concatenate()([z, y])
        x_mean, x_std = self._decoder(z_y)
        p_xz = tfp.distributions.Normal(x_mean, x_std)
        return q_zx, p_xz, z


class Bagel(tf.keras.Model):

    def __init__(self,
                 window_size: int = 120,
                 hidden_dims: Sequence = (100, 100),
                 latent_dim: int = 8,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self._window_size = window_size
        self._cvae = ConditionalVariationalAutoencoder(
            encoder=AutoencoderLayer(hidden_dims, latent_dim),
            decoder=AutoencoderLayer(list(reversed(hidden_dims)), self._window_size),
            dropout_rate=dropout_rate,
        )
        self._p_z = tfp.distributions.Normal(tf.zeros(latent_dim), tf.ones(latent_dim))
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
        log_q_zx = tf.math.reduce_sum(q_zx.log_prob(z), axis=-1)
        log_p_z = tf.math.reduce_sum(p_z.log_prob(z), axis=-1)
        ratio = (tf.math.reduce_sum(normal, axis=-1) / float(normal.shape[-1]))
        return tf.math.reduce_mean(tf.math.reduce_sum(log_p_xz * normal, axis=-1) + log_p_z * ratio - log_q_zx)

    def _missing_imputation(self, x: tf.Tensor, y: tf.Tensor, normal: tf.Tensor, steps: int = 10) -> tf.Tensor:
        cond = tf.cast(normal, 'bool')
        for _ in range(steps):
            _, p_xz, _ = self._cvae([x, y], training=False)
            reconstruction = p_xz.sample()[0]
            x = tf.where(cond, x, reconstruction)
        return x

    def train_step(self, data):
        data, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        x, y, normal = data
        with tf.GradientTape() as tape:
            q_zx, p_xz, z = self._cvae([x, y])
            loss = -self._m_elbo(x, z, normal, q_zx, self._p_z, p_xz)
            loss += tf.math.add_n(self._cvae.losses)
        self.optimizer.minimize(loss, self._cvae.trainable_weights, tape=tape)
        self._loss_tracker.update_state(loss)
        return {'loss': self._loss_tracker.result()}

    def test_step(self, data):
        data, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        x, y, normal = data
        q_zx, p_xz, z = self._cvae([x, y], training=False)
        loss = -self._m_elbo(x, z, normal, q_zx, self._p_z, p_xz)
        loss += tf.math.add_n(self._cvae.losses)
        self._loss_tracker.update_state(loss)
        return {'loss': self._loss_tracker.result()}

    def call(self, inputs, **kwargs):
        x, y, normal = inputs
        x = self._missing_imputation(x, y, normal)
        q_zx, p_xz, z = self._cvae([x, y], n_samples=128, training=False)
        log_p_xz = p_xz.log_prob(x)
        return -tf.math.reduce_mean(log_p_xz[:, :, -1], axis=0)

    def predict(self, *args, **kwargs):
        result = super().predict(*args, **kwargs)
        result = tf.concat([tf.ones(self._window_size - 1) * tf.math.reduce_min(result), result], axis=0)
        return result

    @property
    def metrics(self):
        return [self._loss_tracker]

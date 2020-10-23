import bagel
import unittest
import numpy as np


class ModelsTest(unittest.TestCase):

    def test_cvae(self):
        model = bagel.models.ConditionalVariationalAutoencoder(
            encoder=bagel.models.AutoencoderLayer(
                input_dim=4,
                output_dim=2,
                hidden_dims=[5, 5]
            ),
            decoder=bagel.models.AutoencoderLayer(
                input_dim=4,
                output_dim=2,
                hidden_dims=[5, 5]
            ),
        )

        x = np.asarray([[1., 2.], [3., 4.]], dtype=np.float32)
        y = np.asarray([[6., 7.], [8., 9.]], dtype=np.float32)
        q_zx, p_xz, z = model(inputs=[x, y])
        self.assertEqual(2, len(model.losses))
        self.assertEqual((1, 2, 2), tuple(z.shape))

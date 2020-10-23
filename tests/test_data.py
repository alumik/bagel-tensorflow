import bagel
import unittest
import numpy as np


class DataTest(unittest.TestCase):

    @staticmethod
    def test_load_kpi():
        kpi = bagel.utils.load_kpi('data/test1.csv')
        np.testing.assert_allclose(
            [1., 2., 3., 4., 6., 7., 8., 9., 11., 12., 13., 14., 16., 17., 18., 19., 20.],
            kpi.values
        )
        kpi.complete_timestamp()
        np.testing.assert_allclose(
            [1., 2., 3., 4., 0., 6., 7., 8., 9., 0., 11., 12., 13., 14., 0., 16., 17., 18., 19., 20.],
            kpi.values
        )

    def test_load_dataset(self):
        kpi = bagel.utils.load_kpi('data/test2.csv')
        kpi.complete_timestamp()
        dataset = bagel.data.KPIDataset(kpi, window_size=2)
        dataset = dataset.to_tensorflow().batch(2)
        self.assertEqual(2, len(dataset))

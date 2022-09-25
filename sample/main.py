import json
import pathlib
import tensorflow as tf

import bagel

from typing import *


def main(batch_size: int = 256,
         epochs: int = 50,
         learning_rate: float = 1e-3,
         window_size: int = 120,
         time_feature: Optional[str] = 'MHw',
         hidden_dims: Sequence = (100, 100),
         latent_dim: int = 8,
         dropout_rate: float = 0.1,
         clipnorm: float = 10.0,
         missing_injection_rate: float = 0.01):
    bagel.utils.mkdirs(output_path)
    file_list = bagel.utils.file_list(input_path)

    for file in file_list:
        kpi = bagel.data.load_kpi(file)
        print(f'KPI: {kpi.name}')
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
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler, clipnorm=clipnorm)
        model.compile(optimizer=optimizer)

        print('Training...')
        model.fit(
            x=[dataset.values, dataset.time_code, dataset.normal],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([valid_dataset.values, valid_dataset.time_code, valid_dataset.normal], None),
            validation_batch_size=batch_size,
        )

        print('Predicting...')
        anomaly_scores = model.predict(
            x=[test_dataset.values, test_dataset.time_code, test_dataset.normal],
            batch_size=batch_size,
        )

        print('Metrics:')
        results = bagel.evaluation.get_test_results(
            labels=test_kpi.labels,
            scores=anomaly_scores,
            missing=test_kpi.missing,
            window_size=window_size,
        )
        print(json.dumps(results, indent=2))

        stats = bagel.evaluation.get_kpi_stats(kpi, test_kpi)
        output_dict = {
            'name': kpi.name,
            'results': results,
            'overall_stats': stats[0].__dict__,
            'test_stats': stats[1].__dict__,
        }
        with open(output_path.joinpath(f'{kpi.name}.json'), 'w') as output:
            output.write(json.dumps(output_dict, indent=2))


if __name__ == '__main__':
    input_path = pathlib.Path('data')
    output_path = pathlib.Path('out').joinpath('bagel')
    main()

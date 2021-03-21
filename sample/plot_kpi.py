import bagel
import pathlib
import datetime
import numpy as np
import matplotlib.pyplot as plt


def _expand(a: np.ndarray) -> np.ndarray:
    ret = np.copy(a)
    for i in range(length := len(a)):
        if a[i] == 1:
            if i - 1 >= 0:
                ret[i - 1] = 1
            if i + 1 < length:
                ret[i + 1] = 1
    return ret


def _plot_kpi(kpi: bagel.data.KPI):
    x = [datetime.datetime.fromtimestamp(timestamp) for timestamp in kpi.timestamps]
    y_anomaly, y_missing = np.copy(kpi.values), np.copy(kpi.values)
    y_anomaly[_expand(kpi.labels) == 0] = np.inf
    y_missing[_expand(kpi.missing) == 0] = np.inf
    plt.plot(x, kpi.values)
    plt.plot(x, y_anomaly, color='red')
    plt.plot(x, y_missing, color='orange')
    plt.title(kpi.name)
    plt.ylim(y_limit)


def main():
    bagel.utils.mkdirs(output_path)
    file_list = bagel.utils.file_list(input_path)
    plt.rcParams.update({'font.size': font_size})

    plt.figure(figsize=fig_size, dpi=dpi)
    for i in range(total := len(file_list)):
        kpi = bagel.utils.load_kpi(file_list[i])
        print(f'Plotting: ({i + 1}/{total}) {kpi.name}')
        kpi, _, _ = kpi.standardize()
        kpi.complete_timestamp()
        _plot_kpi(kpi)
        plt.savefig(output_path.joinpath(f'{kpi.name}.png'))
        plt.clf()


if __name__ == '__main__':
    input_path = pathlib.Path('data')
    output_path = pathlib.Path('out').joinpath('plot')
    y_limit = (-7.5, 7.5)
    fig_size = (32, 4)
    dpi = 144
    font_size = 22
    main()

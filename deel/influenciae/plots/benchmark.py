# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Module for creating the visualizations for better understanding the results obtained
through the benchmark module.
"""
import os

import numpy as np
from matplotlib import pyplot as plt

from ..types import Dict, Tuple, Optional


class BenchmarkDisplay:
    """
    A class for generating the visualizations required to properly interpret the benchmarks'
    results.
    """

    @staticmethod
    def load_bench_result(path: str) -> Dict[str, Tuple[np.array, np.array, float]]:
        """
        Loads an evaluation's or whole benchmark's results.

        Parameters
        ----------
        path
            A string with the path to the desired results.

        Returns
        -------
        results
            A dictionary with the experiments' names and their corresponding results (curves, mean curve, roc)
        """
        result = np.load(os.path.join(path), allow_pickle=True)
        return result

    @staticmethod
    def plot_bench_from_path(path: str, path_to_save: str = None) -> None:
        """
        Loads the results from a file on disk, plots it and optionally saves the figure to the disk.

        Parameters
        ----------
        path
            A string with the file from which to load the results.
        path_to_save
            An (optional) string with the path onto which to save the figure.
        """
        result = BenchmarkDisplay.load_bench_result(path)
        BenchmarkDisplay.plot_bench(result, path_to_save)

    @staticmethod
    def plot_bench(
            result: Dict[str, Tuple[np.array, np.array, float]],
            path_to_save: Optional[str] = None,
            title: Optional[str] = None
    ) -> None:
        """
        Plots the results from a whole benchmark on a single figure, and optionally saves the
        figure to the disk.

        Parameters
        ----------
        result
            A dictionary with the experiments' names and their results.
        path_to_save
            An (optional) string with the path onto which to save the generated figure.
            If None, it only shows the figure, otherwise, it only saves the figure but doesn't show it.
        title
            An (optional) string specifying the figure's title.
        """
        fig, axs = plt.subplots(nrows=1, ncols=len(result), figsize=(20, 10))

        if len(result) == 1:
            axs = [axs]

        if title is not None:
            fig.suptitle(title, y=0.99)
        fig.subplots_adjust(top=0.8)

        for i, (name, (curves, mean_curve, roc)) in enumerate(result.items()):
            curve_length = len(mean_curve)
            valid_curvs = []
            for curve in curves:
                if not np.isnan(curve[0]):
                    valid_curvs.append(curve)
                    axs[i].plot(np.linspace(0., 1., curve_length), curve, 'C0', alpha=0.25)
            mean_curv = np.mean(np.asarray(valid_curvs), axis=0)
            axs[i].plot(np.linspace(0., 1., curve_length), mean_curv, 'C0')
            roc = np.mean(mean_curve)

            axs[i].plot(np.linspace(0., 1., curve_length), np.linspace(0., 1., curve_length), 'C1')
            axs[i].set_title(
                f'Mislabeled detection {name} \n ROC={roc} \n Nbr of run={len(valid_curvs)}')
            axs[i].set_xlabel('Part of the dataset searched')
            axs[i].set_ylabel('Part of mislabeled found')
            axs[i].grid('minor')
        if path_to_save is None:
            plt.show()
        else:
            plt.savefig(path_to_save)

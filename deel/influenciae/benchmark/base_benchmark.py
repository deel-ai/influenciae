# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import os
import random
from abc import abstractmethod

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .influence_factory import InfluenceCalculatorFactory
from deel.influenciae.types import Tuple, Dict, Any, Optional

class BaseTrainingProcedure:
    """
    TODO: Docs
    """
    @abstractmethod
    def train(
        self, training_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset,
        train_batch_size: int = 128, test_batch_size: int = 128) -> Tuple[float, float, tf.keras.Model, Any]:
        raise NotImplementedError

class MissingLabelEvaluator:
    """
    TODO: Docs
    """
    def __init__(
        self,
        training_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        training_procedure: BaseTrainingProcedure,
        nb_classes: int,
        misslabeling_ratio: float,
        train_batch_size: int = 128,
        test_batch_size: int = 128) -> None: 

        self.training_dataset = training_dataset
        self.train_batch_size = train_batch_size
        self.test_dataset = test_dataset
        self.test_batch_size = test_batch_size
        self.training_procedure = training_procedure
        self.nb_classes = nb_classes
        self.misslabeling_ratio = misslabeling_ratio

    def bench(self, influence_calculator_factories: Dict[str, InfluenceCalculatorFactory], nbr_of_evaluation: int,
              path_to_save: Optional[str],
              seed: int = 0,
              verbose: bool = True) -> Dict[str, Tuple[np.array, np.array, float]]:
        """
        TODO: Docs
        """
        result = {}
        for name, influence_calculator_factory in influence_calculator_factories.items():
            if verbose:
                print("starting to evaluate=" + str(name))

            if path_to_save is not None:
                current_path_to_save = path_to_save + "/" + name
            else:
                current_path_to_save = None

            curves, mean_curve, roc = self.evaluate(influence_calculator_factory, nbr_of_evaluation, seed, verbose,
                                                    current_path_to_save)

            result[name] = (curves, mean_curve, roc)

            if verbose:
                print(name + " | mean roc=" + str(roc))

        return result

    def evaluate(self, influence_factory: InfluenceCalculatorFactory, nbr_of_evaluation: int, seed: int = 0,
                 verbose: bool = True, path_to_save:str=None) -> Tuple[np.array, np.array, float]:
        """
        TODO: Docs
        """
        curves = []

        for index in range(nbr_of_evaluation):

            self.set_seed(seed + index)
            noisy_training_dataset, noisy_label_indexes = self.build_noisy_training_dataset()

            acc_train, acc_test, model, data_train = self.training_procedure.train(
                noisy_training_dataset,
                self.test_dataset,
                self.train_batch_size,
                self.test_batch_size)

            noisy_training_dataset = noisy_training_dataset.batch(self.train_batch_size)
            
            influence_calculator = influence_factory.build(noisy_training_dataset, model, data_train)
            influences_values = influence_calculator.compute_influence_values(noisy_training_dataset)
            # print(f"For index: {index}, influence values are computed")
            # compute curve and indexes
            sorted_influences_indexes = np.argsort(np.squeeze(influences_values))
            sorted_curve = self.__compute_curve(sorted_influences_indexes, noisy_label_indexes)
            # print(f"For index: {index}, curves are computed")
            curves.append(sorted_curve)

            roc = self._compute_roc(sorted_curve)
            # print(f"For index: {index}, roc is computed")
            if verbose:
                print("seed nbr=" + str(index) + " | acc train=" + str(acc_train) + " | acc test=" + str(
                    acc_test) + " | roc=" + str(roc))

            if path_to_save is not None:
                curves_, mean_curve_, roc_ = self.__build(curves)
                # print(f"For index: {index}, curves are build")
                self.__save(path_to_save, curves_, mean_curve_, roc_)
                # print(f"For index: {index}, temp save")
        # print(f"End for")
        curves, mean_curve, roc = self.__build(curves)
        # print("Final build done")

        return curves, mean_curve, roc

    def __build(self,curves):
        curves = np.asarray(curves)
        mean_curve = np.mean(curves, axis=0)
        roc = self._compute_roc(mean_curve)

        return curves, mean_curve, roc

    def _compute_roc(self, curve: np.array) -> float:
        roc = np.mean(curve)
        return roc

    def set_seed(self, seed: int):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __compute_curve(self, sorted_influences_indexes, noisy_label_indexes) -> np.array:
        index = np.in1d(sorted_influences_indexes, noisy_label_indexes)
        index = tf.cast(index, np.int32)
        curve = np.cumsum(index)
        if curve[-1] != 0:
            curve = curve / curve[-1]

        return curve

    def build_noisy_training_dataset(self) -> Tuple[tf.data.Dataset, np.array]:
        dataset_size = tf.data.experimental.cardinality(self.training_dataset)
        noise_mask = np.random.uniform(size=(dataset_size,)) > self.misslabeling_ratio

        noisy_label = np.round(np.random.uniform(size=(dataset_size,)) * self.nb_classes)
        noisy_label = np.cast[np.int32](noisy_label)
        noisy_label = np.where(noise_mask, -1 * np.ones_like(noisy_label), noisy_label)
        noisy_label = tf.convert_to_tensor(noisy_label, dtype=tf.int32)
        noise_mask_dataset = tf.data.Dataset.from_tensor_slices(noisy_label)

        noisy_dataset = tf.data.Dataset.zip((self.training_dataset, noise_mask_dataset))

        def noise_map(z, y_noise):
            (x, y) = z
            y = tf.where(y_noise > -1, tf.one_hot(y_noise, tf.shape(y)[-1]), y)
            return x, y

        noisy_dataset = noisy_dataset.map(noise_map)

        noise_indexes = np.where(np.logical_not(noise_mask))
        return noisy_dataset, noise_indexes

    def __save(self, path_to_save: str, curves:np.array, mean_curve:np.array, roc:float) -> None:
        dirname = os.path.dirname(path_to_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        np.save(path_to_save, (curves,mean_curve,roc), allow_pickle=True)


class Display:

    @staticmethod
    def load_bench_result(path: str) -> Dict[str, Tuple[np.array, np.array, float]]:
        result = np.load(os.path.join(path), allow_pickle=True)
        return result

    @staticmethod
    def plot_bench_from_path(path: str, path_to_save: str = None):
        result = Display.load_bench_result(path)
        Display.plot_bench(result, path_to_save)

    @staticmethod
    def plot_bench(result: Dict[str, Tuple[np.array, np.array, float]], path_to_save: str = None, title: str = None):
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

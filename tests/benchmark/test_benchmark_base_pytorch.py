import numpy as np
import pytest
import torch

from torch.utils.data import TensorDataset

from deel.influenciae.types import Optional, Tuple, Any
from deel.influenciae.benchmark.base_benchmark import MislabelingDetectorEvaluator, BaseTrainingProcedure


pytestmark = pytest.mark.pytorch


class MockTrainingProcedure(BaseTrainingProcedure):

    def train(
            self,
            training_dataset: Any,
            test_dataset: Any,
            train_batch_size: int = 128,
            test_batch_size: int = 128,
            log_path: Optional[str] = None) -> Tuple[float, float, Any, Any]:
        raise NotImplementedError


def test_noise_pytorch(tmp_path):
    np.random.seed(0)
    torch.manual_seed(0)

    size = 10000
    x = torch.linspace(1, size, steps=size)
    class_nbr = 10
    y = torch.zeros((size, class_nbr), dtype=torch.float32)
    y[:, 0] = 1.0
    training_dataset = TensorDataset(x, y)

    misslabeling_ratio = 0.1
    evaluator = MislabelingDetectorEvaluator(
        training_dataset,
        test_dataset=None,
        training_procedure=MockTrainingProcedure(),
        nb_classes=class_nbr,
        mislabeling_ratio=misslabeling_ratio,
        train_batch_size=128,
        test_batch_size=128,
        config=None,
    )

    noisy_dataset, noise_indexes = evaluator.build_noisy_training_dataset()

    noise_ratio_computed = len(noise_indexes[0]) / size
    assert abs((noise_ratio_computed - misslabeling_ratio) / misslabeling_ratio) < 1E-1

    count = 0
    for _, y_value in noisy_dataset:
        count += int(torch.sum(y_value[1:]).item())

    assert count - len(noise_indexes[0]) == 0

    curve = evaluator._MislabelingDetectorEvaluator__compute_curve(  # pylint: disable=W0212
        sorted_influences_indexes=[2, 6, 3, 4, 5, 1, 7, 8, 9, 10],
        noisy_label_indexes=[6, 8]
    )
    curve_expected = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
    assert np.max(np.abs(curve - curve_expected)) < 1E-6

    curves = [
        [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
    ]
    mean_curve = np.mean(curves, axis=0)
    roc = np.mean(mean_curve)

    save_path = str(tmp_path / "exp1")
    evaluator._MislabelingDetectorEvaluator__save(save_path, curves, mean_curve, roc)  # pylint: disable=W0212

    result = np.load(str(tmp_path / "exp1.npy"), allow_pickle=True)

    assert np.max(np.abs(curve - result[0][0])) < 1E-6
    assert np.max(np.abs(mean_curve - result[1])) < 1E-6
    assert np.max(np.abs(roc - result[2])) < 1E-6

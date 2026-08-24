# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""Backend-independent ASTRA configuration and API tests."""
import pytest

from deel.influenciae.common import AstraConfig, AstraIHVPFactory, IHVPCalculator


pytestmark = pytest.mark.backend_agnostic


def test_astra_config_defaults_and_api_wiring():
    config = AstraConfig()
    assert config.damping == 1e-4
    assert config.preconditioner_damping == config.damping
    assert config.learning_rate_at(0) == 1e-2
    assert IHVPCalculator.from_string("astra") is IHVPCalculator.Astra
    assert AstraIHVPFactory(config).config is config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"damping": 0},
        {"damping": float("nan")},
        {"preconditioner_damping": -1},
        {"n_iterations": -1},
        {"learning_rate": 0},
        {"momentum": 1},
        {"momentum": -0.1},
        {"ggn_batch_size": 0},
        {"ggn_shuffle_buffer_size": 0},
        {"rhs_chunk_size": 0},
        {"seed": "zero"},
    ],
)
def test_astra_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        AstraConfig(**kwargs)


def test_astra_config_validates_callable_learning_rate_per_step():
    config = AstraConfig(learning_rate=lambda step: 0.1 if step == 0 else 0.0)
    assert config.learning_rate_at(0) == 0.1
    with pytest.raises(ValueError, match="step 1"):
        config.learning_rate_at(1)

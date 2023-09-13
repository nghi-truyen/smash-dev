from __future__ import annotations

from smash.core.simulation._standardize import (
    _standardize_simulation_samples,
    _standardize_simulation_mapping,
    _standardize_simulation_optimizer,
    _standardize_simulation_optimize_options,
    _standardize_simulation_cost_options,
    _standardize_simulation_common_options,
    _standardize_simulation_return_options,
    _standardize_simulation_parameters_feasibility,
    _standardize_simulation_optimize_options_finalize,
    _standardize_simulation_cost_options_finalize,
    _standardize_simulation_return_options_finalize,
)

from smash._constant import MAPPING

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.factory.samples.samples import Samples
    from smash._typing import AnyTuple


def _standardize_optimize_args(
    model: Model,
    mapping: str,
    optimizer: str | None,
    optimize_options: dict | None,
    cost_options: dict | None,
    common_options: dict | None,
    return_options: dict | None,
) -> AnyTuple:
    # % In case model.set_opr_parameters or model.set_opr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    mapping = _standardize_simulation_mapping(mapping)

    optimizer = _standardize_simulation_optimizer(mapping, optimizer)

    optimize_options = _standardize_simulation_optimize_options(
        model, mapping, optimizer, optimize_options
    )

    cost_options = _standardize_simulation_cost_options(model, cost_options)

    common_options = _standardize_simulation_common_options(common_options)

    return_options = _standardize_simulation_return_options(
        model, "optimize", return_options
    )

    # % Finalize optimize options
    _standardize_simulation_optimize_options_finalize(
        model, mapping, optimizer, optimize_options
    )

    # % Finalize cost_options

    _standardize_simulation_cost_options_finalize(model, cost_options)
    # % Finalize return_options
    _standardize_simulation_return_options_finalize(model, return_options)

    return (
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
        return_options,
    )


def _standardize_multiple_optimize_args(
    model: Model,
    samples: Samples,
    mapping: str,
    optimizer: str | None,
    optimize_options: dict | None,
    cost_options: dict | None,
    common_options: dict | None,
) -> AnyTuple:
    samples = _standardize_simulation_samples(model, samples)

    # % In case model.set_opr_parameters or model.set_opr_initial_states were not used
    _standardize_simulation_parameters_feasibility(model)

    mapping = _standardize_multiple_optimize_mapping(mapping)

    optimizer = _standardize_simulation_optimizer(mapping, optimizer)

    optimize_options = _standardize_simulation_optimize_options(
        model, mapping, optimizer, optimize_options
    )

    cost_options = _standardize_simulation_cost_options(model, cost_options)

    common_options = _standardize_simulation_common_options(common_options)

    # % Finalize optimize options
    _standardize_simulation_optimize_options_finalize(
        model, mapping, optimizer, optimize_options
    )

    # % Finalize cost_options
    _standardize_simulation_cost_options_finalize(model, cost_options)

    return (
        samples,
        mapping,
        optimizer,
        optimize_options,
        cost_options,
        common_options,
    )


def _standardize_multiple_optimize_mapping(mapping: str) -> str:
    avail_mapping = MAPPING.copy()
    avail_mapping.remove("ann")  # cannot perform multiple optimize with ANN mapping

    if isinstance(mapping, str):
        if mapping.lower() not in avail_mapping:
            raise ValueError(
                f"Invalid mapping '{mapping}' for multiple optimize. Choices: {avail_mapping}"
            )
    else:
        raise TypeError("mapping argument must be a str")

    return mapping.lower()

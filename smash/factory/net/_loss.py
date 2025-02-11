from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smash._constant import OPTIMIZABLE_NN_PARAMETERS
from smash.core.simulation.optimize._tools import _get_parameters_b, _set_parameters_from_net
from smash.fcore._mwd_parameters_manipulation import (
    control_to_parameters as wrap_control_to_parameters,
)
from smash.fcore._mwd_parameters_manipulation import (
    parameters_to_control as wrap_parameters_to_control,
)

from smash.fcore._mw_forward import forward_run as wrap_forward_run

import time

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.factory.net.net import Net
    from smash.fcore._mwd_options import OptionsDT, ParametersDT
    from smash.fcore._mwd_returns import ReturnsDT


def _get_cost_value(instance: Model) -> float:
    return instance._output.cost


def _get_gradient_value(
    net: Net,
    x: np.ndarray,
    calibrated_parameters: np.ndarray,
    instance: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
) -> tuple[np.ndarray, list]:
    # % Update rr_parameters and/or rr_initial_states using net
    y_reshaped = _set_parameters_from_net(net, x, calibrated_parameters, parameters)

    # % Change the mapping to trigger distributed control to get distributed gradients
    wrap_options.optimize.mapping = "distributed"

    ts = time.time()
    wrap_forward_run(
                instance.setup,
                instance.mesh,
                instance._input_data,
                parameters.copy(),
                instance._output,
                wrap_options.copy(),
                wrap_returns.copy(),
            )
    t_forward = time.time() - ts

    # % Run adjoint model
    wrap_parameters_to_control(
        instance.setup,
        instance.mesh,
        instance._input_data,
        parameters,
        wrap_options,
    )
    ts = time.time()
    parameters_b = _get_parameters_b(instance, parameters, wrap_options, wrap_returns)
    t_backward = time.time() - ts

    wrap_control_to_parameters(
        instance.setup, instance.mesh, instance._input_data, parameters_b, wrap_options
    )

    # % Reset mapping to ann
    wrap_options.optimize.mapping = "ann"

    # % Get the gradient of the descriptors-to-parameters (d2p) NN
    grad_d2p = []
    for name in calibrated_parameters:
        if name in parameters.rr_parameters.keys:
            ind = np.argwhere(parameters.rr_parameters.keys == name).item()

            grad_d2p.append(parameters_b.rr_parameters.values[..., ind])

        elif name in parameters.rr_initial_states.keys:
            ind = np.argwhere(parameters.rr_initial_states.keys == name).item()

            grad_d2p.append(parameters_b.rr_initial_states.values[..., ind])

        else:  # nn_parameters excluded from descriptors-to-parameters mapping
            pass

    if y_reshaped:  # in case of Dense (MLP)
        grad_d2p = np.reshape(grad_d2p, (len(grad_d2p), -1)).T

    else:  # in case of CNN
        grad_d2p = np.transpose(grad_d2p, (1, 2, 0))

    # % Get the gradient of parameterization (pmtz) NN if used
    grad_pmtz = [
        getattr(parameters_b.nn_parameters, key).copy()
        for key in OPTIMIZABLE_NN_PARAMETERS[max(0, instance.setup.n_layers - 1)]
    ]

    return (grad_d2p, grad_pmtz, t_forward, t_backward)

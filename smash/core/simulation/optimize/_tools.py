from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smash._constant import (
    CONTROL_PRIOR_DISTRIBUTION,
    CONTROL_PRIOR_DISTRIBUTION_PARAMETERS,
)
from smash.core.model._build_model import _map_dict_to_fortran_derived_type
from smash.fcore._mw_forward import forward_run_b as wrap_forward_run_b
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_parameters_manipulation import (
    parameters_to_control as wrap_parameters_to_control,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.factory.net.net import Net
    from smash.fcore._mwd_parameters import ParametersDT
    from smash.fcore._mwd_returns import ReturnsDT


def _get_control_info(
    model: Model,
    mapping: str,
    optimizer: str,
    optimize_options: dict,
    cost_options: dict,
) -> dict:
    wrap_options = OptionsDT(
        model.setup,
        model.mesh,
        cost_options["njoc"],
        cost_options["njrc"],
    )

    # % Map optimize_options dict to derived type
    _map_dict_to_fortran_derived_type(optimize_options, wrap_options.optimize)

    # % Map cost_options dict to derived type
    _map_dict_to_fortran_derived_type(cost_options, wrap_options.cost)

    wrap_parameters_to_control(model.setup, model.mesh, model._input_data, model._parameters, wrap_options)

    ret = {}
    for attr in dir(model._parameters.control):
        if attr.startswith("_"):
            continue
        value = getattr(model._parameters.control, attr)
        if callable(value):
            continue
        if hasattr(value, "copy"):
            value = value.copy()
        ret[attr] = value

    # Manually dealloc the control
    model._parameters.control.dealloc()

    _finalize_get_control_info(ret, optimize_options)

    return ret


def _finalize_get_control_info(ret: dict, optimize_options: dict):
    # % Handle unbounded and semi-unbounded parameters
    for key in ["l", "u", "l_bkg", "u_bkg"]:
        if key.startswith("l"):
            ret[key] = np.where(np.isin(ret["nbd"], [0, 3]), -np.inf, ret[key])

        elif key.startswith("u"):
            ret[key] = np.where(np.isin(ret["nbd"], [0, 1]), np.inf, ret[key])

    # % Handle control info from net
    net = optimize_options.get("net", None)

    if net is None:
        ret["nbk"] = np.append(ret["nbk"], 0)

    else:
        x_net_name = []
        x_net_size = 0
        for i, layer in enumerate(net.layers):
            if layer.trainable:
                x_net_name.extend(
                    f"reg_weight_{i + 1}-{row + 1}-{col + 1}"
                    for col in range(layer.weight_shape[1])
                    for row in range(layer.weight_shape[0])
                )

                x_net_name.extend(f"reg_bias_{i + 1}-{col + 1}" for col in range(layer.bias_shape[-1]))

                x_net_size += np.prod(layer.weight_shape) + layer.bias_shape[-1]

        ret["name"] = np.append(ret["name"], x_net_name)

        ret["n"] += x_net_size

        ret["nbk"] = np.append(ret["nbk"], x_net_size)

        ret["nbd"] = np.append(ret["nbd"], np.zeros(x_net_size))

        for key in ["l", "u", "l_bkg", "u_bkg"]:
            if key.startswith("l"):
                value = -np.inf
            elif key.startswith("u"):
                value = np.inf

            ret[key] = np.append(ret[key], np.full(x_net_size, value))

        # Set weights and biases if net is not initialized
        # Otherwise, it will be initialized with random values
        if optimize_options["random_state"] is not None:
            np.random.seed(optimize_options["random_state"])
        for layer in net.layers:
            if hasattr(layer, "_initialize"):
                layer._initialize(None)
        x = _net_to_vect(net)

        ret["x"] = np.append(ret["x"], x)

        ret["x_bkg"] = np.append(ret["x_bkg"], x)  # no transformation applied to x


def _net_to_vect(net: Net) -> np.ndarray:
    x = []

    for layer in net.layers:
        if layer.trainable:
            x.append(layer.weight.flatten("F"))  # same order with nn_parameters_weight
            x.append(layer.bias.flatten())

    return np.concatenate(x)


def _net_to_parameters(
    net: Net, x: np.ndarray, model_params_states: np.ndarray, parameters: ParametersDT, flwdir_shape: tuple
):
    # % Forward propogation
    y = net._forward_pass(x)

    output_reshaped = False

    if y.ndim < 3:
        y = y.reshape(flwdir_shape + (-1,))
        output_reshaped = True  # reshape output in case of Dense (MLP)

    # % Set parameters or states
    for i, name in enumerate(model_params_states):
        if name in parameters.rr_parameters.keys:
            ind = np.argwhere(parameters.rr_parameters.keys == name).item()

            parameters.rr_parameters.values[..., ind] = y[..., i]

        elif name in parameters.rr_initial_states.keys:
            ind = np.argwhere(parameters.rr_initial_states.keys == name).item()

            parameters.rr_initial_states.values[..., ind] = y[..., i]

    return output_reshaped


def _get_lcurve_wjreg_best(
    cost_arr: np.ndarray,
    jobs_arr: np.ndarray,
    jreg_arr: np.ndarray,
    wjreg_arr: np.ndarray,
) -> (np.ndarray, float):
    jobs_min = np.min(jobs_arr)
    jobs_max = np.max(jobs_arr)
    jreg_min = np.min(jreg_arr)
    jreg_max = np.max(jreg_arr)

    if (jobs_max - jobs_min) < 0 or (jreg_max - jreg_min) < 0:
        return np.empty(shape=0), 0.0

    max_distance = 0.0
    distance = np.zeros(shape=cost_arr.size)

    for i in range(cost_arr.size):
        lcurve_y = (jreg_arr[i] - jreg_min) / (jreg_max - jreg_min)
        lcurve_x = (jobs_max - jobs_arr[i]) / (jobs_max - jobs_min)
        # % Skip point above y = x
        if lcurve_y < lcurve_x:
            if jobs_arr[i] < jobs_max:
                hypot = np.hypot(lcurve_x, lcurve_y)
                alpha = np.pi * 0.25 - np.arccos(lcurve_x / hypot)
                distance[i] = hypot * np.sin(alpha)

            if distance[i] > max_distance:
                max_distance = distance[i]
                wjreg = wjreg_arr[i]

        else:
            distance[i] = np.nan

    return distance, wjreg


def _handle_bayesian_optimize_control_prior(model: Model, control_prior: dict, options: OptionsDT):
    wrap_parameters_to_control(model.setup, model.mesh, model._input_data, model._parameters, options)

    if control_prior is None:
        control_prior = {}

    elif isinstance(control_prior, dict):
        for key, value in control_prior.items():
            if key not in model._parameters.control.name:
                raise ValueError(
                    f"Unknown control name '{key}' in control_prior cost_options. "
                    f"Choices: {list(model._parameters.control.name)}"
                )
            else:
                if isinstance(value, (list, tuple, np.ndarray)):
                    if value[0] not in CONTROL_PRIOR_DISTRIBUTION:
                        raise ValueError(
                            f"Unknown distribution '{value[0]}' for key '{key}' in control_prior "
                            f"cost_options. Choices: {CONTROL_PRIOR_DISTRIBUTION}"
                        )
                    value[1] = np.array(value[1], dtype=np.float32)
                    if value[1].size != CONTROL_PRIOR_DISTRIBUTION_PARAMETERS[value[0]]:
                        raise ValueError(
                            f"Invalid number of parameter(s) ({value[1].size}) for distribution '{value[0]}' "
                            f"for key '{key}' in control_prior cost_options. "
                            f"Expected: ({CONTROL_PRIOR_DISTRIBUTION_PARAMETERS[value[0]]})"
                        )
                else:
                    raise ValueError(
                        f"control_prior cost_options value for key '{key}' must be of ListLike (List, "
                        f"Tuple, np.ndarray)"
                    )
            control_prior[key] = {"dist": value[0], "par": value[1]}
    else:
        raise TypeError("control_prior cost_options must be a dictionary")

    for key in model._parameters.control.name:
        control_prior.setdefault(key, {"dist": "FlatPrior", "par": np.empty(shape=0)})

    # % allocate control prior
    npar = np.array([p["par"].size for p in control_prior.values()], dtype=np.int32)
    options.cost.alloc_control_prior(model._parameters.control.n, npar)

    # % map control prior dict to derived type array
    for i, prior in enumerate(control_prior.values()):
        _map_dict_to_fortran_derived_type(prior, options.cost.control_prior[i])


def _forward_run_b(
    model: Model,
    parameters: ParametersDT,
    wrap_options: OptionsDT,
    wrap_returns: ReturnsDT,
):
    parameters_b = parameters.copy()
    output_b = model._output.copy()
    output_b.cost = np.float32(1)

    wrap_forward_run_b(
        model.setup,
        model.mesh,
        model._input_data,
        parameters,
        parameters_b,
        model._output,
        output_b,
        wrap_options,
        wrap_returns,
    )

    return parameters_b


def _inf_norm(grad: np.ndarray | list) -> float:
    if isinstance(grad, list):
        if grad:  # If not an empty list
            return max(_inf_norm(g) for g in grad)

        else:
            return 0

    elif isinstance(grad, np.ndarray):
        if grad.size > 0:
            return np.amax(np.abs(grad))

        else:
            return 0

    else:  # Should be unreachable
        pass

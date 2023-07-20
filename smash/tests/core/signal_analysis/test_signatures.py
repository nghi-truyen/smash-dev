from __future__ import annotations

import smash
from smash._constant import CSIGN, ESIGN

import numpy as np
import pytest


def generic_signatures(model: smash.Model, qs: np.ndarray, **kwargs) -> dict:
    res = {}

    instance = model.copy()

    instance.sim_response.q = qs

    signresult = {}

    signresult["obs_by_obs"] = smash.signatures(
        instance, domain="obs", event_seg={"by": "obs"}
    )
    signresult["sim_by_obs"] = smash.signatures(
        instance, domain="sim", event_seg={"by": "obs"}
    )
    signresult["sim_by_sim"] = smash.signatures(
        instance, domain="sim", event_seg={"by": "sim"}
    )

    for typ, sign in zip(
        ["cont", "event"], [CSIGN[:4], ESIGN]
    ):  # % remove percentile signatures calculation
        for dom in signresult.keys():
            res[f"signatures.{typ}_{dom}"] = signresult[dom][typ][sign].to_numpy(
                dtype=np.float32
            )

    return res


def test_signatures():
    res = generic_signatures(pytest.model, pytest.simulated_discharges["sim_q"][:])

    for key, value in res.items():
        # % Check signatures for cont/event and obs/sim
        assert np.allclose(
            value,
            pytest.baseline[key][:],
            equal_nan=True,
            atol=1e-06,
        ), key

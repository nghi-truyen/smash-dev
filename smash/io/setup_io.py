from __future__ import annotations

import yaml
import os
import errno

__all__ = ["save_setup", "read_setup"]


def save_setup(setup: dict, path: str):

    """
    Save setup
    """

    if not path.endswith(".yaml"):

        path = path + ".yaml"

    with open(path, "w") as f:
        yaml.dump(setup, f, default_flow_style=False)


def read_setup(path: str) -> dict:

    """
    Read setup
    """

    if os.path.isfile(path):

        with open(path, "r") as f:

            setup = yaml.safe_load(f)

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    return setup

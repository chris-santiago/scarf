import torch

import scarf.constants

constants = scarf.constants.Constants()
constants.DATA.mkdir(exist_ok=True)


def set_device():
    device = {True: torch.device("mps"), False: torch.device("cpu")}
    return device[torch.backends.mps.is_available()]

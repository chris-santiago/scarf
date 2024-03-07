import pandas as pd
import torch
import torch.utils.data

import scarf.constants
from scarf.models.scarf import MaskGenerator, PretextGenerator

constants = scarf.constants.Constants()


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None):
        self.dataset = torch.tensor(pd.read_csv(file_path).values, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs = self.dataset[:, :-1].__getitem__(idx)
        outputs = self.dataset[:, -1].__getitem__(idx).int()
        if self.transform:
            inputs = self.transform(inputs)
        return inputs, outputs


def generate_corrupted_pairs(ds: CSVDataset, n_iter: int = 10, p_mask: float = 0.6):
    mask_gen = MaskGenerator(p=p_mask)
    corrupt_gen = PretextGenerator()

    pairs = []
    for _ in range(n_iter):
        x = ds.dataset[:, :-1]
        mask = mask_gen(x)
        x_corrupt = corrupt_gen(x, mask)
        pairs.append((x_corrupt, x))

    return torch.concat([x[0] for x in pairs]), torch.concat([x[1] for x in pairs])


def make_static_validation_data(file_name: str = "valid.csv"):
    ds = CSVDataset(constants.DATA.joinpath(file_name))
    x_corrupt, x = generate_corrupted_pairs(ds)

    prefix = file_name.split(".")[0]
    fp = constants.DATA.joinpath(f"{prefix}-corrupt.pt")
    torch.save(x_corrupt, fp)

    fp = constants.DATA.joinpath(f"{prefix}-x.pt")
    torch.save(x, fp)


# Strange hydra behavior when parameterizing these filepaths.
# Hydra didn't recognize _target_ call when followed with filepaths.
# Install is editable, so not sure what was going on...maybe some sort of cache?
class StaticDataset(torch.utils.data.Dataset):
    def __init__(
        self, corrupt_fp: str = "valid-s4e2-ps-corrupt.pt", valid_fp: str = "valid-s4e2-ps-x.pt"
    ):
        self.x_corrupt = torch.load(constants.DATA.joinpath(corrupt_fp))
        self.x = torch.load(constants.DATA.joinpath(valid_fp))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        inputs = self.x_corrupt.__getitem__(idx)
        outputs = self.x.__getitem__(idx)
        return inputs, outputs

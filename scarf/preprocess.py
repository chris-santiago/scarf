from typing import Dict, List

import hydra
import pandas as pd
import sklearn.pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

import scarf.constants

constants = scarf.constants.Constants()


def make_datasets(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    train, test = train_test_split(
        data, test_size=0.1, random_state=constants.SEED, stratify=data.iloc[:, -1]
    )

    train, valid = train_test_split(
        train, test_size=0.1, random_state=constants.SEED, stratify=train.iloc[:, -1]
    )

    return {"train": train, "valid": valid, "test": test}


def make_processor(
    num_features: List[str], cat_features: List[str], target: str
) -> sklearn.compose.ColumnTransformer:
    return ColumnTransformer(
        [
            ("ohe", OneHotEncoder(), cat_features),
            ("scale", StandardScaler(), num_features),
            ("target", OrdinalEncoder(), [target]),
        ]
    )


@hydra.main(version_base="1.3", config_path="./conf", config_name="preprocess")
def main(cfg):
    """
    Pull data sample.

    Parameters
    ----------
    cfg: omegaconf.DictConfig
        An omegaconf dictionary configuration.

    Returns
    -------
    None
    """
    file_in = constants.DATA.joinpath(cfg.data.raw)
    data = pd.read_csv(file_in)
    ds = make_datasets(data)
    processor = make_processor(list(cfg.features.numeric), list(cfg.features.categoric), cfg.target)

    ds["train"] = processor.fit_transform(ds["train"])
    for name, df in ds.items():
        file_out = f"{name}.csv"
        if name != "train":
            df = processor.transform(df)
        df = pd.DataFrame(df, columns=processor.get_feature_names_out())
        df.to_csv(constants.DATA.joinpath(file_out), index=False)


if __name__ == "__main__":
    main()

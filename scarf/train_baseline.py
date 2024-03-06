"""
>>> hydra.initialize(config_path='scarf/conf', version_base="1.3")
>>> cfg = hydra.compose(config_name='config')
"""

import hydra
import joblib
import torch

import scarf.callbacks
import scarf.constants

constants = scarf.constants.Constants()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = constants.REPO.joinpath(hydra_cfg.runtime.output_dir)

    train_dl = hydra.utils.instantiate(cfg.data.train)
    valid_dl = hydra.utils.instantiate(cfg.data.valid)

    x_train = torch.concat(
        [train_dl.dataset.dataset[:, :-1], valid_dl.dataset.dataset[:, :-1]]
    ).numpy()

    y_train = torch.concat(
        [train_dl.dataset.dataset[:, -1], valid_dl.dataset.dataset[:, -1]]
    ).numpy()

    estimator = hydra.utils.instantiate(cfg.model.estimator)
    estimator.fit(x_train, y_train)

    fp = str(out_dir.joinpath(f"{cfg.model.name}-estimator.joblib"))
    joblib.dump(estimator, fp)


if __name__ == "__main__":
    main()

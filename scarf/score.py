"""
>>> hydra.initialize(config_path='scarf/conf', version_base="1.3")
>>> cfg = hydra.compose(config_name='config')
"""

import hydra
import pandas as pd
import torch

import scarf.constants

constants = scarf.constants.Constants()


@hydra.main(config_path="conf", config_name="score", version_base="1.3")
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = constants.REPO.joinpath(hydra_cfg.runtime.output_dir)

    estimator = hydra.utils.instantiate(cfg.model.estimator)
    preprocessor = hydra.utils.instantiate(cfg.preprocessor)

    test = pd.read_csv(cfg.data.test)
    test["NObeyesdad"] = "Normal_Weight"  # adding for processor compatibility
    test["CALC"].replace({"Always": "Frequently"}, inplace=True)  # hotfix

    test_x = torch.tensor(preprocessor.transform(test.iloc[:, 1:])[:, :-1], dtype=torch.float32)

    preds = []
    batch = 1024
    i = 0
    while i < len(test_x):
        estimator.eval()
        with torch.no_grad():
            preds.append(estimator.forward(test_x[i : i + batch, :]).argmax(1))
        i += batch

    preds = torch.concat(preds).numpy()
    labels = preprocessor.named_transformers_.target.inverse_transform(preds.reshape(-1, 1))
    final = pd.DataFrame(
        {
            "id": test["id"],
            "NObeyesdad": labels.ravel(),
        }
    )
    fp = out_dir.joinpath("submission.csv")
    final.to_csv(fp, index=False)


if __name__ == "__main__":
    main()

"""
>>> hydra.initialize(config_path='scarf/conf', version_base="1.3")
>>> cfg = hydra.compose(config_name='eval')
"""

import json

import hydra
import torch
import torch.nn as nn

import scarf.constants

constants = scarf.constants.Constants()


@hydra.main(config_path="conf", config_name="eval", version_base="1.3")
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = constants.REPO.joinpath(hydra_cfg.runtime.output_dir)

    metrics = hydra.utils.instantiate(cfg.metrics)
    plots = hydra.utils.instantiate(cfg.plots)

    test_dl = hydra.utils.instantiate(cfg.data.test)
    test_x = test_dl.dataset.dataset[:, :-1]
    test_y = test_dl.dataset.dataset[:, -1].int()

    results = {}
    for model in cfg.models:
        estimator = hydra.utils.instantiate(cfg.models[model].estimator)

        if isinstance(estimator, nn.Module):
            estimator.eval()
            with torch.no_grad():
                preds = estimator.forward(test_x)

        elif hasattr(estimator, "predict_proba"):  # sklearn estimators
            preds = estimator.predict_proba(test_x.numpy())
            preds = torch.tensor(preds)

        else:
            raise ValueError(f"Unexpected estimator type: {type(estimator)}")

        results[model] = {
            name: round(metric(preds, test_y).item(), 4) for name, metric in metrics.items()
        }

        for name, metric in plots.items():
            fig, ax = metric.plot(metric(preds, test_y))
            ax.set_title(f"{name}, {cfg.models[model].name} , {cfg.data.name.title()} Data")
            fp = out_dir.joinpath(
                hydra_cfg.runtime.output_dir, f"{name} - {cfg.models[model].name}.png"
            )
            fig.savefig(fp)

    with open(str(out_dir.joinpath("results.json")), "w") as fp:
        json.dump(results, fp, indent=4)


if __name__ == "__main__":
    main()

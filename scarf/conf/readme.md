# Hydra

Use Hydra Compose API to interactively debug:

```python

import hydra

hydra.initialize(version_base="1.3", config_path="scarf/conf")
cfg = hydra.compose(config_name='preprocess')
```

To clear config:

```python
import hydra.core.global_hydra

hydra.core.global_hydra.GlobalHydra.instance().clear()
```
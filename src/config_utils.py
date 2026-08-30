"""
src/config_utils.py

WHY THIS TINY FILE EXISTS
---------------------------
`load_config()` was previously duplicated verbatim in both
data_preprocessing.py and train.py. tune.py needs the exact same
behavior, and this was the right moment to stop copy-pasting it a third
time. Pulling it into its own module also gives tune.py something
genuinely useful to import via an absolute path (`from src.config_utils
import load_config`), per this project's move toward absolute imports —
importing a shared utility this way is clean and safe; importing
train.py's copy directly would not be (train.py has its own bare,
non-absolute imports of sibling modules, so importing it as `src.train`
requires a different sys.path setup than importing it as top-level
`train.py`, and mixing the two conventions in one process is a real way
to get confusing, order-dependent import failures).

NOTE FOR FOLLOW-UP: data_preprocessing.py and train.py still have their
own local copies of this function from before this file existed. They
still work correctly, but pointing them at this shared version instead
would remove the last of the duplication — worth doing in a small
follow-up cleanup pass rather than as an unrequested change bundled into
this one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config
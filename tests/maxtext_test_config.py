# SPDX-License-Identifier: Apache-2.0
"""The MaxText base config tests build on, chosen as MaxText's own tests choose it."""

import os

from maxtext.common.gcloud_stub import is_decoupled
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


def get_test_config_path() -> str:
    name = "decoupled_base_test.yml" if is_decoupled() else "base.yml"
    return os.path.join(MAXTEXT_CONFIGS_DIR, name)

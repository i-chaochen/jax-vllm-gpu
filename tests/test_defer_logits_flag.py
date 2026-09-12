"""The output-head deferral flag must be one variable, read identically.

MaxText decides whether to return logits or hidden states; the plugin decides
whether to apply the head itself. If those two disagree, the sampler is handed
hidden states and the run produces garbage rather than a small regression --
so the coupling is asserted here rather than left to convention.
"""

import os

from jax_vllm_plugin.model_runner import DEFER_LOGITS_ENV_VAR


def test_plugin_and_maxtext_agree_on_the_env_var_name():
    from maxtext.models.models import DEFER_OUTPUT_HEAD_ENV

    assert DEFER_LOGITS_ENV_VAR == DEFER_OUTPUT_HEAD_ENV


def test_deferral_is_off_by_default():
    """maxengine samples inside the jitted step and needs the logits."""
    from maxtext.models.models import defer_output_head

    saved = os.environ.pop(DEFER_LOGITS_ENV_VAR, None)
    try:
        assert defer_output_head() is False
    finally:
        if saved is not None:
            os.environ[DEFER_LOGITS_ENV_VAR] = saved


def test_only_the_exact_string_one_enables_deferral():
    from maxtext.models.models import defer_output_head

    saved = os.environ.get(DEFER_LOGITS_ENV_VAR)
    try:
        for value, expected in (("1", True), ("0", False), ("true", False), ("", False)):
            os.environ[DEFER_LOGITS_ENV_VAR] = value
            assert defer_output_head() is expected, value
    finally:
        if saved is None:
            os.environ.pop(DEFER_LOGITS_ENV_VAR, None)
        else:
            os.environ[DEFER_LOGITS_ENV_VAR] = saved

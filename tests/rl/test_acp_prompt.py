#!/usr/bin/env python

import random

from lerobot.rl.acp_prompt import resolve_acp_conditioned_task
from lerobot.rl.acp_tags import ACP_NEGATIVE_TAG, ACP_POSITIVE_TAG


def test_positive_only_conditional_negative_is_plain_task():
    task = "pick the screw"
    out = resolve_acp_conditioned_task(
        task,
        is_positive=False,
        positive_only_conditional=True,
        unconditional_prob=0.0,
        rng=random.Random(0),
    )
    assert out == task
    assert ACP_NEGATIVE_TAG not in out


def test_positive_only_conditional_positive_gets_tag():
    task = "pick the screw"
    out = resolve_acp_conditioned_task(
        task,
        is_positive=True,
        positive_only_conditional=True,
        unconditional_prob=0.0,
        rng=random.Random(0),
    )
    assert ACP_POSITIVE_TAG in out


def test_positive_only_conditional_dropout_applies_to_positive():
    out = resolve_acp_conditioned_task(
        "pick the screw",
        is_positive=True,
        positive_only_conditional=True,
        unconditional_prob=1.0,
        rng=random.Random(0),
    )
    assert out == "pick the screw"


def test_dual_branch_mode_tags_negative():
    task = "pick the screw"
    out = resolve_acp_conditioned_task(
        task,
        is_positive=False,
        positive_only_conditional=False,
        unconditional_prob=0.0,
        rng=random.Random(0),
    )
    assert ACP_NEGATIVE_TAG in out

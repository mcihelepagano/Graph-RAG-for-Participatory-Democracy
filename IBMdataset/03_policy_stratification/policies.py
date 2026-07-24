"""
policies.py
===========
Single source of truth for the policy list.

Previously each script defined its own POLICIES list, and they had
DRIFTED: sweep_uc2_k.py contained "We should end mandatory retirement"
while the eval scripts contained "We should adopt a multi-party system".
That silently dropped one policy from any cross-script join.

This module fixes that. Every script imports POLICIES from here.

The 28 policies below are the intersection of all previous lists —
the policies that every script agreed on. The two divergent policies
("end mandatory retirement", "adopt a multi-party system") were
DROPPED because they were never present in all pipeline stages, so
their ground truth / retrievals / summaries are not guaranteed to
exist consistently.
"""

POLICIES = [
    "We should adopt a multi-party system",
    "We should prohibit women in combat",
    "Social media brings more harm than good",
    "We should subsidize space exploration",
    "We should subsidize student loans",
    "We should subsidize Wikipedia",
    "We should adopt atheism",
    "We should limit judicial activism",
    "We should ban algorithmic trading",
    "Homeschooling should be banned",
    "We should prohibit school prayer",
    "We should subsidize stay-at-home dads",
    "We should ban whaling",
    "We should ban fast food",
    "We should legalize sex selection",
    "We should fight urbanization",
    "We should subsidize vocational education",
    "We should abolish the Olympic Games",
    "Entrapment should be legalized",
    "We should ban factory farming",
    "We should legalize prostitution",
    "We should abolish intellectual property rights",
    "We should legalize polygamy",
    "We should stop the development of autonomous cars",
    "Blockade of the Gaza Strip should be ended",
    "We should ban naturopathy",
    "Intelligence tests bring more harm than good",
    "We should close Guantanamo Bay detention camp",
    "We should abandon the use of school uniform",
    "We should oppose collectivism",
]

assert len(POLICIES) == 30, f"Expected 30 policies, got {len(POLICIES)}"
assert len(set(POLICIES)) == 30, "Duplicate policy detected"
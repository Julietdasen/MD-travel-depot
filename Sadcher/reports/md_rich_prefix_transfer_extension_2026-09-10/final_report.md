# Rich-State Prefix Transfer Pilot

Status: **rich_state_transfer_signal_not_supported**.

Frozen late-stage prefix and prefix-zero rerankers were evaluated over all legal actions in initial stationary states, with exact costs only for selected actions.

| Method | Mean selected-action continuation cost |
|---|---:|
| prefix_zero | 205.8889 |
| prefix | 197.5185 |

Prefix strict win rate versus prefix-zero: 0.3704.
Canonical/reverse prefix choice disagreement: 0.1852.

This is an out-of-distribution transfer pilot, not a retrained rich-state confirmation.

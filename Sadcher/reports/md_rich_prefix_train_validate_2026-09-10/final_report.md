# Rich-State Prefix Train/Validate

Status: **rich_prefix_not_supported_for_confirmation**.

Models were retrained on exact labels from rich initial states and evaluated on independent test states.

| Metric | Prefix-zero | Prefix | Prefix minus zero |
|---|---:|---:|---:|
| Top-1 tolerance-optimal | 0.9583 | 0.7917 | -0.1667 |
| Mean regret | 2.0417 | 5.1667 | 3.1250 |

Maximum prefix order top-1 range: 0.1250.

This package is a development/test validation; confirmation and production regression were not read.

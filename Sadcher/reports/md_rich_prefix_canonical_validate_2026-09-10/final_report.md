# Rich-State Prefix Train/Validate

Status: **rich_prefix_not_supported_for_confirmation**.

Models were retrained on exact labels from rich initial states and evaluated on independent test states.

| Metric | Prefix-zero | Prefix | Prefix minus zero |
|---|---:|---:|---:|
| Top-1 tolerance-optimal | 0.9444 | 0.8333 | -0.1111 |
| Mean regret | 1.8333 | 3.2500 | 1.4167 |

Maximum prefix order top-1 range: 0.0000.

This package is a development/test validation; confirmation and production regression were not read.

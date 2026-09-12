# Ticket 19 Semantic Signal Gate

This controlled semantic probe is not a Ticket 20 benchmark and does not generate production expert data.

Decision: **go**

| Method | Initial agreement | Agreement [95% CI] | Flip [95% CI] | Regret [95% CI] | L/P/R scale | Residual | Saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| physics_only | 0.500 | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.088] | 0.5067 [0.4314, 0.5973] | 0.000/0.243/0.000 | 0.0000 | 0.000 |
| residual_only | 0.375 | 0.838 [0.762, 0.912] | 0.675 [0.520, 0.799] | 0.1001 [0.0458, 0.1611] | 0.197/0.243/0.282 | 0.2822 | 0.125 |
| no_downstream_no_coalition | 0.500 | 0.625 [0.562, 0.688] | 0.250 [0.142, 0.402] | 0.3692 [0.2816, 0.4594] | 0.231/0.243/0.168 | 0.1679 | 0.000 |
| matched_parameter_mlp | 0.500 | 0.938 [0.887, 0.988] | 0.875 [0.739, 0.945] | 0.0211 [0.0013, 0.0459] | 0.247/0.243/0.333 | 0.3333 | 0.156 |
| cross_attention_full | 0.500 | 0.950 [0.900, 0.988] | 0.900 [0.769, 0.960] | 0.0135 [0.0006, 0.0295] | 0.250/0.243/0.325 | 0.3251 | 0.081 |
| eta_unlock_heuristic | 0.625 | 0.625 [0.562, 0.688] | 0.250 [0.142, 0.402] | 0.3106 [0.2252, 0.3903] | 0.000/0.269/0.000 | 0.0000 | 0.000 |

Unique pair templates: 200; train/evaluation overlap: 0; confidence-interval unit: semantic pair.
The signed opportunity prior was disabled, so improvement over initial evaluation comes from fitted parameters.

The exact proxy MIP optimizes only the pre-registered one-step semantic utility. Full scheduling quality, scaling, and latency remain outside this gate.

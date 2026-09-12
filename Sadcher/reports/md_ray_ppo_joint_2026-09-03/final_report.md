# Fixed-Scale MRTA/Sadcher IL -> RL Joint-Action Experiment

Date: 2026-09-03  
Environment: 5 robots x 12 tasks, fixed 64/16/16 seed split  
IL checkpoint: `/data/ZJZ/3D-Foundation-Model/MRTA/Sadcher/runs/md_offline_il_pilot_2026-08-27/training/best_checkpoint.pt`  
Ray: 2.49.2  
Torch/CUDA: 2.4.0+cu121 / CUDA 12.1  
KL weight: 0.02  
Encoder: frozen and verified after optimizer step and checkpoint restore

## Joint-action implementation

The RLlib legacy `ModelV2` now uses a fixed robot-order autoregressive action
distribution. Each prefix recomputes legality from the pair mask, robot skills,
task requirements, current process coalition coverage, and already-used
singleton transport tasks. `env.step()` asserts support membership and simulator
feasibility, then executes the sampled vector unchanged; it does not filter or
repair actions. The same conditional masks are used for sample, log-probability,
entropy, and KL-to-IL.

Focused tests: 20 passed, including singleton transport, process coalition,
100 random joint actions, sample/logp/entropy/gradient finiteness, IL-logit
equivalence, deterministic/stochastic action, restore, frozen encoders, and GPU
forward/backward. The fixed 16-instance validation gate also completed with
success 16/16 and illegal/rejected assignments 0.

## Held-out results

| model seed | IL makespan | RL makespan | relative gain | wins/ties/losses | IL/RL success | IL/RL starvation | IL/RL P95 latency (s) | best iter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5101 | 235.500 | 236.750 | -0.531% | 0/15/1 | 100%/100% | 162.875/163.625 | 0.03916/0.03824 | 10 |
| 5102 | 248.438 | 248.438 | 0.000% | 0/16/0 | 100%/100% | 207.750/207.750 | 0.05187/0.05200 | 10 |
| 5103 | 240.500 | 240.500 | 0.000% | 0/16/0 | 100%/100% | 175.188/174.000 | 0.04878/0.04816 | 20 |
| pooled (48) | 241.479 | 241.896 | -0.173% | 0/47/1 | 100%/100% | 181.938/181.792 | 0.05187/0.05200 | - |

Latency uses one unmeasured warmup inference per episode. IL-first/RL-first
evaluation order alternates by seed index to control warmup/order bias. The
pooled RL/IL P95 ratio is 100.25%, below the 110% safety limit.

Illegal/rejected assignment count was exactly zero for every formal seed and
both policies. Checkpoint restore succeeded for every seed and frozen encoder
verification was true for every seed.

## Decision

**NO-GO.** The joint-action semantic defect is fixed and all safety legality
gates pass, but the pooled makespan gain is -0.173% and the three seeds do not
show a stable positive direction. Seed 5101 also regresses material starvation
and makespan. The experiment does not support claiming effective RL improvement.

Per-seed JSON and metadata are in `runs/md_ray_ppo_joint_seed5101/`,
`runs/md_ray_ppo_joint_seed5102/`, and `runs/md_ray_ppo_joint_seed5103/`;
the pooled summary is `runs/md_ray_ppo_joint_pooled_summary.json`.

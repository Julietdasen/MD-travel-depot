# MRTA/Sadcher IL -> RL Improvement Research

Date: 2026-09-04

## Executive conclusion

The coalition-aware action semantics are now correct, but the current PPO run
does not show a useful policy improvement. The strongest evidence is not simply
the pooled `-0.173%` makespan result: the deterministic policy matched the IL
reference on 47/48 held-out instances, while the three runs made measurable but
small parameter changes only in gates, adapters, and scoring heads. The policy
therefore moved slightly without discovering a better action ordering.

The next experiment should be a small diagnostic matrix, not a 100-iteration
run. First make the learning signal and credit assignment measurable; then
test a larger on-policy batch and a stronger state-value estimator. Only after
one of those changes produces a repeatable positive result should the full
three-seed protocol be repeated.

## Evidence from the completed runs

- Three seeds, 20 iterations, 48 paired held-out instances.
- Pooled IL/RL makespan: `241.479 / 241.896`, relative gain `-0.173%`.
- Paired wins/ties/losses: `0/47/1`.
- Success: `100%/100%`; illegal/rejected assignments: `0/0`.
- Material starvation: `181.938 / 181.792`; P95 latency ratio: `100.25%`.
- Best validation iteration: 10, 10, and 20 for seeds 5101, 5102, and 5103.
- Training used `train_batch_size=256`, default PPO `num_epochs=30`, and
  `minibatch_size=128`. A generated episode is usually about 230-260 steps, so
  one update contains approximately one correlated episode and only two
  minibatches repeated 30 times.
- The first iteration can have `NaN` episode reward/length because the batch is
  a truncated fragment. This is a logging/credit-assignment warning, not a GPU
  NaN, but it confirms that updates are not episode-aligned.

## Main causes and recommended fixes

### 1. PPO data efficiency is currently poor

The current update is too small for a delayed scheduling objective. Repeating
30 epochs over two minibatches encourages fitting one or two trajectories and
does not provide independent action-ordering evidence. Use 2048-4096 environment
steps per update, 4-10 PPO epochs, and a minibatch of 256-512. Keep the learning
rate sweep narrow (`1e-6`, `3e-6`, `1e-5`, `3e-5`) and record approximate KL,
clip fraction, entropy, explained variance, and parameter drift.

This is the first low-risk ablation because it does not alter the simulator or
the policy contract.

### 2. The critic is too weak for this state space

`MDRLlibModel` predicts the value from the mean of 7-dimensional robot features
and 9-dimensional task features. That pooling discards task graph structure,
material readiness, coalition coverage, and the identity of the critical work.
An uninformative critic produces noisy advantages even when the actor is
reasonable.

Reuse the existing typed MD encoder and add a state-value head over a
graph/type-aware pooled representation. The existing local enhancement work
already has a `V(s)` head and remaining-makespan supervision; enable it as an
explicit auxiliary loss, with the label defined as remaining terminal
makespan, not an action value. Track value loss and explained variance.

### 3. Reward shaping has the wrong credit-assignment profile

The time term telescopes to `-T/G` over an episode, while the terminal term adds
`(G-T)/G`. Together they contribute `1 - 2T/G` before completion and starvation
terms. The starvation metric is a schedule-level quantity that can appear as a
large late jump; with penalty `0.02`, a change of 180 contributes about `-3.6`,
comparable to or larger than the whole normalized makespan signal.

Recommended ablations, all on the same seeds and no architecture change:

1. terminal makespan only;
2. potential-based time shaping plus terminal makespan, without starvation;
3. the same reward with a much smaller starvation coefficient (`0.002` or
   `0.005`);
4. separate reporting of latest real-task completion and return-to-exit tail.

The prior replay audit found that the terminal return tail can account for a
large portion of makespan differences. This should be measured, not hidden in a
single scalar reward.

### 4. The current autoregressive policy is legal but only weakly conditional

The distribution conditions later masks on the sampled prefix, so singleton
transport and process skill legality are correct. However, the network emits
all robot logits before sampling; later robots do not receive a learned
embedding of the earlier selected task. The factorization is therefore close to
independent per-robot logits plus a hard support filter.

The next architecture increment should preserve `MDEnhancedSchedulerNetwork`
and add a small prefix decoder (GRU or one-layer Transformer) over robot order:

`robot state -> choose task -> task embedding + selected-task mask -> next robot`.

The decoder must expose the same conditional logits to sample, logp, entropy,
and KL. A fixed 5-robot problem makes this tractable and avoids padding or
multi-scale changes. Before training, test that changing robot 0's selected
transport task changes robot 1's logits even when the pairwise mask is unchanged.

### 5. KL and entropy should be measured as autoregressive quantities

The current implementation applies the reference KL on the legal categorical
distribution for the observed action prefixes. That is aligned with the
executed action, but it is not the exact sequence-level KL expectation over all
prefixes. Entropy is similarly evaluated from the current sampled prefix.

For the fixed five-robot case, compare two implementations:

- Monte Carlo prefix KL using the same sampled prefixes for policy and IL;
- exact dynamic-programming expectation over legal prefixes for a small unit
  fixture, then use the Monte Carlo estimator for full training.

Sweep KL weight `{0, 1e-3, 5e-3, 1e-2, 2e-2}`. A 0.02 reference penalty can be
either too restrictive or irrelevant; without logging actual KL and policy
entropy there is no evidence for choosing it.

There are currently two distinct constraints: RLlib PPO retains its default
old-policy KL loss (`use_kl_loss=true`, initial `kl_coeff=0.2`) while the custom
model adds frozen-IL KL with weight `0.02`. Log both separately. In the ablation,
hold the PPO old-policy KL fixed while sweeping IL KL first; then test disabling
the adaptive PPO KL only if clip fraction and approximate KL show that it is
preventing useful updates. Do not change both constraints in the same cell.

### 6. Checkpoint selection is not aligned with the GO metric

The current best checkpoint is selected by validation shaped reward. The gate is
primarily paired makespan with safety constraints. Add a fixed validation
summary at every tenth iteration and select the best checkpoint lexicographically:

1. illegal/rejected assignment count must be zero;
2. success rate must not fall below IL;
3. minimum mean makespan;
4. material starvation and P95 latency as safety tie-breakers.

Continue reporting shaped reward, but do not use it as the sole model-selection
criterion.

### 7. PPO may be the wrong second-stage optimizer

The project already has high-quality IL demonstrations and a deterministic
simulator. Pure on-policy PPO is expensive because useful differences are rare
and terminal outcomes are delayed. A behavior-regularized offline-to-online
method is a better follow-up if PPO ablations fail:

- AWAC/AWR: weight IL or replay actions by positive advantage while retaining a
  behavior-cloning floor;
- IQL: learn a conservative value/advantage from IL and counterfactual replay,
  then fine-tune online;
- DAgger: query the existing coalition-aware expert on states visited by RL and
  aggregate those states before another short PPO phase.

These methods should be evaluated against the same fixed 64/16/16 split and the
same joint action semantics. They are not permission to replace the production
decoder.

### 8. Decision-focused supervision is likely higher leverage than longer PPO

The existing research notes already identify the key issue: expert reward
matrices can contain arbitrary tie-breaking, while multiple joint actions may
have the same or nearly the same rollout cost. Generate small-state
counterfactuals by forcing each legal joint action and evaluating a fixed
continuation. Train a listwise/pairwise action-regret head or structured loss,
then use PPO only for residual improvement. The acceptance signal should be
top-k near-optimal coverage and rollout regret, not reward-MSE.

## Proposed experiment order

### Phase A: one-seed diagnostics

Use seed 5101 only, 5 iterations per cell, no formal claim:

| cell | batch | PPO epochs | KL | reward |
|---|---:|---:|---:|---|
| A0 | 256 | 30 | 0.02 | current |
| A1 | 2048 | 5 | 0.005 | current |
| A2 | 2048 | 5 | 0.0 | current |
| A3 | 2048 | 5 | 0.005 | no starvation |
| A4 | 2048 | 5 | 0.005 | value auxiliary |

Record action KL to IL, entropy, clip fraction, explained variance, value loss,
policy parameter drift, deterministic action disagreement with IL, real-task
completion time, return-tail time, starvation, and latency.

### Phase B: prefix-conditioning check

On a synthetic fixed fixture and 100 generated states, verify that prefix task
choices affect later logits and that sample/logp/entropy/KL remain finite and
exactly support-consistent. Only then compare prefix decoder versus current
mask-only factorization on A1/A4.

### Phase C: three-seed confirmation

Run the best one or two cells for seeds 5101/5102/5103 with 20 iterations. Use
the original 64/16/16 split, alternating IL/RL rollout order, one warmup call,
and the existing GO gates. Run 100 iterations only if all safety gates remain
zero-regression and at least one cell shows a positive paired makespan direction
on all three seeds.

## Literature pointers

- Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
- Huang and Ontañón, *A Closer Look at Invalid Action Masking in Policy
  Gradient Algorithms*, arXiv:2006.14171.
- Ross, Gordon, and Bagnell, *A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning*, AISTATS 2011 (DAgger).
- Peng et al., *Advantage-Weighted Regression: Simple and Scalable Off-Policy
  Reinforcement Learning*, arXiv:1910.00177.
- Nair et al., *Accelerating Online Reinforcement Learning with Offline
  Datasets*, arXiv:2006.09359 (AWAC).
- Kostrikov et al., *Offline Reinforcement Learning with Implicit Q-Learning*,
  arXiv:2110.06169.
- Bello et al., *Neural Combinatorial Optimization with Reinforcement
  Learning*, arXiv:1611.0994.
- Kool, van Hoof, and Welling, *Attention, Learn to Solve Routing Problems!*,
  arXiv:1803.08475.
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games*, arXiv:2103.01955 (MAPPO).
- Elmachtoub and Grigas, *Smart “Predict, then Optimize”*, Management Science,
  2022 (decision-focused learning).

## Bottom line

The highest-probability next step is **A1/A3/A4**: larger and less repetitive PPO
updates, a critic that sees typed graph state, and a reward ablation that removes
or downweights the late starvation jump. Do not spend the next budget on 100
iterations of the current configuration; it has already demonstrated that it
preserves IL behavior without producing a measurable makespan gain.

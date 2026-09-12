# MD-SADCHER++ 神经策略与低延迟在线推理规格

**状态**：Draft，待用户确认 ticket 粒度
**日期**：2026-08-26
**范围**：MD-SADCHER++ 的离线 MILP 教师、神经 Policy 训练、在线快速 constrained decoder 和端到端延迟评估

## Problem Statement

当前 SADCHER 用 GAT 和 Transformer 为 robot-task pair 预测 reward，再交给 relaxed bipartite matching 完成联合分配。这个结构可以学习任务优先级，但当前 MD 扩展还缺少一条清晰的训练/推理边界：

- MILP/Gurobi 适合为小规模问题生成高质量 expert schedule，但不应在每个在线 decision point 被调用；
- 当前 matcher 是带 process coalition skill coverage 的 0-1 MIP，不是标准一对一 permutation matching；
- 直接使用 Sinkhorn/Gumbel-Sinkhorn 会错误表达 process coalition；
- 当前神经 Policy 还没有 Cross-Attention、transport-specific score、downstream unlock value 或在线 learned decoder；
- 只报告 neural forward time 不能代表完整调度延迟。

用户需要一个神经学习成分更充分、同时满足 MD hard feasibility、process coalition 和低延迟约束的完整 Policy 流程。

## Solution

采用 teacher-student 结构：

1. 离线阶段用 Gurobi/MILP 在小规模 MD instances 上求解 optimal 或 time-limited feasible schedule，回放每个 decision point，保存 state、typed graph、hard mask、expert assignment、makespan 和失败原因。
2. 训练阶段读取 MD Expert Dataset，由 legacy-preserving GAT/Transformer、可选一层双向 Robot-Task Cross-Attention 和 transport-aware scoring 学习合法动作之间的优先级。
3. Transport score 分为固定或单调受限的 physics utility 与有界 learned downstream unlock residual。网络不能学习 readiness、capacity legality 或 transport duration formula。
4. 在线阶段默认使用 neural score、centralized hard mask、learned constrained decoder 和快速 legality repair；不在正常路径调用 MILP。
5. MIP 保留为离线 oracle、benchmark reference，以及低置信度或 repair 失败时的显式 fallback。
6. 所有方法报告完整端到端 latency，而不是只报告神经 forward；至少包含 P50/P95/P99、success rate、illegal assignment count、makespan regret 和 fallback rate。

## User Stories

1. As a scheduling researcher, I want MILP to generate MD expert schedules offline, so that neural training has high-quality labels without making online inference slow.
2. As a scheduling researcher, I want each expert decision point to include the observed MD state, so that the policy learns state-to-action behavior rather than only final schedules.
3. As a scheduling researcher, I want optimal, time-limited feasible, and heuristic expert labels distinguished, so that label quality is auditable.
4. As a scheduling researcher, I want task/instance-level dataset splits, so that decision samples from one instance cannot leak across train and test.
5. As a policy engineer, I want legacy process features and checkpoints to remain loadable, so that MD extensions can be compared with original SADCHER.
6. As a policy engineer, I want MD metadata added through separate adapters, so that enabling MD features does not change the legacy feature tensor contract.
7. As a policy engineer, I want normal precedence and material-delivery edges distinguished, so that the encoder can tell process order from material unlock.
8. As a policy engineer, I want robot and task tokens to exchange information, so that a pair score can account for competing robots and tasks.
9. As a policy engineer, I want transport ETA, capacity margin, load, loaded speed, and robot availability represented explicitly, so that the network does not have to rediscover deterministic physics.
10. As a policy engineer, I want each carry to gather its downstream process representation, so that the policy can estimate the value of unlocking that process.
11. As a policy engineer, I want the learned residual bounded around a physics prior, so that training cannot silently discard reliable transport timing.
12. As a policy engineer, I want process tasks to retain the original scorer, so that transport improvements do not hide regressions in process coalition scheduling.
13. As a policy engineer, I want the final neural output to remain a robot-task score matrix, so that existing scheduling interfaces can be reused.
14. As a policy engineer, I want a single centralized hard-feasibility mask, so that type mismatch, capacity failure, readiness failure, occupancy, and material blocking are never learned as soft preferences.
15. As a policy engineer, I want blocked process tasks visible as context but excluded as actions, so that the network can reason about future value without selecting illegal work.
16. As an online scheduler, I want the normal inference path to avoid MILP, so that decision latency remains low as the number of decision points grows.
17. As an online scheduler, I want a learned constrained decoder to preserve process skill coverage, so that fast inference does not break multi-robot process coalitions.
18. As an online scheduler, I want transport tasks to remain singleton assignments in the MVP, so that one transport job cannot be silently assigned to multiple robots.
19. As an online scheduler, I want fast repair to reject or repair invalid neural assignments, so that the output is legal before it reaches the simulator.
20. As an online scheduler, I want MIP fallback to be explicit and measurable, so that rare slow paths cannot be hidden inside average latency.
21. As a researcher, I want an optional state-value head to predict remaining makespan, so that shared representations can receive a long-horizon auxiliary signal.
22. As a researcher, I want assignment-conditioned value separated from state value, so that V(s) is not incorrectly claimed to rank candidate assignments.
23. As a researcher, I want decoder-aware structured loss to reuse the coalition feasible set, so that training can optimize assignment quality without pretending the MIP is differentiable.
24. As a researcher, I want standard Sinkhorn to be rejected for this MVP, so that the loss does not impose invalid one-to-one coalition semantics.
25. As an evaluator, I want paired counterfactual instances, so that each policy module is tested on the semantic factor it claims to use.
26. As an evaluator, I want matched decoder and hard-mask settings across ablations, so that gains cannot be attributed to changed legality.
27. As an evaluator, I want success and failure reported separately from makespan, so that deadlock and timeout are not hidden by synthetic completion times.
28. As an evaluator, I want legacy process-only regression reported, so that MD gains cannot be obtained by degrading original SADCHER behavior.
29. As an evaluator, I want full latency broken into encoder, scorer, decoder, repair, and fallback components, so that a neural speed claim is end-to-end verifiable.
30. As a paper author, I want claims restricted to measured behavior, so that physics prior, Cross-Attention, value auxiliary loss, and structured loss are not overclaimed as universal improvements.

## Implementation Decisions

- The system has two explicit modes: offline teacher generation and online learned inference.
- Offline teacher generation uses the existing MD simulator, Gurobi/MILP oracle, structured execution records, and the experiment protocol. Every sample records instance identity, seed, split, current state, real task IDs, typed graph, downstream links, hard-feasibility mask, expert assignment, expert quality class, and remaining makespan when available.
- The legacy feature path remains unchanged when MD metadata is absent or all MD flags are disabled. New robot metadata, task metadata, pair features, and graph relations enter through optional adapters.
- The encoder retains robot/task GAT and Transformer blocks. A first neural-enhancement version may add one bidirectional Robot-Task Cross-Attention block. Context visibility masks are separate from action feasibility masks; blocked future tasks remain visible as context.
- The scorer keeps the process path compatible with legacy SADCHER. Transport scoring consumes explicit ETA, load, capacity margin, loaded speed, downstream representation, unfinished predecessor counts, coalition availability estimates, slack, and scarcity context.
- Transport utility is decomposed into a fixed or monotonic physics utility plus a bounded learned residual. The residual is zero-initialized or otherwise calibrated and clipped. The network never predicts legality or replaces deterministic duration formulas.
- The policy output remains a robot-task score matrix, with idle handled according to the existing contract. The decoder receives the centralized hard-feasibility mask and must preserve robot availability, type matching, process skill coverage, transport singleton semantics, readiness, and occupancy.
- Online normal path uses a learned constrained decoder and fast repair. The exact MIP decoder is not the default online path. It remains an offline oracle, benchmark reference, and explicit fallback.
- A learned decoder may be sequential or greedy-with-learned-priorities, but it must be independently benchmarked against a deterministic masked greedy decoder and exact MIP on the same states.
- Standard Sinkhorn/Gumbel-Sinkhorn is excluded from the main decoder because process coalition assignment is not doubly stochastic. Decoder-aware training may use structured hinge or loss-augmented inference over the existing coalition feasible set; solver outputs are treated as constants for the backward pass.
- The first value head is V(s), trained on normalized remaining makespan and pooled with attention or type-conditioned pooling. Q(s,A) is a separate later experiment and cannot be inferred from V(s).
- Latency is measured at the complete decision boundary, including preprocessing, neural forward, decoder, repair, and fallback. P50/P95/P99 and fallback rate are mandatory.
- Existing Tickets 17–22 remain the baseline dataset, scorer, smoke, comparison, scaling, and reproducibility path. New tickets extend that path and do not modify or close the earlier parent tickets.

## Testing Decisions

- Tests validate observable behavior at the highest practical seams: dataset sample round-trip, policy forward contract, decoder assignment contract, simulator transition, and protocol result envelope.
- Legacy checkpoint tests must show flags-off output and process-only rollout remain equivalent within a documented tolerance.
- Neural smoke tests must cover random finite forward/backward, zero-distance finite behavior, task/robot permutation consistency, optional metadata absence, and independent ablation switches.
- Hard-mask tests must show illegal robot-task pairs cannot be selected, blocked process tasks remain context-visible but action-ineligible, and transport assignments remain singleton.
- Decoder tests must compare learned decoder output against mask and coalition constraints without testing private layer internals.
- Offline teacher tests must replay stored expert actions through the MD simulator and reproduce terminal success/failure and makespan metadata.
- Structured-loss tests must verify loss-augmented inference uses the same feasible set and produces finite score gradients; they must not assert gradients through PuLP/CBC.
- Value-head tests must distinguish V(s) calibration from assignment ranking and report the correct claim.
- Latency tests must record component timings and tail latency under increasing robot/task counts; a neural module is not accepted on forward time alone.
- Evaluation tests reuse prior MD behavior scenarios, Greedy strategy diagnostics, Gurobi oracle checks, and experiment protocol statistics. The same paired instances and seed list must be used across methods.
- The known zero-distance normalization regression must be fixed before new neural modules are used for performance claims.

### Task-oriented instance families

Changing only the random seed does not test structural generalization. The
default 12-task generator configuration remains the historical balanced
baseline, while new experiments sample and report the following instance
families separately:

- process_scarce: fewer process robots, emphasizing coalition and idle choices;
- transport_bottleneck: one capacity-tight, slower carrier and more transport work;
- dependency_deep: a deeper and denser process DAG;
- mixed_hard: combined process, transport, and dependency pressure;
- scale_medium: 18 tasks and 7 robots for neural generalization and end-to-end latency.

All current profiles retain three skill channels to match the neural input
contract. Train, development, and test use disjoint seeds within every family.
Every compared policy receives the same profile/seed manifest. Results are
reported per family plus an equal-weight macro average, so a method cannot hide
a regression on one task regime behind many easy balanced instances.

Do not select every state only because it has 24--96 complete actions. That
filter removes concentrated transport cases as well as difficult high-branching
cases. Record action-count buckets explicitly and report oracle timeouts instead.
The scale_medium family is primarily a neural latency/generalization test:
initial action enumeration reached 2518 actions and 14.7 seconds in a five-seed
Mac CPU smoke check, so exhaustive continuation labels are not required for
every scale_medium state.

Existing prefix checkpoints were trained only on the balanced family. Shape
compatibility with another profile is not evidence of scheduling quality:
profile-aware checkpoints must be trained and evaluated before cross-family
performance claims are made. Decoder latency is also reported per profile
because constrained-search cost changes with branching.

## Out of Scope

- Online MILP as the normal decoder path.
- Standard Sinkhorn/Gumbel-Sinkhorn replacing the coalition decoder.
- Neural prediction of readiness, capacity legality, task type legality, predecessor legality, or transport duration.
- Full autoregressive route generation, MAPPO as the primary policy, or decentralized communication claims.
- Cooperative transport, capacity aggregation, synchronization waiting, relay, multi-trip, preemption, inventory, battery, collision/path planning, and low-level control.
- Rewriting the legacy SADCHER decoder before a learned decoder baseline is measured.
- Claiming universal real-time performance without end-to-end latency evidence.
- Adding Cross-Attention, structured loss, Value head, and PPO simultaneously without isolated ablations.

## Further Notes

The recommended order is:

1. Complete the existing MD Expert Dataset ticket.
2. Complete the base MD scorer and ablation ticket.
3. Add the one-layer Cross-Attention and bounded physics-residual scorer as an isolated extension.
4. Add the learned constrained decoder and fast repair.
5. Run end-to-end online latency and quality comparisons against masked greedy and exact MIP.
6. Only after the MVP is stable, pilot structured loss and the V(s) auxiliary head.
7. Consider PPO or Q(s,A) only after the offline and online contracts have independent evidence.

The central paper claim should be narrow: the neural Policy learns downstream coalition opportunity value that deterministic transport physics cannot express, while exact hard constraints remain outside the learned model.

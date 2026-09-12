# ADR-002: Transport State Machine and Timing

- Status: accepted
- Scope: MD-SADCHER++ MVP

## Decision

Transport uses explicit phases:

```text
WAITING -> TO_PICKUP -> LOADING -> TO_DELIVERY -> UNLOADING -> COMPLETE
```

The TransportRobot is busy and non-preemptive from assignment through completion. Task start is loading start; task completion is unloading completion. Movement in the discrete simulator uses `ceil(distance / speed)`.

Each execution record stores assignment, pickup arrival, loading start/completion, delivery arrival, completion, and empty-travel, loading, loaded-transport, unloading, service, and occupied durations.

## Rationale

A single aggregate duration cannot represent two locations or support reliable transport-utilization and material-arrival metrics. Explicit phases preserve verifiable semantics without adding low-level path planning.

## Out of scope

Preemption, synchronization waiting, relay, multi-trip, and cooperative transport are excluded from MVP.

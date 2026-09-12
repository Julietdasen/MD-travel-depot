# Accepted Design Decisions

This is an incremental supplement to the existing MD-SADCHER guides. It does not replace the existing guide files; implementation follows this document and the ADRs as the latest decisions.

## Frozen

- Real tasks are only `PROCESS` and `TRANSPORT`.
- Real robots are mutually exclusive `PROCESS_ROBOT` and `TRANSPORT_ROBOT`.
- ProcessRobot uses original SADCHER skill coalitions; TransportRobot uses transport capability, capacity, unloaded speed, and loaded speed.
- MVP implements solo transport only, with no transport coalition, capacity aggregation, synchronization waiting, or preemption.
- Each transport unlocks exactly one process; transport has no normal predecessor; delivery equals the downstream process location.
- Transport uses an explicit state machine and structured execution records.
- Centralized hard feasibility and stable reason codes are shared by all schedulers.
- Legacy mode remains runnable with frozen regression fixtures; internal structures may change.
- The official dataset may directly change loaders and related structures, but runtime domain and MD state-machine semantics remain stable.
- English and Chinese guides are maintained together; the external dataset schema remains provisional until inspection.

## Deferred

- `SOLO`, `SMALL_COALITION`, and `LARGE_COALITION` cooperative transport.
- Learned mode selection after a rule-based implementation.
- Typed-edge encoder, downstream-aware scoring, and new neural features.
- Whether the official dataset travel matrix can replace coordinate-based MD distances.

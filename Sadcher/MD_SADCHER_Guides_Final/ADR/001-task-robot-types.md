# ADR-001: Task and Robot Types

- Status: accepted
- Scope: MD-SADCHER++ MVP

## Decision

Real tasks are `PROCESS` or `TRANSPORT`; real robots are mutually exclusive `PROCESS_ROBOT` or `TRANSPORT_ROBOT`.

Process tasks retain original SADCHER skill requirements and dynamic coalitions. Transport tasks use transport capability and capacity, and are executed by TransportRobot only. MVP forbids hybrid robots, transport coalitions, and capacity aggregation.

Each transport unlocks exactly one process in a strict one-to-one relation. Transport tasks have no normal predecessors; material is available at `t=0`. Delivery location must equal the downstream process location.

## Rationale

Separating operation skills from transport capability prevents transport from being treated as an ordinary process skill. The one-to-one, no-inventory relation expresses the minimum material-delivery semantics without hidden material nodes.

## Future

Transport may later support fixed `SOLO`, `SMALL_COALITION`, and `LARGE_COALITION` modes composed only of TransportRobots. Mode selection starts rule-based before any learned selector is considered.

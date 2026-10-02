# Serverless Deployment Roadmap

The `serverless/` directory contains abstract interfaces that prepare H-SFP for
deployment on serverless infrastructure (AWS Lambda, Google Cloud Run, etc.).

## Why H-SFP is serverless-friendly

Clients already send **prototypes + distributions** (small tensors), not full model
weights. Each round is stateless — the aggregator just needs the current prototypes.
This maps cleanly onto function-as-a-service: each tier is a separate function
invocation triggered by an event.

## Current state

All tiers communicate in-process via shared Python objects.
`serverless/adapters/local.py` (`LocalCommunicationBackend`) makes this explicit.

## Steps to go serverless

| Step | What | Where |
|------|------|--------|
| 1 | Implement `CommunicationBackend` for cloud transport (SQS / Pub-Sub / Redis) | `serverless/adapters/<cloud>.py` |
| 2 | Refactor `classification/H-SFP/hierarchy.py` to call `backend.send/receive` | `hierarchy.py` |
| 3 | Wrap each tier as a standalone handler function | `serverless/handlers/` (create) |
| 4 | Store round state (prototypes, weights) in object storage (S3 / GCS) | `serverless/state/` (create) |
| 5 | Add a round coordinator / orchestrator | `serverless/orchestrator.py` (create) |

## Key abstractions (already defined)

- `serverless/interfaces/communication.py` — `CommunicationBackend` + `Payload`
  - `Payload.to_bytes()` / `from_bytes()` for serialisation
  - `Payload.size_mb()` to monitor communication cost

- `serverless/interfaces/aggregator.py` — `AggregatorBackend`
  - `aggregate(payloads)` — merge prototypes from multiple clients
  - `is_ready(received, expected)` — sync vs async aggregation policy

## Swap example (future)

```python
# Current (local)
from serverless.adapters.local import LocalCommunicationBackend
backend = LocalCommunicationBackend()

# Future (AWS SQS)
from serverless.adapters.aws_sqs import SQSCommunicationBackend
backend = SQSCommunicationBackend(queue_url="https://sqs.us-east-1.amazonaws.com/...")
```

No changes needed to `prototype.py`, `ssl.py`, or the model code.

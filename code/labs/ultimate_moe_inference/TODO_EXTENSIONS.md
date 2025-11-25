# Future Extensions

This document lists production serving features that are out of scope for the
performance optimization focus of this lab, but would be valuable additions
for a production deployment.

## SLO & Observability

- [ ] **SLO Compliance Tracking**: Track % of requests meeting latency targets
  - Define SLO thresholds (e.g., p99 TTFT < 100ms, p99 TPOT < 20ms)
  - Alert when SLO violations exceed threshold
  
- [ ] **Request Tracing**: End-to-end trace through each component
  - OpenTelemetry integration
  - Span tracking: tokenization → prefill → decode → detokenization
  - Distributed tracing across multi-node setups

## Scheduling & QoS

- [ ] **Request Preemption**: Pause/resume requests for priority scheduling
  - Save KV cache state for preempted requests
  - Priority queues (high/medium/low)
  - Fair scheduling across users

- [ ] **Rate Limiting**: Per-user/per-tenant request limits
  - Token bucket algorithm
  - Graceful degradation under load

## Multi-Model & Adaptation

- [ ] **LoRA Hot-Swapping**: Switch adapters without reloading base model
  - Adapter registry with lazy loading
  - Per-request adapter selection
  - Memory-efficient adapter storage

- [ ] **Multi-Model Router**: Route requests to appropriate model size
  - Complexity-based routing (simple queries → smaller model)
  - Cost-aware routing
  - A/B testing support

## Advanced Generation

- [ ] **Structured Output / JSON Mode**: Grammar-constrained decoding
  - JSON schema validation during generation
  - Regex/grammar constraints
  - Function calling with tool definitions

- [ ] **Embeddings Endpoint**: Not just generation, also embeddings
  - Mean pooling, last token, etc.
  - Batch embedding support

## Cost & Efficiency

- [ ] **Cost Calculator**: $/million tokens based on power + hardware
  - Real-time cost tracking per request
  - Cost attribution per user/tenant
  - TCO analysis tools

- [ ] **Degradation Alerts**: Detect when performance drops vs baseline
  - Automated regression detection
  - Performance baseline tracking
  - Alerting integration (PagerDuty, Slack)

## Deployment

- [ ] **Kubernetes Operator**: Deploy and scale inference clusters
  - HPA based on request queue depth
  - Rolling updates with zero downtime
  - GPU node affinity

- [ ] **Model Versioning**: A/B testing and gradual rollout
  - Traffic splitting
  - Canary deployments
  - Rollback support

---

## Contributing

If you implement any of these features, please submit a PR! Each extension
should include:

1. Implementation in `components/` or new directory
2. Tests in `tests/`
3. Documentation update
4. Benchmark showing performance impact



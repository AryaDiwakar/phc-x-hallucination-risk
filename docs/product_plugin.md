# PHC-X Product Track

PHC-X is also a local-first plugin product, not only an experiment:

```text
browser chat composer -> MV3 extension -> local PHC-X API -> policy decision
                                      -> generate / retrieve / clarify / abstain
```

## Product Behaviour

- The extension intercepts Enter/send events without sending conversation data
  to a hosted service.
- The API runs locally at `127.0.0.1`, loads the trained checkpoint, and logs
  decisions append-only for review.
- Low-risk messages proceed normally.
- Retrieval recommendations show the user the risk factors and evidence path.
- Clarify/abstain decisions require an explicit `send anyway` override.
- If the local service is unavailable, the extension fails open after a short
  timeout rather than trapping the user.
- Dashboard controls expose policy thresholds, action distribution, benchmark
  gates, and feedback.

## Product Acceptance

```bash
phc serve
# open extension/demo/chat.html with the MV3 extension loaded
```

The acceptance path checks health, version, dashboard static assets, single
analysis, batch analysis, policy GET/PUT, results, decision logging, extension
JavaScript syntax, and a benign first-turn greeting (`hi -> LOW/GENERATE`).

## Research/Product Boundary

The product can ship as a local research preview now. Production deployment
still needs privacy review, browser-specific integration tests, rate limiting,
secure policy storage, telemetry consent, and human validation of false
blocks. The external evaluation work is intentionally separate from product
telemetry so user data is not silently used as research labels.

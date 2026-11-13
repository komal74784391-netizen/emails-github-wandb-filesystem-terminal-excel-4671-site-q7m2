# moe-routing-viz

Small plotting toolkit that we use to inspect routing statistics across
mixture-of-experts models (expert utilisation, routing entropy and token
overflow). It is a general diagnostics helper, not the implementation of any
specific publication.

## Layout

- `src/metrics.py` - routing entropy and utilisation metrics
- `scripts/plot_router_entropy.py` - command-line plotting helper

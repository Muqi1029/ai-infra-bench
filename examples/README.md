# Examples

Run an SLO search against an OpenAI-compatible endpoint:

```bash
aib slo examples/slo.yaml -o slo-result.yaml
```

Edit `slo.yaml` to set the endpoint, request payload, search range, and SLO
conditions. The search uses the same request implementation as `aib bench`.

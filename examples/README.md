# General Bench
`general_bench` is designed to evaluate how a specific deployment performs under multiple client configurations.
For example, if you deploy `Qwen3-30B-A3B-FP8` on a particular GPU, you can use this function to measure its performance at different request rates, such as `10` or `20`.

---

## Arguments

1. **server_cmds (List[str])**
   A list of deployment commands, where each string represents one server configuration.

2. **client_cmds (List[List[str]])**
   A list of client command lists. Each sublist contains multiple client settings corresponding to the server at the same index in `server_cmds`.

3. **input_features (List[str])**
   The key client parameters you vary during benchmarking (e.g., `request_rate`, `max_concurrency`). These are highlighted in results to clarify what was changed.

4. **metrics (List[str])**
   The performance metrics to record and display in tables and plots (e.g., `latency`, `throughput`).

5. **labels (List[str])**
   Labels for each server configuration. Used as titles in output tables and plots.

6. **host (str)**
   The host name or identifier where the benchmark is run.

7. **port (Union[str, int])**
   The service port of the deployment under test.

8. **output_dir (str)**
   The directory where all output—tables, plots, and generated files—will be stored.


# Compare Bench
Coming soon

# SLO Bench
Coming soon

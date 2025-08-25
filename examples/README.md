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


# Cmp Bench
`cmp_bench` is designed to compare multiple deployment options under identical client settings.
For example, if you want to evaluate the performance gains from enabling `cuda-graph`, you can launch one server with `cuda-graph` enabled and another without it. This provides a clear, quantitative comparison on your specific hardware.

---
Arguments:

1. **server_cmds (List[str])**
    A list of server launch commands, each representing a different deployment configuration to compare.

2. **client_cmds (List[str])**
    A list of client commands that are run against all server configurations for consistent benchmarking.

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

## SLO Bench

`slo_bench` identifies the most demanding client settings (e.g., maximum concurrency) that still satisfy the defined Service Level Objectives (SLOs). This helps assess whether a given deployment can handle real-world workloads while meeting performance requirements. The core algorithm used in `slo_bench` is **binary search**.

### Arguments

1. **server_cmds (List[str])**
   A list of server launch commands, where each string specifies one deployment configuration.

2. **client_cmds (List[str])**
   A list of base client commands. The full command is constructed by appending parameters such as `--request-rate` and `--max-concurrency`.

3. **request_rates (List[Tuple[int, int]])**
   A list of `(low, high)` integer pairs, defining the search range for request rates (or maximum concurrency) used during binary search.

4. **input_features (List[str])**
   Client parameters varied during benchmarking. For this bench, valid options are `request_rate` and `max_concurrency`. These are highlighted in results to indicate what changed.

5. **metrics (List[str])**
   The performance metrics to collect and display in tables and plots (e.g., `latency`, `throughput`).

6. **labels (List[str])**
   Human-readable labels for each server configuration, used as titles in output tables and plots.

7. **host (str)**
   The host name or identifier where the benchmark is executed.

8. **port (Union[str, int])**
   The service port of the deployment being tested.

9. **check_slo (Callable[[Dict], bool])**
   A function that evaluates whether the collected metrics satisfy the SLO. It should return `True` if the SLO is met and `False` otherwise.

10. **output_dir (str)**
    The directory where all benchmark results—including tables, plots, and generated files—will be saved.

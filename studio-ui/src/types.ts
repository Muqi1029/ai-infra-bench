export interface PlatformInfo {
  version?: string;
  prometheus_enabled?: boolean;
  prometheus_port?: number | null;
  grafana_enabled?: boolean;
  grafana_port?: number | null;
}

export interface TelemetryPoint {
  ts?: number;
  gen_throughput_tps?: number | null;
  kv_cache_pct?: number | null;
  cache_hit_pct?: number | null;
  running_reqs?: number | null;
  waiting_reqs?: number | null;
}

export interface LiveTarget {
  label: string;
  base_url: string;
  metrics_url?: string;
  mode: "pull" | "push" | string;
  session_id?: string;
  session_name?: string;
  labels?: Record<string, string>;
  server_info?: {
    reachable?: boolean;
    engine?: string;
    model?: string;
    launch_command?: string;
    server_args?: Record<string, unknown>;
  };
  hardware_info?: {
    summary?: string;
  };
  latest?: TelemetryPoint;
  series?: TelemetryPoint[];
}

export interface BenchRun {
  run_id?: string;
  subcommand: string;
  workload?: string;
  mode?: string;
  max_concurrency?: number | null;
  peak_concurrency?: number | null;
  request_rate?: number | null;
  summary?: {
    output_throughput_tps?: number;
    total_token_throughput_tps?: number;
    request_goodput_rps?: number;
    request_goodput_pct?: number;
    request_throughput_rps?: number;
    global_cache_ratio_pct?: number;
  };
  metrics?: Record<string, Record<string, number>>;
  telemetry?: {
    summary?: {
      kv_cache_peak_pct?: number;
      max_waiting_reqs?: number;
    };
  };
}

export interface CorrectnessRun {
  eval_id: string;
  eval_type: "dataset" | "logits" | "hidden_states" | string;
  suite_name: string;
  passed?: boolean;
  correct_rate?: number | null;
  truncated_rate?: number | null;
  ref_base_url?: string;
  summary?: Record<string, any>;
}

export interface ServiceSession {
  session_id: string;
  name: string;
  base_url: string;
  status: "online" | "offline" | string;
  mode: string;
  is_baseline?: boolean;
  bench_count: number;
  eval_count: number;
  peak_output_tps?: number | null;
  bench_runs?: BenchRun[];
  correctness_runs?: CorrectnessRun[];
}

export interface ConfigDiff {
  key: string;
  baseline: unknown;
  candidate: unknown;
}

export interface CompareResponse {
  sessions: ServiceSession[];
  config_diffs: ConfigDiff[];
}

export interface JobRecord {
  job_id: string;
  session_id: string;
  job_type: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled" | string;
  progress?: {
    stage?: string;
    pct?: number;
    live_tps?: number | null;
  };
  logs_text?: string;
  error_message?: string;
}

export interface EvalCasesDoc {
  eval_id: string;
  eval_type: "dataset" | "logits" | "hidden_states" | string;
  suite_name?: string;
  ref_base_url?: string;
  summary?: Record<string, any>;
  cases?: any[];
}

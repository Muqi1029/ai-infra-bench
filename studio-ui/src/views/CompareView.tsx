import * as React from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  GitCompareArrows,
  Pin,
  Play,
  Trash2,
  XCircle,
} from "lucide-react";
import type { BenchRun, CompareResponse, ServiceSession } from "../types";
import { cn, fmtNum } from "../lib/utils";
import { Button } from "../components/ui/button";
import { Label } from "../components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Badge } from "../components/ui/sheet";

interface CompareViewProps {
  sessions: ServiceSession[];
  baseSessionId: string | null;
  candSessionId: string | null;
  compareData: CompareResponse | null;
  onSelectBase: (id: string) => void;
  onSelectCand: (id: string) => void;
  onPinBaseline: () => Promise<void>;
  onDeleteSession: (session: ServiceSession) => void;
  onJumpToLaunch: () => void;
}

export function CompareView({
  sessions,
  baseSessionId,
  candSessionId,
  compareData,
  onSelectBase,
  onSelectCand,
  onPinBaseline,
  onDeleteSession,
  onJumpToLaunch,
}: CompareViewProps) {
  const [xMode, setXMode] = React.useState<"auto" | "concurrency" | "qps">("auto");
  const [statKey, setStatKey] = React.useState<"mean" | "p50" | "p95" | "p99">("mean");
  const [cmpMetric, setCmpMetric] = React.useState("tpot_ms");
  const [pinBusy, setPinBusy] = React.useState(false);

  const handlePin = async () => {
    setPinBusy(true);
    try {
      await onPinBaseline();
    } finally {
      setPinBusy(false);
    }
  };

  if (!sessions.length || !compareData?.sessions?.length) {
    return (
      <div className="rounded-[12px] border border-dashed border-[var(--control-border)] bg-[var(--surface)] px-6 py-12 text-center">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-[10px] bg-[var(--surface-subtle)] text-[var(--brand)]">
          <GitCompareArrows className="w-5 h-5" strokeWidth={1.75} aria-hidden="true" />
        </div>
        <h2 className="text-[15px] font-bold text-[var(--text)] m-0">暂无持久化压测或校验会话</h2>
        <p className="text-[13px] text-[var(--text-muted)] max-w-lg mx-auto mt-1.5 mb-4">
          前往「发起压测」执行基准测试，或在命令行使用 <code className="tabular-mono px-1.5 py-0.5 rounded bg-[var(--surface-subtle)]">--report-to {window.location.origin}</code> 上报结果后即可在此对比 Pareto 前沿与配置差异。
        </p>
        <Button variant="primary" size="sm" onClick={onJumpToLaunch}>
          <Play className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
          <span>前往发起压测</span>
        </Button>
      </div>
    );
  }

  const base = compareData.sessions[0];
  const cand = compareData.sessions[1] || base;
  const sameSession = base.session_id === cand.session_id;
  const diffs = compareData.config_diffs || [];

  const bRuns = (base.bench_runs || []).filter((r) => r.subcommand === "bench");
  const cRuns = (cand.bench_runs || []).filter((r) => r.subcommand === "bench");

  const maxOf = (arr: BenchRun[], fn: (r: BenchRun) => number | undefined) => {
    const vals = arr.map(fn).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    return vals.length ? Math.max(...vals) : null;
  };
  const minOf = (arr: BenchRun[], fn: (r: BenchRun) => number | undefined) => {
    const vals = arr.map(fn).filter((v): v is number => typeof v === "number" && v > 0);
    return vals.length ? Math.min(...vals) : null;
  };

  const bPeakTps = maxOf(bRuns, (r) => r.summary?.output_throughput_tps);
  const cPeakTps = maxOf(cRuns, (r) => r.summary?.output_throughput_tps);
  const bMinTpot = minOf(bRuns, (r) => r.metrics?.tpot_ms?.[statKey]);
  const cMinTpot = minOf(cRuns, (r) => r.metrics?.tpot_ms?.[statKey]);

  const cDsEvals = (cand.correctness_runs || []).filter((e) => e.eval_type === "dataset");
  const cLogits = [...(cand.correctness_runs || [])].reverse().find((e) => e.eval_type === "logits");

  const runLoadX = (r: BenchRun) => {
    if (xMode === "qps") return r.request_rate ?? r.max_concurrency ?? 1;
    if (xMode === "concurrency") return r.max_concurrency ?? r.peak_concurrency ?? r.request_rate ?? 1;
    return r.mode === "qps" ? r.request_rate ?? 1 : r.max_concurrency ?? r.request_rate ?? 1;
  };

  const extractMetric = (r: BenchRun, key: string): number | undefined => {
    const s = r.summary || {};
    const m = r.metrics || {};
    if (key === "output_tps") return s.output_throughput_tps;
    if (key === "total_tps") return s.total_token_throughput_tps;
    if (key === "goodput_rps") return s.request_goodput_rps ?? s.request_throughput_rps;
    if (key === "cache_pct") return s.global_cache_ratio_pct;
    return m[key]?.[statKey] ?? m[key]?.mean;
  };

  // Merge X points for dual-series Recharts
  const buildMergedLoadData = (yExtractor: (r: BenchRun) => number | undefined) => {
    const map = new Map<number, { x: number; base?: number; cand?: number }>();
    if (!sameSession) {
      bRuns.forEach((r) => {
        const y = yExtractor(r);
        if (typeof y === "number") {
          const x = runLoadX(r);
          map.set(x, { ...(map.get(x) || { x }), base: Number(y.toFixed(2)) });
        }
      });
    }
    cRuns.forEach((r) => {
      const y = yExtractor(r);
      if (typeof y === "number") {
        const x = runLoadX(r);
        map.set(x, { ...(map.get(x) || { x }), cand: Number(y.toFixed(2)) });
      }
    });
    return [...map.values()].sort((a, b) => a.x - b.x);
  };

  const paretoData = React.useMemo(() => {
    const pts: Array<{ speed: number; baseTps?: number; candTps?: number }> = [];
    if (!sameSession) {
      bRuns.forEach((r) => {
        const tpot = r.metrics?.tpot_ms?.[statKey] ?? r.metrics?.tpot_ms?.mean;
        const tps = r.summary?.output_throughput_tps;
        if (tpot && tpot > 0 && typeof tps === "number") {
          pts.push({ speed: Number((1000 / tpot).toFixed(2)), baseTps: Number(tps.toFixed(2)) });
        }
      });
    }
    cRuns.forEach((r) => {
      const tpot = r.metrics?.tpot_ms?.[statKey] ?? r.metrics?.tpot_ms?.mean;
      const tps = r.summary?.output_throughput_tps;
      if (tpot && tpot > 0 && typeof tps === "number") {
        pts.push({ speed: Number((1000 / tpot).toFixed(2)), candTps: Number(tps.toFixed(2)) });
      }
    });
    return pts.sort((a, b) => a.speed - b.speed);
  }, [bRuns, cRuns, sameSession, statKey]);

  const metricSweepData = buildMergedLoadData((r) => extractMetric(r, cmpMetric));
  const kvSweepData = buildMergedLoadData((r) => r.telemetry?.summary?.kv_cache_peak_pct);

  const renderDelta = (nv: number | null, ov: number | null, lowerBetter = false) => {
    if (nv == null || ov == null || ov === 0 || sameSession) {
      return <span className="text-xs text-[var(--text-muted)] tabular-mono">Base: {ov != null ? ov.toFixed(1) : "—"}</span>;
    }
    const pct = ((nv - ov) / ov) * 100;
    const good = lowerBetter ? pct <= 0 : pct >= 0;
    return (
      <span
        className={cn(
          "inline-flex items-center gap-0.5 text-xs font-semibold tabular-mono",
          good ? "text-[var(--emerald)]" : "text-[var(--crimson)]"
        )}
      >
        {pct >= 0 ? (
          <ArrowUpRight className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
        ) : (
          <ArrowDownRight className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
        )}
        <span>
          {pct > 0 ? "+" : ""}
          {pct.toFixed(1)}% vs Base ({ov.toFixed(1)})
        </span>
      </span>
    );
  };

  const combinedRuns = (sameSession ? [[cand, "Current"] as const] : [[base, "Baseline"] as const, [cand, "Candidate"] as const]).flatMap(
    ([sess, role]) => (sess.bench_runs || []).map((r, i) => ({ key: `${sess.session_id}-${i}`, role, sessName: sess.name, ...r }))
  );

  return (
    <div className="flex flex-col gap-5">
      {/* Selector & Axis Controls Bar */}
      <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] divide-y divide-[var(--border)]">
        <div className="p-4 flex flex-wrap items-end justify-between gap-4">
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1.5 min-w-[230px]">
              <Label htmlFor="cmp-base-trigger">Baseline 基准会话</Label>
              <Select value={baseSessionId || ""} onValueChange={onSelectBase}>
                <SelectTrigger id="cmp-base-trigger">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {sessions.map((s) => (
                    <SelectItem key={s.session_id} value={s.session_id}>
                      {s.is_baseline ? "[默认基线] " : ""}
                      {s.name} ({s.bench_count}B / {s.eval_count}E)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-1.5 min-w-[230px]">
              <Label htmlFor="cmp-cand-trigger">Candidate 对比会话</Label>
              <Select value={candSessionId || ""} onValueChange={onSelectCand}>
                <SelectTrigger id="cmp-cand-trigger">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {sessions.map((s) => (
                    <SelectItem key={s.session_id} value={s.session_id}>
                      {s.name} ({s.bench_count}B / {s.eval_count}E)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Button variant="secondary" loading={pinBusy} onClick={handlePin}>
              <Pin className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
              <span>设为默认基线</span>
            </Button>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Tabs value={xMode} onValueChange={(v) => setXMode(v as any)}>
              <TabsList aria-label="负载横轴维度">
                <TabsTrigger value="auto">Auto X</TabsTrigger>
                <TabsTrigger value="concurrency">Concurrency</TabsTrigger>
                <TabsTrigger value="qps">QPS</TabsTrigger>
              </TabsList>
            </Tabs>

            <Tabs value={statKey} onValueChange={(v) => setStatKey(v as any)}>
              <TabsList aria-label="延迟统计分位数">
                <TabsTrigger value="mean">Mean</TabsTrigger>
                <TabsTrigger value="p50">P50</TabsTrigger>
                <TabsTrigger value="p95">P95</TabsTrigger>
                <TabsTrigger value="p99">P99</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </div>

        {/* Config Diff Strip */}
        <div className="px-4 py-2.5 bg-[var(--surface-subtle)]/55 flex flex-wrap items-center gap-2 text-xs">
          <span className="font-semibold text-[var(--text-sec)]">实例配置差异:</span>
          {sameSession ? (
            <span className="text-[var(--text-muted)]">当前选择同一会话 ({base.name})，选择不同会话可查看参数与性能 A/B 差异</span>
          ) : diffs.length > 0 ? (
            diffs.slice(0, 10).map((d) => (
              <Badge key={d.key} tone="brand" className="tabular-mono">
                <span>{d.key}:</span>
                <span>{String(d.baseline ?? "none")}</span>
                <ArrowRight className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                <span>{String(d.candidate ?? "none")}</span>
              </Badge>
            ))
          ) : (
            <Badge tone="success">
              <CheckCircle2 className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
              <span>启动参数与标签完全一致</span>
            </Badge>
          )}
        </div>
      </div>

      {/* Unified 4-Column A/B Instrument Strip (zero nested cards, zero eyebrows) */}
      <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-[var(--border)]">
        <div className="p-4">
          <div className="text-xs font-semibold text-[var(--text-muted)]">峰值输出吞吐</div>
          <div className="text-xl font-bold tabular-mono my-1 text-[var(--text)]">
            {cPeakTps != null ? `${cPeakTps.toFixed(1)} tok/s` : "—"}
          </div>
          {renderDelta(cPeakTps, bPeakTps, false)}
        </div>

        <div className="p-4">
          <div className="text-xs font-semibold text-[var(--text-muted)]">最低 TPOT ({statKey.toUpperCase()})</div>
          <div className="text-xl font-bold tabular-mono my-1 text-[var(--text)]">
            {cMinTpot != null ? `${cMinTpot.toFixed(2)} ms` : "—"}
          </div>
          {renderDelta(cMinTpot, bMinTpot, true)}
        </div>

        <div className="p-4">
          <div className="text-xs font-semibold text-[var(--text-muted)]">数据集准确率</div>
          <div className="text-base font-bold tabular-mono my-1.5 text-[var(--text)] truncate">
            {cDsEvals.length
              ? cDsEvals.map((e) => `${e.suite_name}: ${((e.correct_rate || 0) * 100).toFixed(1)}%`).join(" · ")
              : "未运行"}
          </div>
          <div className="text-xs text-[var(--text-muted)]">共 {cDsEvals.length} 项数据集评测记录</div>
        </div>

        <div className="p-4">
          <div className="text-xs font-semibold text-[var(--text-muted)]">Logits 数值对齐</div>
          <div className="my-1.5 flex items-center gap-2">
            {cLogits ? (
              <>
                <Badge tone={cLogits.passed ? "success" : "danger"} className="tabular-mono">
                  {cLogits.passed ? (
                    <CheckCircle2 className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                  ) : (
                    <XCircle className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                  )}
                  <span>{cLogits.passed ? "PASS" : "FAIL"}</span>
                </Badge>
                <span className="text-sm font-bold tabular-mono">
                  max|Δ|={cLogits.summary?.max_abs_diff ?? "—"}
                </span>
              </>
            ) : (
              <span className="text-base font-bold text-[var(--text-muted)]">未运行</span>
            )}
          </div>
          <div className="text-xs text-[var(--text-muted)] truncate">
            {cLogits ? `Ref: ${cLogits.ref_base_url || ""}` : "可在「发起压测」启用对齐校验"}
          </div>
        </div>
      </div>

      {/* Charts Workbench */}
      <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-[var(--border)]">
        <div className="p-4 min-w-0">
          <div className="flex items-center justify-between gap-2 mb-3">
            <h3 className="text-sm font-bold m-0 text-[var(--text)]">Pareto 最优前沿：输出吞吐 vs 单用户速度</h3>
            <span className="text-xs text-[var(--text-muted)] tabular-mono">tok/s vs 1000/TPOT</span>
          </div>
          <div className="h-[240px] w-full min-w-0">
            {paretoData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={paretoData} margin={{ top: 8, right: 26, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis
                    dataKey="speed"
                    type="number"
                    domain={["auto", "auto"]}
                    padding={{ left: 8, right: 16 }}
                    tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                    unit=" tok/s/u"
                  />
                  <YAxis tick={{ fontSize: 11, fill: "var(--text-muted)" }} width={48} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      borderRadius: "8px",
                      color: "#f8fafc",
                      fontSize: "12px",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  {!sameSession && (
                    <Line
                      type="monotone"
                      dataKey="baseTps"
                      name={`Base: ${base.name}`}
                      stroke="var(--text-muted)"
                      strokeWidth={2}
                      strokeDasharray="5 4"
                      connectNulls
                      isAnimationActive={false}
                    />
                  )}
                  <Line
                    type="monotone"
                    dataKey="candTps"
                    name={`Cand: ${cand.name}`}
                    stroke="var(--brand)"
                    strokeWidth={2.4}
                    connectNulls
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-[var(--text-muted)]">
                暂无可绘制的 Pareto 曲线数据
              </div>
            )}
          </div>
        </div>

        <div className="p-4 min-w-0">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
            <h3 className="text-sm font-bold m-0 text-[var(--text)]">负载扫描指标对比</h3>
            <div className="w-56">
              <Label htmlFor="cmp-metric-trigger" className="sr-only">
                选择对比指标
              </Label>
              <Select value={cmpMetric} onValueChange={setCmpMetric}>
                <SelectTrigger id="cmp-metric-trigger" className="min-h-[32px] text-xs tabular-mono py-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="output_tps">Output Throughput (tok/s)</SelectItem>
                  <SelectItem value="total_tps">Total Token Throughput (tok/s)</SelectItem>
                  <SelectItem value="tpot_ms">TPOT 逐 Token 延迟 (ms)</SelectItem>
                  <SelectItem value="ttft_ms">TTFT 首字延迟 (ms)</SelectItem>
                  <SelectItem value="itl_ms">ITL 块间延迟 (ms)</SelectItem>
                  <SelectItem value="latency_ms">E2E 请求总延迟 (ms)</SelectItem>
                  <SelectItem value="goodput_rps">Request Goodput (req/s)</SelectItem>
                  <SelectItem value="cache_pct">Global Cache Ratio (%)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="h-[240px] w-full min-w-0">
            {metricSweepData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metricSweepData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="x" tick={{ fontSize: 11, fill: "var(--text-muted)" }} />
                  <YAxis tick={{ fontSize: 11, fill: "var(--text-muted)" }} width={48} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      borderRadius: "8px",
                      color: "#f8fafc",
                      fontSize: "12px",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  {!sameSession && (
                    <Line
                      type="monotone"
                      dataKey="base"
                      name={`Base: ${base.name}`}
                      stroke="var(--text-muted)"
                      strokeWidth={2}
                      strokeDasharray="5 4"
                      connectNulls
                      isAnimationActive={false}
                    />
                  )}
                  <Line
                    type="monotone"
                    dataKey="cand"
                    name={`Cand: ${cand.name}`}
                    stroke="var(--emerald)"
                    strokeWidth={2.4}
                    connectNulls
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-[var(--text-muted)]">
                暂无负载扫描曲线数据
              </div>
            )}
          </div>
        </div>
      </div>

      {/* KV Peak Curve + Sessions Ledger */}
      <div className="grid grid-cols-1 lg:grid-cols-[0.88fr_1.12fr] gap-5">
        <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] p-4 min-w-0">
          <div className="flex items-center justify-between gap-2 mb-3">
            <h3 className="text-sm font-bold m-0 text-[var(--text)]">压测窗口服务端 KV Cache 峰值 (%)</h3>
            <Badge tone="neutral" className="tabular-mono">Windowed /metrics</Badge>
          </div>
          <div className="h-[230px] w-full min-w-0">
            {kvSweepData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={kvSweepData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="x" tick={{ fontSize: 11, fill: "var(--text-muted)" }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "var(--text-muted)" }} width={42} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      borderRadius: "8px",
                      color: "#f8fafc",
                      fontSize: "12px",
                    }}
                  />
                  <ReferenceLine
                    y={85}
                    stroke="var(--crimson)"
                    strokeDasharray="3 3"
                    label={{ value: "85% 饱和", position: "insideTopRight", fontSize: 10, fill: "var(--crimson)" }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  {!sameSession && (
                    <Line
                      type="monotone"
                      dataKey="base"
                      name="Base KV Peak %"
                      stroke="var(--amber)"
                      strokeWidth={2}
                      strokeDasharray="5 4"
                      connectNulls
                      isAnimationActive={false}
                    />
                  )}
                  <Line
                    type="monotone"
                    dataKey="cand"
                    name="Cand KV Peak %"
                    stroke="var(--brand)"
                    strokeWidth={2.4}
                    connectNulls
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-[var(--text-muted)]">
                暂无压测窗口服务端 KV Cache 记录
              </div>
            )}
          </div>
        </div>

        <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
          <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between gap-2">
            <h3 id="sessions-ledger-heading" className="text-sm font-bold m-0 text-[var(--text)]">
              全部服务会话账本 ({sessions.length})
            </h3>
            <span className="text-xs text-[var(--text-muted)]">按单次服务实例归档</span>
          </div>
          <div tabIndex={0} role="region" aria-labelledby="sessions-ledger-heading" className="overflow-auto max-h-[245px]">
            <table className="w-full border-collapse text-xs tabular-nums">
              <thead className="sticky top-0 z-10 bg-[var(--surface-subtle)] text-[var(--text-muted)] uppercase tracking-wider border-b border-[var(--border)]">
                <tr>
                  <th scope="col" className="py-2 px-3 text-left font-semibold">会话名称</th>
                  <th scope="col" className="py-2 px-2.5 text-left font-semibold">状态</th>
                  <th scope="col" className="py-2 px-2.5 text-left font-semibold">记录数</th>
                  <th scope="col" className="py-2 px-2.5 text-right font-semibold">峰值 tok/s</th>
                  <th scope="col" className="py-2 px-3 text-right font-semibold">操作</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border)]">
                {sessions.map((s) => (
                  <tr
                    key={s.session_id}
                    className={cn(
                      s.session_id === candSessionId &&
                        "bg-[var(--brand-soft)] shadow-[inset_3px_0_0_var(--brand)]"
                    )}
                  >
                    <td className="py-2.5 px-3">
                      <div className="font-bold text-[var(--text)] flex items-center gap-1.5">
                        {s.is_baseline && <Pin className="w-3 h-3 text-[var(--brand)]" strokeWidth={2} aria-hidden="true" />}
                        <span>{s.name}</span>
                      </div>
                      <div className="text-[11px] text-[var(--text-muted)] tabular-mono">{s.base_url}</div>
                    </td>
                    <td className="py-2.5 px-2.5">
                      <Badge tone={s.status === "online" ? "success" : "neutral"} className="tabular-mono">
                        {s.status} · {s.mode}
                      </Badge>
                    </td>
                    <td className="py-2.5 px-2.5 tabular-mono">
                      {s.bench_count}B / {s.eval_count}E
                    </td>
                    <td className="py-2.5 px-2.5 text-right font-bold tabular-mono">
                      {s.peak_output_tps != null ? s.peak_output_tps.toFixed(1) : "—"}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <div className="inline-flex items-center gap-1">
                        <Button variant="secondary" size="sm" onClick={() => onSelectBase(s.session_id)}>
                          Base
                        </Button>
                        <Button variant="secondary" size="sm" onClick={() => onSelectCand(s.session_id)}>
                          Cand
                        </Button>
                        <Button
                          variant="danger"
                          size="icon"
                          aria-label={`删除会话 ${s.name}`}
                          onClick={() => onDeleteSession(s)}
                        >
                          <Trash2 className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Detailed Runs Comparison Table */}
      <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
        <div className="px-5 py-3 border-b border-[var(--border)]">
          <h3 id="runs-detail-heading" className="text-sm font-bold m-0 text-[var(--text)]">
            选中会话压测明细对比表 ({combinedRuns.length})
          </h3>
        </div>
        <div tabIndex={0} role="region" aria-labelledby="runs-detail-heading" className="overflow-x-auto">
          <table className="w-full border-collapse text-xs tabular-nums whitespace-nowrap">
            <thead className="bg-[var(--surface-subtle)] text-[var(--text-muted)] text-[11px] uppercase tracking-wider border-b border-[var(--border)]">
              <tr>
                <th scope="col" className="py-2.5 pl-4 pr-2.5 text-left font-semibold">会话</th>
                <th scope="col" className="py-2.5 px-2.5 text-left font-semibold">Workload</th>
                <th scope="col" className="py-2.5 px-2 text-right font-semibold">Conc</th>
                <th scope="col" className="py-2.5 px-2 text-right font-semibold">QPS</th>
                <th scope="col" className="py-2.5 px-2.5 text-right font-semibold">Out tok/s</th>
                <th scope="col" className="py-2.5 px-2.5 text-right font-semibold">Total tok/s</th>
                <th scope="col" className="py-2.5 px-2.5 text-right font-semibold">Goodput</th>
                <th scope="col" className="py-2.5 px-2.5 text-right font-semibold">TTFT Mean/P99</th>
                <th scope="col" className="py-2.5 px-2.5 text-right font-semibold">TPOT Mean/P99</th>
                <th scope="col" className="py-2.5 pl-2 pr-4 text-right font-semibold">KV Peak</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border)]">
              {combinedRuns.map((r) => (
                <tr key={r.key} className="hover:bg-[var(--surface-subtle)]">
                  <td className="py-2.5 pl-4 pr-2.5">
                    <Badge tone={r.role === "Baseline" ? "neutral" : "brand"} className="mr-1.5">
                      {r.role}
                    </Badge>
                    <span className="font-semibold">{r.sessName}</span>
                  </td>
                  <td className="py-2.5 px-2.5 tabular-mono">{r.workload || r.subcommand}</td>
                  <td className="py-2.5 px-2 text-right tabular-mono">{r.max_concurrency ?? "inf"}</td>
                  <td className="py-2.5 px-2 text-right tabular-mono">{r.request_rate ?? "inf"}</td>
                  <td className="py-2.5 px-2.5 text-right font-bold tabular-mono">{fmtNum(r.summary?.output_throughput_tps)}</td>
                  <td className="py-2.5 px-2.5 text-right tabular-mono">{fmtNum(r.summary?.total_token_throughput_tps)}</td>
                  <td className="py-2.5 px-2.5 text-right tabular-mono">
                    {r.summary?.request_goodput_rps != null
                      ? `${fmtNum(r.summary.request_goodput_rps)} (${r.summary.request_goodput_pct}%)`
                      : fmtNum(r.summary?.request_throughput_rps)}
                  </td>
                  <td className="py-2.5 px-2.5 text-right tabular-mono">
                    {fmtNum(r.metrics?.ttft_ms?.mean)} / {fmtNum(r.metrics?.ttft_ms?.p99)} ms
                  </td>
                  <td className="py-2.5 px-2.5 text-right tabular-mono">
                    {fmtNum(r.metrics?.tpot_ms?.mean)} / {fmtNum(r.metrics?.tpot_ms?.p99)} ms
                  </td>
                  <td className="py-2.5 pl-2 pr-4 text-right tabular-mono">
                    {r.telemetry?.summary?.kv_cache_peak_pct != null ? `${r.telemetry.summary.kv_cache_peak_pct}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

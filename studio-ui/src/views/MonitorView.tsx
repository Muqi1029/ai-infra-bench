import * as React from "react";
import {
  Area,
  AreaChart,
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
  Activity,
  CheckCircle2,
  Cpu,
  ExternalLink,
  Play,
  Plus,
  Radio,
  Server,
  Terminal,
  Trash2,
  UploadCloud,
  XCircle,
} from "lucide-react";
import type { LiveTarget } from "../types";
import { cn } from "../lib/utils";
import { Button } from "../components/ui/button";
import { Input, Label } from "../components/ui/input";
import { Badge } from "../components/ui/sheet";

interface MonitorViewProps {
  targets: LiveTarget[];
  focusedTarget: string | null;
  onSelectTarget: (label: string) => void;
  onJumpToBench: (label: string) => void;
  onRequestDeregister: (label: string) => void;
  onRegisterManual: (url: string, labels: string[]) => Promise<void>;
  onOpenCliSheet: () => void;
  grafanaBaseUrl: string | null;
}

export function MonitorView({
  targets,
  focusedTarget,
  onSelectTarget,
  onJumpToBench,
  onRequestDeregister,
  onRegisterManual,
  onOpenCliSheet,
  grafanaBaseUrl,
}: MonitorViewProps) {
  const [regUrl, setRegUrl] = React.useState("");
  const [regLabels, setRegLabels] = React.useState("");
  const [regBusy, setRegBusy] = React.useState(false);
  const [regFeedback, setRegFeedback] = React.useState<{ ok: boolean; msg: string } | null>(null);
  const urlInputRef = React.useRef<HTMLInputElement>(null);

  const handleManualSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = regUrl.trim();
    if (!trimmed) {
      urlInputRef.current?.focus();
      return;
    }
    setRegBusy(true);
    setRegFeedback(null);
    try {
      const parsedLabels = regLabels
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      await onRegisterManual(trimmed, parsedLabels);
      setRegUrl("");
      setRegFeedback({ ok: true, msg: `已接入监控目标 ${trimmed}` });
    } catch (err: any) {
      setRegFeedback({ ok: false, msg: `注册失败: ${err?.message || "请求异常"}` });
    } finally {
      setRegBusy(false);
    }
  };

  const activeTarget = React.useMemo(() => {
    if (!targets.length) return null;
    return targets.find((t) => t.label === focusedTarget) || targets[0];
  }, [targets, focusedTarget]);

  const chartData = React.useMemo(() => {
    if (!activeTarget?.series?.length) return [];
    const pts = activeTarget.series;
    const tLast = pts[pts.length - 1]?.ts || 0;
    return pts.map((p) => ({
      relSec: Math.round((p.ts || 0) - tLast),
      tps: p.gen_throughput_tps != null ? Number(p.gen_throughput_tps) : null,
      kvPct: p.kv_cache_pct != null ? Number(p.kv_cache_pct) : null,
      hitPct: p.cache_hit_pct != null ? Number(p.cache_hit_pct) : null,
      runWait: `${p.running_reqs ?? 0} / ${p.waiting_reqs ?? 0}`,
    }));
  }, [activeTarget]);

  const srvArgsInfo = React.useMemo(() => {
    if (!activeTarget) return { text: "", isCode: false };
    const args = activeTarget.server_info?.server_args || {};
    const pairs = Object.entries(args)
      .slice(0, 6)
      .map(([k, v]) => `${k}=${String(v)}`);
    if (pairs.length > 0) return { text: pairs.join(" · "), isCode: true };
    if (activeTarget.server_info?.launch_command) return { text: activeTarget.server_info.launch_command, isCode: true };
    return { text: "正在实时采集服务端指标", isCode: false };
  }, [activeTarget]);

  return (
    <div className="flex flex-col gap-5">
      {/* Top Workspace Header + Inline Registration Bar */}
      <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] divide-y divide-[var(--border)]">
        <div className="px-5 py-3.5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-bold m-0 text-[var(--text)]">在线推理集群与实时遥测</h2>
            <p className="text-xs text-[var(--text-muted)] m-0 mt-0.5">
              支持 GPU 服务启动自动注册（直连 Pull 与跨网段 Push）或手动接入已运行实例
            </p>
          </div>
          <Button variant="secondary" size="sm" onClick={onOpenCliSheet}>
            <Terminal className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
            <span>自动注册命令与组件状态</span>
          </Button>
        </div>

        <form onSubmit={handleManualSubmit} aria-label="手动注册推理后端" className="px-5 py-3.5">
          <div className="grid grid-cols-1 md:grid-cols-[minmax(240px,1.25fr)_minmax(200px,1fr)_auto] gap-3 items-end">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="reg-url-input">手动注册推理 Backend URL</Label>
              <Input
                ref={urlInputRef}
                id="reg-url-input"
                name="url"
                type="url"
                placeholder="http://10.x.x.x:30000"
                value={regUrl}
                onChange={(e) => setRegUrl(e.target.value)}
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                className="tabular-mono"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="reg-labels-input">分组标签 (可选，逗号分隔)</Label>
              <Input
                id="reg-labels-input"
                name="labels"
                type="text"
                placeholder="cluster=bj,env=prod"
                value={regLabels}
                onChange={(e) => setRegLabels(e.target.value)}
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                className="tabular-mono"
              />
            </div>
            <Button type="submit" variant="primary" loading={regBusy}>
              <Plus className="w-4 h-4" strokeWidth={1.75} aria-hidden="true" />
              <span>注册监控目标</span>
            </Button>
          </div>
          {regFeedback && (
            <div
              role="status"
              className={cn(
                "mt-2.5 inline-flex items-center gap-1.5 text-xs font-medium",
                regFeedback.ok ? "text-[var(--emerald)]" : "text-[var(--crimson)]"
              )}
            >
              {regFeedback.ok ? (
                <CheckCircle2 className="w-3.5 h-3.5 shrink-0" strokeWidth={1.75} aria-hidden="true" />
              ) : (
                <XCircle className="w-3.5 h-3.5 shrink-0" strokeWidth={1.75} aria-hidden="true" />
              )}
              <span>{regFeedback.msg}</span>
            </div>
          )}
        </form>
      </div>

      {/* Signature Fleet Telemetry Ledger or Instructive Empty State */}
      {targets.length === 0 ? (
        <div className="rounded-[12px] border border-dashed border-[var(--control-border)] bg-[var(--surface)] px-6 py-11 text-center">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-[10px] bg-[var(--surface-subtle)] text-[var(--brand)]">
            <Server className="w-5 h-5" strokeWidth={1.75} aria-hidden="true" />
          </div>
          <h3 className="text-[15px] font-bold text-[var(--text)] m-0">当前暂无在线注册的推理后端</h3>
          <p className="text-[13px] text-[var(--text-muted)] max-w-xl mx-auto mt-1.5 mb-2.5">
            在 GPU 节点启动 SGLang / vLLM 时包裹下方命令即可自动注册上线，或在上方填入已启动服务的地址：
          </p>
          <div className="mb-4">
            <code className="inline-block max-w-full overflow-x-auto whitespace-nowrap tabular-mono text-xs px-2.5 py-1 rounded-[6px] border border-[var(--border)] bg-[var(--surface-subtle)] text-[var(--text-sec)]">
              aib monitor --server {window.location.origin} -- python3 -m sglang.launch_server ...
            </code>
          </div>
          <div className="flex flex-wrap justify-center gap-2.5">
            <Button variant="primary" size="sm" onClick={onOpenCliSheet}>
              <Terminal className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
              <span>查看启动自动注册命令</span>
            </Button>
            <Button variant="secondary" size="sm" onClick={() => urlInputRef.current?.focus()}>
              <Plus className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
              <span>手动填写 Backend URL</span>
            </Button>
          </div>
        </div>
      ) : (
        <>
          <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
            <div className="px-5 py-3 border-b border-[var(--border)] flex flex-wrap items-center justify-between gap-2">
              <h3 id="fleet-ledger-heading" className="text-sm font-bold m-0 text-[var(--text)]">
                在线服务遥测矩阵 ({targets.length})
              </h3>
              <span className="text-xs text-[var(--text-muted)]">点击实例名称切换下方实时趋势曲线</span>
            </div>

            <div tabIndex={0} role="region" aria-labelledby="fleet-ledger-heading" className="overflow-x-auto">
              <table className="w-full border-collapse text-[12.5px] tabular-nums whitespace-nowrap">
                <thead>
                  <tr className="bg-[var(--surface-subtle)] text-[var(--text-muted)] text-[11px] uppercase tracking-wider border-b border-[var(--border)]">
                    <th scope="col" className="py-2.5 pl-4 pr-3 text-left font-semibold">实例与归档会话</th>
                    <th scope="col" className="py-2.5 px-2.5 text-left font-semibold">模型与硬件</th>
                    <th scope="col" className="py-2.5 px-2.5 text-left font-semibold">模式</th>
                    <th scope="col" className="py-2.5 px-2.5 text-right font-semibold">实时吞吐</th>
                    <th scope="col" className="py-2.5 px-3 text-left font-semibold min-w-[122px]">KV Cache</th>
                    <th scope="col" className="py-2.5 px-2.5 text-right font-semibold">Run/Wait</th>
                    <th scope="col" className="py-2.5 px-2.5 text-right font-semibold">Hit %</th>
                    <th scope="col" className="md:sticky md:right-0 md:z-10 bg-[var(--surface-subtle)] md:shadow-[-1px_0_0_var(--border)] py-2.5 pl-2.5 pr-4 text-right font-semibold">
                      操作
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border)]">
                  {targets.map((t) => {
                    const lt = t.latest || {};
                    const modelName = t.labels?.model || t.server_info?.model || "Unknown Model";
                    const cluster = t.labels?.cluster || t.labels?.env || "default";
                    const kvVal = lt.kv_cache_pct != null ? Number(lt.kv_cache_pct) : null;
                    const kvClamped = kvVal != null ? Math.min(100, Math.max(0, kvVal)) : 0;
                    const kvBarColor =
                      kvVal == null
                        ? "bg-[var(--text-muted)]"
                        : kvVal > 85
                        ? "bg-[var(--crimson)]"
                        : kvVal > 68
                        ? "bg-[var(--amber)]"
                        : "bg-[var(--brand)]";
                    const isSelected = activeTarget?.label === t.label;
                    const showSessionName =
                      t.session_name && t.session_name !== t.label && t.session_name !== t.base_url;
                    const gLink = grafanaBaseUrl
                      ? `${grafanaBaseUrl}/d/sglang-dashboard/sglang-overview?orgId=1&var-aib_target=${encodeURIComponent(t.label)}&refresh=5s`
                      : null;

                    return (
                      <tr
                        key={t.label}
                        onClick={(e) => {
                          if ((e.target as HTMLElement).closest("button, a")) return;
                          onSelectTarget(t.label);
                        }}
                        className={cn(
                          "group cursor-pointer transition-colors",
                          isSelected
                            ? "bg-[var(--brand-soft)] shadow-[inset_3px_0_0_var(--brand)]"
                            : "hover:bg-[var(--surface-subtle)]"
                        )}
                      >
                        <td className="py-2.5 pl-4 pr-3">
                          <div className="flex items-center gap-1.5">
                            <button
                              type="button"
                              aria-pressed={isSelected}
                              onClick={() => onSelectTarget(t.label)}
                              className="font-bold text-[13px] text-[var(--text)] hover:text-[var(--brand)] tabular-mono cursor-pointer bg-transparent border-0 p-0 text-left"
                            >
                              {t.label}
                            </button>
                            <Badge tone="neutral" className="tabular-mono">{cluster}</Badge>
                          </div>
                          {showSessionName && (
                            <div className="text-[11.5px] font-medium text-[var(--text-sec)] mt-0.5">
                              {t.session_name}
                            </div>
                          )}
                          <div className="text-[11.5px] text-[var(--text-muted)] tabular-mono mt-0.5">{t.base_url}</div>
                        </td>

                        <td className="py-2.5 px-2.5">
                          <div className="font-semibold text-[var(--text)] tabular-mono">{modelName}</div>
                          {t.hardware_info?.summary && (
                            <div className="inline-flex items-center gap-1 text-[11.5px] text-[var(--brand)] tabular-mono mt-0.5">
                              <Cpu className="w-3 h-3 shrink-0" strokeWidth={1.75} aria-hidden="true" />
                              <span>{t.hardware_info.summary}</span>
                            </div>
                          )}
                        </td>

                        <td className="py-2.5 px-2.5">
                          {t.mode === "pull" ? (
                            <Badge tone="success">
                              <Radio className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                              <span>Pull 直连</span>
                            </Badge>
                          ) : (
                            <Badge tone="warning">
                              <UploadCloud className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                              <span>Push 上报</span>
                            </Badge>
                          )}
                        </td>

                        <td className="py-2.5 px-2.5 text-right font-bold text-[13px] tabular-mono">
                          {lt.gen_throughput_tps != null ? `${lt.gen_throughput_tps} tok/s` : "—"}
                        </td>

                        <td className="py-2.5 px-3">
                          <div className="flex items-center gap-2">
                            <div className="w-14 h-2 rounded-full bg-[var(--surface-subtle)] overflow-hidden shrink-0">
                              <div
                                className={cn("h-full rounded-full transition-all duration-200", kvBarColor)}
                                style={{ width: `${kvClamped}%` }}
                              />
                            </div>
                            <span className="text-xs font-semibold tabular-mono">
                              {kvVal != null ? `${kvVal}%` : "—"}
                            </span>
                          </div>
                        </td>

                        <td className="py-2.5 px-2.5 text-right tabular-mono">
                          {lt.running_reqs != null || lt.waiting_reqs != null
                            ? `${lt.running_reqs ?? 0} / ${lt.waiting_reqs ?? 0}`
                            : "—"}
                        </td>

                        <td className="py-2.5 px-2.5 text-right tabular-mono">
                          {lt.cache_hit_pct != null ? `${lt.cache_hit_pct}%` : "—"}
                        </td>

                        <td
                          className={cn(
                            "md:sticky md:right-0 md:z-10 md:shadow-[-1px_0_0_var(--border)] py-2.5 pl-2.5 pr-4 text-right transition-colors",
                            isSelected
                              ? "md:bg-[color-mix(in_oklab,var(--surface)_90%,var(--brand))]"
                              : "md:bg-[var(--surface)] group-hover:md:bg-[var(--surface-subtle)]"
                          )}
                        >
                          <div className="inline-flex items-center justify-end gap-1.5">
                            {gLink && (
                              <a
                                href={gLink}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 min-h-[30px] px-2 py-1 text-xs font-semibold rounded-[7px] border border-[var(--control-border)] bg-[var(--control-bg)] text-[var(--text)] hover:border-[var(--control-border-hover)] no-underline"
                              >
                                <span>Grafana</span>
                                <ExternalLink className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                              </a>
                            )}
                            <Button variant="primary" size="sm" className="min-h-[30px] px-2.5" onClick={() => onJumpToBench(t.label)}>
                              <Play className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                              <span>压测</span>
                            </Button>
                            <Button variant="danger" size="sm" className="min-h-[30px] px-2" onClick={() => onRequestDeregister(t.label)}>
                              <Trash2 className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                              <span>注销</span>
                            </Button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Dual Live Telemetry Recharts for Active Target */}
          {activeTarget && (
            <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
              <div className="px-5 py-3.5 border-b border-[var(--border)] flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="text-sm font-bold m-0 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[var(--text)]">
                    <Activity className="w-4 h-4 text-[var(--brand)] shrink-0" strokeWidth={1.75} aria-hidden="true" />
                    <span>实时趋势：</span>
                    <span className="tabular-mono">{activeTarget.label}</span>
                    {activeTarget.session_name &&
                      activeTarget.session_name !== activeTarget.label &&
                      activeTarget.session_name !== activeTarget.base_url && (
                        <span className="text-xs font-medium text-[var(--text-sec)]">
                          · {activeTarget.session_name}
                        </span>
                      )}
                    <span className="text-xs font-normal text-[var(--text-muted)] tabular-mono">
                      ({activeTarget.base_url})
                    </span>
                  </h3>
                  <p
                    className={cn(
                      "text-xs text-[var(--text-muted)] m-0 mt-0.5 break-all",
                      srvArgsInfo.isCode && "tabular-mono"
                    )}
                  >
                    {srvArgsInfo.text}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {grafanaBaseUrl && (
                    <a
                      href={`${grafanaBaseUrl}/d/sglang-dashboard/sglang-overview?orgId=1&var-aib_target=${encodeURIComponent(activeTarget.label)}&refresh=5s`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1.5 min-h-[32px] px-2.5 py-1 text-xs font-semibold rounded-[7px] border border-[var(--control-border)] bg-[var(--control-bg)] text-[var(--text)] hover:border-[var(--control-border-hover)] no-underline"
                    >
                      <span>Grafana 实例大盘</span>
                      <ExternalLink className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                    </a>
                  )}
                  <Button variant="primary" size="sm" onClick={() => onJumpToBench(activeTarget.label)}>
                    <Play className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                    <span>对该实例发起压测</span>
                  </Button>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-[var(--border)]">
                <div className="p-4 min-w-0">
                  <div className="text-xs font-semibold text-[var(--text-sec)] mb-2.5">实时生成吞吐 (tok/s)</div>
                  <div className="h-[220px] w-full min-w-0">
                    {chartData.some((d) => d.tps != null) ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 8, right: 14, left: 0, bottom: 0 }}>
                          <defs>
                            <linearGradient id="tpsFill" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="var(--brand)" stopOpacity={0.24} />
                              <stop offset="95%" stopColor="var(--brand)" stopOpacity={0.0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                          <XAxis
                            dataKey="relSec"
                            tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                            tickFormatter={(v) => `${v}s`}
                          />
                          <YAxis tick={{ fontSize: 11, fill: "var(--text-muted)" }} width={46} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0f172a",
                              borderColor: "#334155",
                              borderRadius: "8px",
                              color: "#f8fafc",
                              fontSize: "12px",
                            }}
                            labelFormatter={(v) => `相对当前 ${v}s`}
                          />
                          <Area
                            type="monotone"
                            dataKey="tps"
                            name="吞吐 (tok/s)"
                            stroke="var(--brand)"
                            strokeWidth={2}
                            fill="url(#tpsFill)"
                            isAnimationActive={false}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-full flex items-center justify-center text-xs text-[var(--text-muted)]">
                        等待实时吞吐采样数据...
                      </div>
                    )}
                  </div>
                </div>

                <div className="p-4 min-w-0">
                  <div className="text-xs font-semibold text-[var(--text-sec)] mb-2.5">
                    KV Cache 占用率 (%) 与 Prefix Cache 命中率 (%)
                  </div>
                  <div className="h-[220px] w-full min-w-0">
                    {chartData.some((d) => d.kvPct != null || d.hitPct != null) ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 8, right: 14, left: 0, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                          <XAxis
                            dataKey="relSec"
                            tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                            tickFormatter={(v) => `${v}s`}
                          />
                          <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "var(--text-muted)" }} width={42} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0f172a",
                              borderColor: "#334155",
                              borderRadius: "8px",
                              color: "#f8fafc",
                              fontSize: "12px",
                            }}
                            labelFormatter={(v) => `相对当前 ${v}s`}
                          />
                          <ReferenceLine
                            y={85}
                            stroke="var(--amber)"
                            strokeDasharray="3 3"
                            label={{ value: "85% 饱和警戒", position: "insideTopRight", fontSize: 10, fill: "var(--amber)" }}
                          />
                          <Legend wrapperStyle={{ fontSize: "12px" }} />
                          <Line
                            type="monotone"
                            dataKey="kvPct"
                            name="KV Cache (%)"
                            stroke="var(--brand)"
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false}
                          />
                          <Line
                            type="monotone"
                            dataKey="hitPct"
                            name="Prefix Hit (%)"
                            stroke="var(--emerald)"
                            strokeWidth={2}
                            strokeDasharray="4 3"
                            dot={false}
                            isAnimationActive={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-full flex items-center justify-center text-xs text-[var(--text-muted)]">
                        等待服务端缓存采样数据...
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

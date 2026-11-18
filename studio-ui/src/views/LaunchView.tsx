import * as React from "react";
import {
  CheckCircle2,
  GitCompareArrows,
  Play,
  Radar,
  SlidersHorizontal,
  Terminal,
  XCircle,
} from "lucide-react";
import type { JobRecord, LiveTarget } from "../types";
import { api, cn, parseNumList } from "../lib/utils";
import { Button } from "../components/ui/button";
import { Input, Label } from "../components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { Badge, Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "../components/ui/sheet";

export const MANUAL_TARGET_VALUE = "__manual__";

export interface LaunchPreset {
  onlineTarget: string;
  baseUrl: string;
  sessionName: string;
  model: string;
}

interface LaunchViewProps {
  targets: LiveTarget[];
  jobs: JobRecord[];
  preset: LaunchPreset;
  onPresetChange: (next: LaunchPreset) => void;
  onJobSubmitted: (jobId: string) => Promise<void>;
  onOpenJobConsole: (jobId: string) => void;
  onOpenSessionInCompare: (sessionId: string) => void;
}

export function LaunchView({
  targets,
  jobs,
  preset,
  onPresetChange,
  onJobSubmitted,
  onOpenJobConsole,
  onOpenSessionInCompare,
}: LaunchViewProps) {
  const [forceNew, setForceNew] = React.useState(false);
  const [probeBusy, setProbeBusy] = React.useState(false);
  const [probeInfo, setProbeInfo] = React.useState<any | null>(null);
  const [probeError, setProbeError] = React.useState<string | null>(null);

  // Suite switches
  const [enableBench, setEnableBench] = React.useState(true);
  const [enableEvalDs, setEnableEvalDs] = React.useState(false);
  const [enableLogits, setEnableLogits] = React.useState(false);
  const [enableHidden, setEnableHidden] = React.useState(false);
  const [enableSlo, setEnableSlo] = React.useState(false);

  // Bench core & advanced parameters
  const [bDataset, setBDataset] = React.useState("random");
  const [bConc, setBConc] = React.useState("1 8 32");
  const [bNumReq, setBNumReq] = React.useState("64");
  const [bInLen, setBInLen] = React.useState("1024");
  const [bOutLen, setBOutLen] = React.useState("256");
  const [bWarmup, setBWarmup] = React.useState("5");
  const [bQps, setBQps] = React.useState("");
  const [bCacheRatio, setBCacheRatio] = React.useState("0.0");
  const [bGoodput, setBGoodput] = React.useState("");
  const [bNoFlush, setBNoFlush] = React.useState(false);
  const [advOpen, setAdvOpen] = React.useState(false);

  // Eval dataset parameters
  const [dsSelected, setDsSelected] = React.useState<string[]>(["gsm8k"]);
  const [edNumQ, setEdNumQ] = React.useState("50");
  const [edConc, setEdConc] = React.useState("32");
  const [edRepeat, setEdRepeat] = React.useState("1");

  // Logits / hidden states parameters
  const [elRefUrl, setElRefUrl] = React.useState("");
  const [elPrompt, setElPrompt] = React.useState("The capital of France is");
  const [elAtol, setElAtol] = React.useState("0.01");
  const [elRtol, setElRtol] = React.useState("0.0");
  const [elMaxTok, setElMaxTok] = React.useState("8");
  const [elTopk, setElTopk] = React.useState("3");

  // SLO search parameters
  const [sloParam, setSloParam] = React.useState("qps");
  const [sloMin, setSloMin] = React.useState("1");
  const [sloMax, setSloMax] = React.useState("32");
  const [sloTtft, setSloTtft] = React.useState("800");

  const [submitBusy, setSubmitBusy] = React.useState(false);
  const [submitStatus, setSubmitStatus] = React.useState<{ ok: boolean; msg: string } | null>(null);

  const advCustomizedCount = React.useMemo(() => {
    let n = 0;
    if (bQps.trim()) n++;
    if (parseFloat(bCacheRatio) > 0) n++;
    if (bGoodput.trim()) n++;
    if (bNoFlush) n++;
    return n;
  }, [bQps, bCacheRatio, bGoodput, bNoFlush]);

  const handleSelectOnline = (val: string) => {
    if (val === MANUAL_TARGET_VALUE) {
      onPresetChange({ ...preset, onlineTarget: MANUAL_TARGET_VALUE });
      return;
    }
    const found = targets.find((t) => t.label === val);
    if (found) {
      onPresetChange({
        onlineTarget: val,
        baseUrl: found.base_url.startsWith("push://") ? preset.baseUrl : found.base_url,
        sessionName: found.session_name || "",
        model: found.labels?.model || found.server_info?.model || "",
      });
    }
  };

  const handleProbe = async () => {
    const url = preset.baseUrl.trim();
    if (!url) return;
    setProbeBusy(true);
    setProbeError(null);
    setProbeInfo(null);
    try {
      const res = await api("/api/probe", { method: "POST", body: JSON.stringify({ base_url: url }) });
      setProbeInfo(res);
      if (res.server_info?.model && !preset.model) {
        onPresetChange({ ...preset, model: res.server_info.model });
      }
    } catch (err: any) {
      setProbeError(err?.message || "探测失败");
    } finally {
      setProbeBusy(false);
    }
  };

  const toggleDataset = (name: string) => {
    setDsSelected((prev) => (prev.includes(name) ? prev.filter((x) => x !== name) : [...prev, name]));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitStatus(null);

    const onlineLbl = preset.onlineTarget !== MANUAL_TARGET_VALUE ? preset.onlineTarget : "";
    const baseUrl = preset.baseUrl.trim();
    const sessionName = preset.sessionName.trim();
    const model = preset.model.trim() || undefined;

    if (!onlineLbl && !baseUrl) {
      setSubmitStatus({ ok: false, msg: "请选择已注册在线服务或填写 Target Backend URL" });
      return;
    }

    const steps: any[] = [];
    if (enableLogits) {
      if (!elRefUrl.trim()) {
        setSubmitStatus({ ok: false, msg: "启用 Logits 数值对齐时请填写 Reference 黄金基准服务 URL" });
        return;
      }
      steps.push({
        type: "eval_logits",
        params: {
          ref_base_url: elRefUrl.trim(),
          model,
          atol: parseFloat(elAtol) || 0.01,
          rtol: parseFloat(elRtol) || 0.0,
          max_tokens: parseInt(elMaxTok, 10) || 8,
          top_logprobs: parseInt(elTopk, 10) || 3,
          prompt: elPrompt,
        },
      });
    }
    if (enableHidden) {
      if (!elRefUrl.trim()) {
        setSubmitStatus({ ok: false, msg: "启用 Hidden States 对齐时请填写 Reference 黄金基准服务 URL" });
        return;
      }
      steps.push({
        type: "eval_hidden_states",
        params: {
          ref_base_url: elRefUrl.trim(),
          model,
          atol: parseFloat(elAtol) || 0.01,
          rtol: parseFloat(elRtol) || 0.01,
          max_tokens: parseInt(elMaxTok, 10) || 1,
          prompt: elPrompt,
        },
      });
    }
    if (enableEvalDs) {
      if (!dsSelected.length) {
        setSubmitStatus({ ok: false, msg: "请至少勾选一个评测数据集" });
        return;
      }
      const nq = parseInt(edNumQ, 10);
      steps.push({
        type: "eval_dataset",
        params: {
          evals: dsSelected,
          model,
          num_questions: Number.isFinite(nq) && nq > 0 ? nq : null,
          max_concurrency: parseInt(edConc, 10) || 32,
          repeat: parseInt(edRepeat, 10) || 1,
        },
      });
    }
    if (enableBench) {
      const qpsList = parseNumList(bQps, true);
      const concList = parseNumList(bConc, false);
      steps.push({
        type: "bench",
        params: {
          model,
          label: sessionName || undefined,
          dataset: bDataset,
          max_concurrency: concList.length ? concList : qpsList.length ? null : [32],
          qps: qpsList.length ? qpsList : null,
          input_len: parseInt(bInLen, 10) || 1024,
          output_len: parseInt(bOutLen, 10) || 256,
          num_requests: parseInt(bNumReq, 10) || 64,
          num_warmup_requests: parseInt(bWarmup, 10) || 0,
          cache_ratio: parseFloat(bCacheRatio) || 0.0,
          goodput: bGoodput.trim() || null,
          disable_flush_cache: bNoFlush,
        },
      });
    }
    if (enableSlo) {
      steps.push({
        type: "slo",
        params: {
          model,
          parameter: sloParam,
          min: parseInt(sloMin, 10) || 1,
          max: parseInt(sloMax, 10) || 32,
          input_len: parseInt(bInLen, 10) || 1024,
          output_len: parseInt(bOutLen, 10) || 256,
          conditions: [
            { metric: "success_rate", operator: ">=", value: 0.99 },
            { metric: "p99_ttft_ms", operator: "<=", value: parseFloat(sloTtft) || 800 },
          ],
        },
      });
    }

    if (!steps.length) {
      setSubmitStatus({ ok: false, msg: "请至少开启一项压测或正确性校验套件" });
      return;
    }

    const payload = {
      target: onlineLbl || undefined,
      base_url: baseUrl || undefined,
      session_name: sessionName || undefined,
      force_new_session: forceNew,
      job_type: steps.length === 1 ? steps[0].type : "suite",
      params: steps.length === 1 ? steps[0].params : { steps },
    };

    setSubmitBusy(true);
    try {
      const job = await api<{ job_id: string }>("/api/jobs", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setSubmitStatus({ ok: true, msg: `已提交任务 ${job.job_id}` });
      await onJobSubmitted(job.job_id);
    } catch (err: any) {
      setSubmitStatus({ ok: false, msg: `提交失败: ${err?.message || "服务异常"}` });
    } finally {
      setSubmitBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-5">
      {/* Single Flat Unnested Workbench Form */}
      <form
        onSubmit={handleSubmit}
        aria-labelledby="launch-form-heading"
        className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] divide-y divide-[var(--border)]"
      >
        <div className="px-5 py-3.5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 id="launch-form-heading" className="text-base font-bold m-0 text-[var(--text)]">
              发起性能压测与正确性校验
            </h2>
            <p className="text-xs text-[var(--text-muted)] m-0 mt-0.5">
              同一 Backend 的任务自动排队串行执行，避免并发压测相互干扰
            </p>
          </div>
          {jobs.length > 0 && (
            <Button variant="secondary" size="sm" onClick={() => onOpenJobConsole(jobs[0].job_id)}>
              <Terminal className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
              <span>打开实时日志控制台</span>
            </Button>
          )}
        </div>

        {/* Zone 1: Target & Session */}
        <div className="p-5 flex flex-col gap-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="job-online-trigger">已注册在线服务</Label>
              <Select value={preset.onlineTarget} onValueChange={handleSelectOnline}>
                <SelectTrigger id="job-online-trigger" className="tabular-mono">
                  <SelectValue placeholder="选择在线实例或手动填写 URL" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={MANUAL_TARGET_VALUE}>手动输入任意可达 Backend URL</SelectItem>
                  {targets.map((t) => (
                    <SelectItem key={t.label} value={t.label} className="tabular-mono">
                      {t.label} · {t.session_name || t.base_url}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="job-base-url">Target Backend URL</Label>
              <div className="flex gap-2">
                <Input
                  id="job-base-url"
                  type="url"
                  placeholder="http://127.0.0.1:30000"
                  value={preset.baseUrl}
                  onChange={(e) => onPresetChange({ ...preset, baseUrl: e.target.value })}
                  autoCorrect="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  className="tabular-mono flex-1"
                />
                <Button variant="secondary" loading={probeBusy} onClick={handleProbe}>
                  <Radar className="w-4 h-4" strokeWidth={1.75} aria-hidden="true" />
                  <span>探测</span>
                </Button>
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="job-session-name">归档会话名称 (便于 A/B 对比)</Label>
              <Input
                id="job-session-name"
                placeholder="例如: glm5-nvfp4-cp-tp8"
                value={preset.sessionName}
                onChange={(e) => onPresetChange({ ...preset, sessionName: e.target.value })}
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                className="tabular-mono"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="job-model">Model 名称 (留空自动探测)</Label>
              <Input
                id="job-model"
                placeholder="自动读取服务默认模型"
                value={preset.model}
                onChange={(e) => onPresetChange({ ...preset, model: e.target.value })}
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                className="tabular-mono"
              />
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <Switch id="job-force-new" checked={forceNew} onCheckedChange={setForceNew} />
            <Label htmlFor="job-force-new" className="cursor-pointer font-medium">
              强制新建独立归档会话（不合并至当前在线服务会话）
            </Label>
          </div>

          {(probeInfo || probeError) && (
            <div
              role="status"
              className="rounded-[8px] bg-[var(--surface-subtle)] px-3.5 py-2.5 text-xs tabular-mono text-[var(--text-sec)]"
            >
              {probeError ? (
                <span className="text-[var(--crimson)]">探测失败: {probeError}</span>
              ) : (
                <div className="flex flex-col gap-1">
                  <div>
                    <b>Reachable:</b>{" "}
                    <span className={probeInfo.server_info?.reachable ? "text-[var(--emerald)]" : "text-[var(--crimson)]"}>
                      {probeInfo.server_info?.reachable ? "YES" : "NO"}
                    </span>{" "}
                    · <b>Engine:</b> {probeInfo.server_info?.engine || "unknown"} · <b>Model:</b>{" "}
                    {probeInfo.server_info?.model || "N/A"} · <b>/metrics:</b>{" "}
                    {probeInfo.metrics_available ? "Available" : "Unavailable"}
                  </div>
                  {probeInfo.server_info?.server_args &&
                    Object.keys(probeInfo.server_info.server_args).length > 0 && (
                      <div className="text-[var(--text-muted)] break-all">
                        <b>Args:</b> {JSON.stringify(probeInfo.server_info.server_args)}
                      </div>
                    )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Zone 2: Suite Switches */}
        <div className="p-5">
          <h3 className="text-[13px] font-bold text-[var(--text)] m-0 mb-3">选择要执行的测试套件（可组合串联）</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-5 gap-2.5">
            {[
              { id: "sw-bench", label: "性能压测 (bench)", checked: enableBench, set: setEnableBench },
              { id: "sw-eval-ds", label: "数据集评测 (eval-dataset)", checked: enableEvalDs, set: setEnableEvalDs },
              { id: "sw-logits", label: "Logits 数值对齐", checked: enableLogits, set: setEnableLogits },
              { id: "sw-hidden", label: "Hidden States 对齐", checked: enableHidden, set: setEnableHidden },
              { id: "sw-slo", label: "SLO 容量搜索", checked: enableSlo, set: setEnableSlo },
            ].map((item) => (
              <label
                key={item.id}
                htmlFor={item.id}
                className={cn(
                  "flex items-center justify-between gap-2 rounded-[8px] border px-3 py-2.5 cursor-pointer transition-colors",
                  item.checked
                    ? "border-[var(--brand)] bg-[var(--brand-soft)] text-[var(--text)]"
                    : "border-[var(--control-border)] bg-[var(--control-bg)] text-[var(--text-sec)] hover:border-[var(--control-border-hover)]"
                )}
              >
                <span className="text-xs font-semibold">{item.label}</span>
                <Switch id={item.id} checked={item.checked} onCheckedChange={item.set} />
              </label>
            ))}
          </div>
        </div>

        {/* Zone 3: Bench Parameters */}
        {enableBench && (
          <div className="p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3.5">
              <h3 className="text-[13.5px] font-bold m-0 text-[var(--text)]">
                性能压测参数 (aib bench · 自动固化窗口 /metrics)
              </h3>
              <Button variant="secondary" size="sm" onClick={() => setAdvOpen(true)}>
                <SlidersHorizontal className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                <span>高级压测参数</span>
                {advCustomizedCount > 0 && <Badge tone="brand">{advCustomizedCount}</Badge>}
              </Button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="b-dataset-trigger">数据集 (--dataset)</Label>
                <Select value={bDataset} onValueChange={setBDataset}>
                  <SelectTrigger id="b-dataset-trigger">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="random">random (合成固定长度)</SelectItem>
                    <SelectItem value="gsm8k">gsm8k</SelectItem>
                    <SelectItem value="gpqa">gpqa</SelectItem>
                    <SelectItem value="sharegpt">sharegpt</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col gap-1.5">
                <Label htmlFor="b-conc">并发度列表 (空格分隔)</Label>
                <Input
                  id="b-conc"
                  value={bConc}
                  onChange={(e) => setBConc(e.target.value)}
                  placeholder="1 8 32 64"
                  autoCorrect="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  className="tabular-mono"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <Label htmlFor="b-num-req">请求总数 (--num-requests)</Label>
                <Input
                  id="b-num-req"
                  type="number"
                  min={1}
                  value={bNumReq}
                  onChange={(e) => setBNumReq(e.target.value)}
                  className="tabular-mono"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <Label htmlFor="b-in-len">输入长度 (Input Len)</Label>
                <Input
                  id="b-in-len"
                  type="number"
                  min={1}
                  value={bInLen}
                  onChange={(e) => setBInLen(e.target.value)}
                  className="tabular-mono"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <Label htmlFor="b-out-len">输出长度 (Output Len)</Label>
                <Input
                  id="b-out-len"
                  type="number"
                  min={1}
                  value={bOutLen}
                  onChange={(e) => setBOutLen(e.target.value)}
                  className="tabular-mono"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <Label htmlFor="b-warmup">预热请求数 (Warmup)</Label>
                <Input
                  id="b-warmup"
                  type="number"
                  min={0}
                  value={bWarmup}
                  onChange={(e) => setBWarmup(e.target.value)}
                  className="tabular-mono"
                />
              </div>
            </div>
          </div>
        )}

        {/* Zone 4: Eval Dataset Parameters */}
        {enableEvalDs && (
          <div className="p-5">
            <h3 className="text-[13.5px] font-bold m-0 mb-3 text-[var(--text)]">
              数据集正确性评测 (aib eval-dataset · 自动剥离 &lt;think&gt; 保存逐题明细)
            </h3>
            <div className="flex flex-wrap gap-2 mb-4">
              {["gsm8k", "gpqa", "aime25", "constrained_decoding"].map((ds) => {
                const active = dsSelected.includes(ds);
                return (
                  <button
                    key={ds}
                    type="button"
                    aria-pressed={active}
                    onClick={() => toggleDataset(ds)}
                    className={cn(
                      "px-3 py-1.5 rounded-[7px] border text-xs font-semibold tabular-mono cursor-pointer transition-colors",
                      active
                        ? "border-[var(--brand)] bg-[var(--brand-soft)] text-[var(--brand)]"
                        : "border-[var(--control-border)] bg-[var(--control-bg)] text-[var(--text-sec)]"
                    )}
                  >
                    {ds}
                  </button>
                );
              })}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="ed-num-q">抽样题数 (留空全量)</Label>
                <Input
                  id="ed-num-q"
                  type="number"
                  min={1}
                  value={edNumQ}
                  onChange={(e) => setEdNumQ(e.target.value)}
                  className="tabular-mono"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="ed-conc">评测并发度</Label>
                <Input
                  id="ed-conc"
                  type="number"
                  min={1}
                  value={edConc}
                  onChange={(e) => setEdConc(e.target.value)}
                  className="tabular-mono"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="ed-repeat">重复轮数 (Repeat)</Label>
                <Input
                  id="ed-repeat"
                  type="number"
                  min={1}
                  value={edRepeat}
                  onChange={(e) => setEdRepeat(e.target.value)}
                  className="tabular-mono"
                />
              </div>
            </div>
          </div>
        )}

        {/* Zone 5: Logits / Hidden States Parameters */}
        {(enableLogits || enableHidden) && (
          <div className="p-5">
            <h3 className="text-[13.5px] font-bold m-0 mb-3.5 text-[var(--text)]">
              跨端点数值对齐参数 (Reference vs Target)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="el-ref-url">Reference 黄金基准服务 URL</Label>
                <Input
                  id="el-ref-url"
                  type="url"
                  placeholder="http://10.x.x.x:30001"
                  value={elRefUrl}
                  onChange={(e) => setElRefUrl(e.target.value)}
                  autoCorrect="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  className="tabular-mono"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="el-prompt">测试 Prompt</Label>
                <Input
                  id="el-prompt"
                  value={elPrompt}
                  onChange={(e) => setElPrompt(e.target.value)}
                  className="tabular-mono"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="el-atol">绝对误差容限 (atol)</Label>
                <Input
                  id="el-atol"
                  value={elAtol}
                  onChange={(e) => setElAtol(e.target.value)}
                  inputMode="decimal"
                  autoCorrect="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  className="tabular-mono"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="el-rtol">相对误差容限 (rtol)</Label>
                <Input
                  id="el-rtol"
                  value={elRtol}
                  onChange={(e) => setElRtol(e.target.value)}
                  inputMode="decimal"
                  autoCorrect="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  className="tabular-mono"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="el-max-tok">Max Tokens</Label>
                <Input
                  id="el-max-tok"
                  type="number"
                  min={1}
                  value={elMaxTok}
                  onChange={(e) => setElMaxTok(e.target.value)}
                  className="tabular-mono"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="el-topk">Top-K Logprobs</Label>
                <Input
                  id="el-topk"
                  type="number"
                  min={1}
                  max={20}
                  value={elTopk}
                  onChange={(e) => setElTopk(e.target.value)}
                  className="tabular-mono"
                />
              </div>
            </div>
          </div>
        )}

        {/* Zone 6: SLO Capacity Search */}
        {enableSlo && (
          <div className="p-5">
            <h3 className="text-[13.5px] font-bold m-0 mb-3.5 text-[var(--text)]">
              SLO 二分容量搜索参数 (aib slo)
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="slo-param-trigger">搜索维度</Label>
                <Select value={sloParam} onValueChange={setSloParam}>
                  <SelectTrigger id="slo-param-trigger" className="tabular-mono">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="qps">qps</SelectItem>
                    <SelectItem value="max_concurrency">max_concurrency</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="slo-min">搜索下界 (Min)</Label>
                <Input
                  id="slo-min"
                  type="number"
                  min={1}
                  value={sloMin}
                  onChange={(e) => setSloMin(e.target.value)}
                  className="tabular-mono"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="slo-max">搜索上界 (Max)</Label>
                <Input
                  id="slo-max"
                  type="number"
                  min={1}
                  value={sloMax}
                  onChange={(e) => setSloMax(e.target.value)}
                  className="tabular-mono"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="slo-ttft">P99 TTFT 上限 (ms)</Label>
                <Input
                  id="slo-ttft"
                  type="number"
                  min={1}
                  value={sloTtft}
                  onChange={(e) => setSloTtft(e.target.value)}
                  className="tabular-mono"
                />
              </div>
            </div>
          </div>
        )}

        {/* Submit Bar */}
        <div className="px-5 py-4 flex flex-wrap items-center gap-3 bg-[var(--surface-subtle)]/50">
          <Button type="submit" variant="primary" size="lg" loading={submitBusy}>
            <Play className="w-4 h-4" strokeWidth={1.75} aria-hidden="true" />
            <span>提交并启动任务</span>
          </Button>
          {submitStatus && (
            <div
              role="status"
              className={cn(
                "inline-flex items-center gap-1.5 text-[13px] font-medium",
                submitStatus.ok ? "text-[var(--emerald)]" : "text-[var(--crimson)]"
              )}
            >
              {submitStatus.ok ? (
                <CheckCircle2 className="w-4 h-4 shrink-0" strokeWidth={1.75} aria-hidden="true" />
              ) : (
                <XCircle className="w-4 h-4 shrink-0" strokeWidth={1.75} aria-hidden="true" />
              )}
              <span>{submitStatus.msg}</span>
            </div>
          )}
        </div>
      </form>

      {/* Recent Jobs Queue Table (click row to open right-hand Live Console Drawer) */}
      <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
        <div className="px-5 py-3 border-b border-[var(--border)] flex items-center justify-between gap-2">
          <h3 id="recent-jobs-heading" className="text-sm font-bold m-0 text-[var(--text)]">
            任务执行队列 ({jobs.length})
          </h3>
          <span className="text-xs text-[var(--text-muted)]">点击任意任务滑出右侧实时日志抽屉</span>
        </div>
        <div tabIndex={0} role="region" aria-labelledby="recent-jobs-heading" className="overflow-x-auto max-h-72">
          <table className="w-full border-collapse text-[13px] tabular-nums">
            <thead className="sticky top-0 z-10 bg-[var(--surface-subtle)] text-[var(--text-muted)] text-[11.5px] uppercase tracking-wider border-b border-[var(--border)]">
              <tr>
                <th scope="col" className="py-2.5 px-4 text-left font-semibold">Job ID</th>
                <th scope="col" className="py-2.5 px-3 text-left font-semibold">类型</th>
                <th scope="col" className="py-2.5 px-3 text-left font-semibold">状态</th>
                <th scope="col" className="py-2.5 px-3 text-left font-semibold">当前阶段与进度</th>
                <th scope="col" className="py-2.5 px-4 text-right font-semibold">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border)]">
              {jobs.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-xs text-[var(--text-muted)]">
                    暂无执行记录，在上方提交任务后将在此实时更新队列与进度
                  </td>
                </tr>
              ) : (
                jobs.map((j) => {
                  const pct = j.status === "succeeded" ? 100 : j.progress?.pct ?? 0;
                  const tone =
                    j.status === "succeeded"
                      ? "success"
                      : j.status === "failed"
                      ? "danger"
                      : j.status === "running"
                      ? "brand"
                      : "neutral";
                  return (
                    <tr
                      key={j.job_id}
                      onClick={(e) => {
                        if ((e.target as HTMLElement).closest("button")) return;
                        onOpenJobConsole(j.job_id);
                      }}
                      className="cursor-pointer hover:bg-[var(--surface-subtle)] transition-colors"
                    >
                      <td className="py-2.5 px-4 font-semibold tabular-mono">{j.job_id}</td>
                      <td className="py-2.5 px-3 tabular-mono">{j.job_type}</td>
                      <td className="py-2.5 px-3">
                        <Badge tone={tone} className="tabular-mono">{j.status}</Badge>
                      </td>
                      <td className="py-2.5 px-3">
                        <div className="flex items-center gap-2.5 max-w-xs">
                          <div className="w-24 h-1.5 rounded-full bg-[var(--surface-subtle)] overflow-hidden shrink-0">
                            <div className="h-full bg-[var(--brand)]" style={{ width: `${pct}%` }} />
                          </div>
                          <span className="text-xs tabular-mono text-[var(--text-sec)] truncate">
                            {j.progress?.stage || "-"} ({pct}%)
                          </span>
                        </div>
                      </td>
                      <td className="py-2.5 px-4 text-right">
                        <div className="inline-flex items-center gap-1.5">
                          <Button variant="secondary" size="sm" onClick={() => onOpenJobConsole(j.job_id)}>
                            <Terminal className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                            <span>日志</span>
                          </Button>
                          <Button variant="ghost" size="sm" onClick={() => onOpenSessionInCompare(j.session_id)}>
                            <GitCompareArrows className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                            <span>查看对比</span>
                          </Button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Right Slide-Over Sheet: Advanced Benchmark Configuration */}
      <Sheet open={advOpen} onOpenChange={setAdvOpen}>
        <SheetContent side="right" className="max-w-md">
          <SheetHeader>
            <SheetTitle>高级压测参数 (aib bench)</SheetTitle>
            <SheetDescription>配置开环请求速率扫描、前缀缓存比例、SLO Goodput 阈值与缓存清理行为</SheetDescription>
          </SheetHeader>
          <div className="p-5 flex-1 overflow-y-auto flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="adv-qps">开环 QPS 列表 (空格分隔，留空则按并发闭环压测)</Label>
              <Input
                id="adv-qps"
                placeholder="如: 2 8 16 32"
                value={bQps}
                onChange={(e) => setBQps(e.target.value)}
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                className="tabular-mono"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="adv-cache-ratio">共享前缀比例 Cache Ratio (0.0 ~ 1.0)</Label>
              <Input
                id="adv-cache-ratio"
                value={bCacheRatio}
                onChange={(e) => setBCacheRatio(e.target.value)}
                inputMode="decimal"
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                className="tabular-mono"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="adv-goodput">Goodput 达标阈值 (可选)</Label>
              <Input
                id="adv-goodput"
                placeholder="ttft:500 tpot:50"
                value={bGoodput}
                onChange={(e) => setBGoodput(e.target.value)}
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                className="tabular-mono"
              />
            </div>

            <div className="pt-2 border-t border-[var(--border)] flex items-center justify-between gap-3">
              <div>
                <Label htmlFor="adv-no-flush" className="cursor-pointer">
                  跳过压测前 /flush_cache
                </Label>
                <p className="text-xs text-[var(--text-muted)] m-0 mt-0.5">
                  保留服务端现有 KV 缓存状态，适用于热缓存回归测试
                </p>
              </div>
              <Switch id="adv-no-flush" checked={bNoFlush} onCheckedChange={setBNoFlush} />
            </div>
          </div>
          <div className="px-5 py-3.5 border-t border-[var(--border)] flex justify-end">
            <Button variant="primary" onClick={() => setAdvOpen(false)}>
              完成配置
            </Button>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}

import * as React from "react";
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  Eye,
  Play,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import type { CompareResponse, CorrectnessRun, EvalCasesDoc } from "../types";
import { api, cn } from "../lib/utils";
import { Button } from "../components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Badge, Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "../components/ui/sheet";

interface CorrectnessViewProps {
  compareData: CompareResponse | null;
  onJumpToLaunch: () => void;
}

export function CorrectnessView({ compareData, onJumpToLaunch }: CorrectnessViewProps) {
  const [drawerOpen, setDrawerOpen] = React.useState(false);
  const [loadingCases, setLoadingCases] = React.useState(false);
  const [evalDoc, setEvalDoc] = React.useState<EvalCasesDoc | null>(null);
  const [baseCasesMap, setBaseCasesMap] = React.useState<Record<number, any>>({});
  const [caseFilter, setCaseFilter] = React.useState<"all" | "regressed" | "truncated">("all");
  const [logitPos, setLogitPos] = React.useState(0);

  const base = compareData?.sessions?.[0];
  const cand = compareData?.sessions?.[1] || base;

  const rows = React.useMemo(() => {
    if (!base || !cand) return [];
    const bMap = Object.fromEntries((base.correctness_runs || []).map((e) => [`${e.eval_type}:${e.suite_name}`, e]));
    const cMap = Object.fromEntries((cand.correctness_runs || []).map((e) => [`${e.eval_type}:${e.suite_name}`, e]));
    const keys = [...new Set([...Object.keys(bMap), ...Object.keys(cMap)])];
    return keys.map((k) => {
      const b: CorrectnessRun | undefined = bMap[k];
      const c: CorrectnessRun | undefined = cMap[k];
      const active = (c || b)!;
      return { key: k, b, c, active };
    });
  }, [base, cand]);

  const handleInspect = async (candEvalId: string, baseEvalId?: string) => {
    setDrawerOpen(true);
    setLoadingCases(true);
    setLogitPos(0);
    try {
      const doc = await api<EvalCasesDoc>(`/api/evals/${encodeURIComponent(candEvalId)}/cases`);
      setEvalDoc(doc);
      const nextMap: Record<number, any> = {};
      if (baseEvalId && baseEvalId !== candEvalId) {
        try {
          const bDoc = await api<EvalCasesDoc>(`/api/evals/${encodeURIComponent(baseEvalId)}/cases`);
          (bDoc.cases || []).forEach((item: any) => {
            nextMap[item.index] = item;
          });
        } catch (_) {}
      }
      setBaseCasesMap(nextMap);
    } finally {
      setLoadingCases(false);
    }
  };

  const filteredDatasetCases = React.useMemo(() => {
    if (!evalDoc || evalDoc.eval_type !== "dataset") return [];
    const all = evalDoc.cases || [];
    if (caseFilter === "regressed") {
      return all.filter((c: any) => !c.is_right || (baseCasesMap[c.index]?.is_right && !c.is_right));
    }
    if (caseFilter === "truncated") {
      return all.filter((c: any) => c.is_truncated || c.finish_reason === "length");
    }
    return all;
  }, [evalDoc, caseFilter, baseCasesMap]);

  if (!rows.length) {
    return (
      <div className="rounded-[12px] border border-dashed border-[var(--control-border)] bg-[var(--surface)] px-6 py-12 text-center">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-[10px] bg-[var(--surface-subtle)] text-[var(--brand)]">
          <ShieldCheck className="w-5 h-5" strokeWidth={1.75} aria-hidden="true" />
        </div>
        <h2 className="text-[15px] font-bold text-[var(--text)] m-0">当前选中会话暂无正确性校验记录</h2>
        <p className="text-[13px] text-[var(--text-muted)] max-w-lg mx-auto mt-1.5 mb-4">
          支持数据集精度评测 (<code className="tabular-mono">eval-dataset</code>)、跨服务逐位置 Logprobs 对齐 (<code className="tabular-mono">eval-logits</code>) 与隐藏层张量余弦相似度校验 (<code className="tabular-mono">eval-hidden-states</code>)。
        </p>
        <Button variant="primary" size="sm" onClick={onJumpToLaunch}>
          <Play className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
          <span>去发起正确性校验</span>
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="rounded-[12px] border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
        <div className="px-5 py-3.5 border-b border-[var(--border)] flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 id="corr-table-heading" className="text-base font-bold m-0 text-[var(--text)]">
              正确性校验成绩单对比 ({base?.name} vs {cand?.name})
            </h2>
            <p className="text-xs text-[var(--text-muted)] m-0 mt-0.5">
              点击右侧「查看明细」滑出抽屉检视逐题推理链或逐位置 Top-K Logprobs
            </p>
          </div>
          <Badge tone="success">Dataset · Logits · Hidden States</Badge>
        </div>

        <div tabIndex={0} role="region" aria-labelledby="corr-table-heading" className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px] tabular-nums">
            <thead className="bg-[var(--surface-subtle)] text-[var(--text-muted)] text-[11.5px] uppercase tracking-wider border-b border-[var(--border)]">
              <tr>
                <th scope="col" className="py-2.5 px-4 text-left font-semibold">套件 / 数据集</th>
                <th scope="col" className="py-2.5 px-3 text-left font-semibold">类型</th>
                <th scope="col" className="py-2.5 px-3 text-right font-semibold">Baseline</th>
                <th scope="col" className="py-2.5 px-3 text-right font-semibold">Candidate</th>
                <th scope="col" className="py-2.5 px-3 text-left font-semibold">差值 / 误差指标</th>
                <th scope="col" className="py-2.5 px-3 text-right font-semibold">length 截断率</th>
                <th scope="col" className="py-2.5 px-3 text-left font-semibold">状态</th>
                <th scope="col" className="py-2.5 px-4 text-right font-semibold">明细</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border)]">
              {rows.map(({ key, b, c, active }) => {
                const isDs = active.eval_type === "dataset";
                const bTxt = b ? (isDs ? `${((b.correct_rate || 0) * 100).toFixed(2)}%` : b.passed ? "PASS" : "FAIL") : "—";
                const cTxt = c ? (isDs ? `${((c.correct_rate || 0) * 100).toFixed(2)}%` : c.passed ? "PASS" : "FAIL") : "—";

                let diffNode: React.ReactNode = "—";
                if (
                  isDs &&
                  b &&
                  c &&
                  b.correct_rate != null &&
                  c.correct_rate != null &&
                  base?.session_id !== cand?.session_id
                ) {
                  const d = (c.correct_rate - b.correct_rate) * 100;
                  const ok = d >= -0.5;
                  diffNode = (
                    <span
                      className={cn(
                        "inline-flex items-center gap-0.5 font-semibold tabular-mono",
                        ok ? "text-[var(--emerald)]" : "text-[var(--crimson)]"
                      )}
                    >
                      {d >= 0 ? (
                        <ArrowUpRight className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                      ) : (
                        <ArrowDownRight className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                      )}
                      <span>
                        {d >= 0 ? "+" : ""}
                        {d.toFixed(2)}%
                      </span>
                    </span>
                  );
                } else if (active.eval_type === "logits") {
                  diffNode = (
                    <span className="tabular-mono text-xs">
                      max|Δ|={active.summary?.max_abs_diff ?? "—"}, mean|Δ|={active.summary?.mean_abs_diff ?? "—"}
                    </span>
                  );
                } else if (active.eval_type === "hidden_states") {
                  diffNode = (
                    <span className="tabular-mono text-xs">
                      cos={Number(active.summary?.cosine_similarity ?? 0).toFixed(6)}, rmse=
                      {Number(active.summary?.rmse ?? 0).toExponential(2)}
                    </span>
                  );
                }

                return (
                  <tr key={key} className="hover:bg-[var(--surface-subtle)]">
                    <td className="py-3 px-4 font-bold tabular-mono">{active.suite_name}</td>
                    <td className="py-3 px-3 tabular-mono text-xs text-[var(--text-sec)]">{active.eval_type}</td>
                    <td className="py-3 px-3 text-right tabular-mono">{bTxt}</td>
                    <td className="py-3 px-3 text-right font-bold tabular-mono">{cTxt}</td>
                    <td className="py-3 px-3">{diffNode}</td>
                    <td className="py-3 px-3 text-right tabular-mono">
                      {c?.truncated_rate != null && c.truncated_rate > 0 ? (
                        <Badge tone="warning" className="tabular-mono">
                          <AlertTriangle className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                          <span>{(c.truncated_rate * 100).toFixed(1)}%</span>
                        </Badge>
                      ) : (
                        "0.0%"
                      )}
                    </td>
                    <td className="py-3 px-3">
                      <Badge tone={active.passed ? "success" : "danger"} className="tabular-mono">
                        {active.passed ? (
                          <CheckCircle2 className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                        ) : (
                          <XCircle className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                        )}
                        <span>{active.passed ? "PASS" : "FAIL"}</span>
                      </Badge>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => handleInspect((c || b)!.eval_id, b?.eval_id)}
                      >
                        <Eye className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                        <span>查看明细</span>
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Right Slide-Over Sheet: Case & Logits Diff Inspector */}
      <Sheet open={drawerOpen} onOpenChange={setDrawerOpen}>
        <SheetContent side="right" className="max-w-2xl">
          <SheetHeader>
            <SheetTitle>
              {evalDoc ? `${evalDoc.suite_name || evalDoc.eval_type} 校验明细检视` : "加载评测明细"}
            </SheetTitle>
            <SheetDescription>
              检视逐题推理输出（剥离思考链前后）或逐 Token 位置的 Top-K 对数概率对齐详情
            </SheetDescription>
          </SheetHeader>

          <div className="p-5 flex-1 overflow-y-auto flex flex-col gap-4">
            {loadingCases ? (
              <div className="py-12 text-center text-xs text-[var(--text-muted)]">正在加载评测明细数据...</div>
            ) : !evalDoc ? (
              <div className="py-12 text-center text-xs text-[var(--text-muted)]">未找到明细记录</div>
            ) : evalDoc.eval_type === "hidden_states" ? (
              <div className="rounded-[10px] border border-[var(--border)] bg-[var(--surface-subtle)] p-4 text-xs tabular-mono flex flex-col gap-1.5">
                <div>
                  <b>Reference:</b> {evalDoc.ref_base_url}
                </div>
                <div>
                  <b>Shape Match:</b> {String(evalDoc.summary?.shape_match)} (
                  {JSON.stringify(evalDoc.summary?.reference_shape)} vs {JSON.stringify(evalDoc.summary?.target_shape)})
                </div>
                <div>
                  <b>Cosine Similarity:</b> {evalDoc.summary?.cosine_similarity}
                </div>
                <div>
                  <b>RMSE:</b> {evalDoc.summary?.rmse} · <b>Mean |Δ|:</b> {evalDoc.summary?.mean_abs_diff} ·{" "}
                  <b>Max |Δ|:</b> {evalDoc.summary?.max_abs_diff} @ {JSON.stringify(evalDoc.summary?.max_diff_index)}
                </div>
              </div>
            ) : evalDoc.eval_type === "logits" ? (
              (() => {
                const positions = evalDoc.cases || [];
                const cur = positions[logitPos] || positions[0];
                if (!cur) return <div className="text-xs text-[var(--text-muted)]">无逐位置明细</div>;
                return (
                  <>
                    <div className="text-xs text-[var(--text-muted)] tabular-mono">
                      Reference: <code>{evalDoc.ref_base_url}</code> · Prompt:{" "}
                      <code>{evalDoc.summary?.prompt || ""}</code>
                    </div>

                    <div className="flex flex-wrap gap-1.5">
                      {positions.map((p: any, idx: number) => (
                        <button
                          key={idx}
                          type="button"
                          aria-pressed={idx === logitPos}
                          onClick={() => setLogitPos(idx)}
                          className={cn(
                            "inline-flex flex-col items-center px-2.5 py-1.5 rounded-[7px] border cursor-pointer min-w-[70px] transition-colors",
                            idx === logitPos
                              ? "border-[var(--brand)] bg-[var(--brand-soft)]"
                              : "border-[var(--control-border)] bg-[var(--control-bg)]"
                          )}
                        >
                          <span className="tabular-mono text-xs font-bold">
                            {JSON.stringify(p.tgt_token ?? p.ref_token)}
                          </span>
                          <span
                            className={cn(
                              "tabular-mono text-[10.5px]",
                              p.passed ? "text-[var(--emerald)]" : "text-[var(--crimson)]"
                            )}
                          >
                            |Δ|={p.abs_diff != null ? Number(p.abs_diff).toExponential(1) : "—"}
                          </span>
                        </button>
                      ))}
                    </div>

                    <div className="rounded-[10px] border border-[var(--border)] overflow-hidden">
                      <div className="px-3.5 py-2.5 bg-[var(--surface-subtle)] border-b border-[var(--border)] text-xs tabular-mono">
                        <b>Position {cur.position}</b> · Ref={JSON.stringify(cur.ref_token)} ({cur.ref_logprob}) vs Tgt=
                        {JSON.stringify(cur.tgt_token)} ({cur.tgt_logprob}) · |Δ|={cur.abs_diff}
                      </div>
                      <div tabIndex={0} role="region" aria-label="Top-K Logprobs 对比表" className="overflow-x-auto">
                        <table className="w-full border-collapse text-xs tabular-nums">
                          <thead className="bg-[var(--surface-subtle)] text-[var(--text-muted)] border-b border-[var(--border)]">
                            <tr>
                              <th scope="col" className="py-2 px-3 text-left">Rank</th>
                              <th scope="col" className="py-2 px-3 text-left">Ref Token</th>
                              <th scope="col" className="py-2 px-3 text-right">Ref LP</th>
                              <th scope="col" className="py-2 px-3 text-left">Tgt Token</th>
                              <th scope="col" className="py-2 px-3 text-right">Tgt LP</th>
                              <th scope="col" className="py-2 px-3 text-right">|Δ|</th>
                              <th scope="col" className="py-2 px-3 text-left">状态</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-[var(--border)]">
                            {(cur.top_k || []).map((k: any) => (
                              <tr key={k.rank}>
                                <td className="py-2 px-3 tabular-mono">#{k.rank}</td>
                                <td className="py-2 px-3 tabular-mono">{JSON.stringify(k.ref_token)}</td>
                                <td className="py-2 px-3 text-right tabular-mono">{k.ref_logprob}</td>
                                <td className="py-2 px-3 tabular-mono">{JSON.stringify(k.tgt_token)}</td>
                                <td className="py-2 px-3 text-right tabular-mono">{k.tgt_logprob}</td>
                                <td className="py-2 px-3 text-right tabular-mono">{k.abs_diff}</td>
                                <td className="py-2 px-3">
                                  <Badge tone={k.passed ? "success" : "danger"}>{k.passed ? "PASS" : "FAIL"}</Badge>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                );
              })()
            ) : (
              <>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-[var(--text-muted)]">
                    匹配 {filteredDatasetCases.length} 题（最多显示前 80 题）
                  </span>
                  <Tabs value={caseFilter} onValueChange={(v) => setCaseFilter(v as any)}>
                    <TabsList aria-label="题目筛选">
                      <TabsTrigger value="all">全部</TabsTrigger>
                      <TabsTrigger value="regressed">退化 / 失败</TabsTrigger>
                      <TabsTrigger value="truncated">截断 (length)</TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>

                {filteredDatasetCases.length === 0 ? (
                  <div className="py-10 text-center text-xs text-[var(--text-muted)]">当前筛选条件下无匹配题目</div>
                ) : (
                  <div className="divide-y divide-[var(--border)] border border-[var(--border)] rounded-[10px]">
                    {filteredDatasetCases.slice(0, 80).map((c: any) => {
                      const bCase = baseCasesMap[c.index];
                      return (
                        <div key={c.index} className="p-3.5 text-xs flex flex-col gap-1.5">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <strong className="tabular-mono">
                              #{c.index} · Expected: {String(c.expected ?? "—")}
                            </strong>
                            <div className="inline-flex items-center gap-1.5">
                              {bCase && (
                                <>
                                  <Badge tone={bCase.is_right ? "success" : "danger"}>
                                    Base: {bCase.is_right ? "正确" : "错误"}
                                  </Badge>
                                  <ArrowRight className="w-3 h-3 text-[var(--text-muted)]" strokeWidth={1.75} aria-hidden="true" />
                                </>
                              )}
                              <Badge tone={c.is_right ? "success" : "danger"}>
                                Cand: {c.is_right ? "正确" : "错误"}
                              </Badge>
                              {c.is_truncated && <Badge tone="warning">length 截断</Badge>}
                            </div>
                          </div>
                          <div className="text-[var(--text-muted)]">
                            <b>Prompt:</b> {c.prompt}
                          </div>
                          {c.thinking_snippet && (
                            <div className="text-[var(--text-muted)] bg-[var(--surface-subtle)] px-2.5 py-1.5 rounded-[6px]">
                              <b>[&lt;think&gt; 摘要]:</b> {String(c.thinking_snippet).slice(0, 360)}
                            </div>
                          )}
                          <div className="tabular-mono text-[var(--text)]">
                            <b>[剥离思考后输出]:</b> {String(c.answer_text || "").slice(0, 500)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </>
            )}
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}

import * as React from "react";
import {
  Activity,
  Check,
  Copy,
  ExternalLink,
  Gauge,
  GitCompareArrows,
  Menu,
  Moon,
  Play,
  RefreshCw,
  ShieldCheck,
  StopCircle,
  Sun,
  Terminal,
} from "lucide-react";
import type { CompareResponse, JobRecord, LiveTarget, PlatformInfo, ServiceSession } from "./types";
import { api, cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import {
  Badge,
  ConfirmDialog,
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "./components/ui/sheet";
import { MonitorView } from "./views/MonitorView";
import { LaunchView, MANUAL_TARGET_VALUE, type LaunchPreset } from "./views/LaunchView";
import { CompareView } from "./views/CompareView";
import { CorrectnessView } from "./views/CorrectnessView";

type ViewId = "monitor" | "launch" | "compare" | "correctness";

export function App() {
  const [view, setView] = React.useState<ViewId>("monitor");
  const [mobileNavOpen, setMobileNavOpen] = React.useState(false);

  const [info, setInfo] = React.useState<PlatformInfo>({});
  const [sseOnline, setSseOnline] = React.useState(true);
  const [targets, setTargets] = React.useState<LiveTarget[]>([]);
  const [focusedTarget, setFocusedTarget] = React.useState<string | null>(null);
  const [sessions, setSessions] = React.useState<ServiceSession[]>([]);
  const [jobs, setJobs] = React.useState<JobRecord[]>([]);

  const [baseSessionId, setBaseSessionId] = React.useState<string | null>(null);
  const [candSessionId, setCandSessionId] = React.useState<string | null>(null);
  const [compareData, setCompareData] = React.useState<CompareResponse | null>(null);

  const [launchPreset, setLaunchPreset] = React.useState<LaunchPreset>({
    onlineTarget: MANUAL_TARGET_VALUE,
    baseUrl: "",
    sessionName: "",
    model: "",
  });

  // Slide-over Sheets & Confirmation Dialog state
  const [cliSheetOpen, setCliSheetOpen] = React.useState(false);
  const [copiedCmd, setCopiedCmd] = React.useState(false);
  const [jobConsoleOpen, setJobConsoleOpen] = React.useState(false);
  const [selectedJobId, setSelectedJobId] = React.useState<string | null>(null);
  const logPreRef = React.useRef<HTMLPreElement>(null);

  const [confirmState, setConfirmState] = React.useState<{
    open: boolean;
    title: string;
    description: string;
    confirmLabel: string;
    action: () => Promise<void>;
  }>({
    open: false,
    title: "",
    description: "",
    confirmLabel: "确认",
    action: async () => {},
  });

  // Theme state synced with <meta name="color-scheme">
  const [renderedDark, setRenderedDark] = React.useState<boolean>(() => {
    const metaVal = document.querySelector('meta[name="color-scheme"]')?.getAttribute("content");
    if (metaVal === "dark") return true;
    if (metaVal === "light") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  React.useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const syncTheme = () => {
      const metaVal = document.querySelector('meta[name="color-scheme"]')?.getAttribute("content");
      if (metaVal === "dark") setRenderedDark(true);
      else if (metaVal === "light") setRenderedDark(false);
      else setRenderedDark(mq.matches);
    };
    mq.addEventListener("change", syncTheme);
    const onStorage = (e: StorageEvent) => {
      if (e.key === "color-scheme") {
        const meta = document.querySelector('meta[name="color-scheme"]');
        if (meta) {
          meta.setAttribute(
            "content",
            e.newValue === "light" || e.newValue === "dark" ? e.newValue : "light dark"
          );
        }
        syncTheme();
      }
    };
    window.addEventListener("storage", onStorage);
    return () => {
      mq.removeEventListener("change", syncTheme);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  const toggleTheme = () => {
    const sysDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const sysScheme = sysDark ? "dark" : "light";
    const target = renderedDark ? "light" : "dark";
    const meta = document.querySelector('meta[name="color-scheme"]');
    if (target === sysScheme) {
      try {
        localStorage.removeItem("color-scheme");
      } catch (_) {}
      meta?.setAttribute("content", "light dark");
    } else {
      try {
        localStorage.setItem("color-scheme", target);
      } catch (_) {}
      meta?.setAttribute("content", target);
    }
    setRenderedDark(target === "dark");
  };

  const grafanaBaseUrl = React.useMemo(() => {
    if (!info.grafana_enabled || !info.grafana_port) return null;
    return `${window.location.protocol}//${window.location.hostname}:${info.grafana_port}`;
  }, [info]);

  const promBaseUrl = React.useMemo(() => {
    if (!info.prometheus_enabled || !info.prometheus_port) return null;
    return `${window.location.protocol}//${window.location.hostname}:${info.prometheus_port}`;
  }, [info]);

  const autoRegCommand = `aib monitor --server ${window.location.origin} --labels cluster=gpu -- python3 -m sglang.launch_server --model-path /path/to/model --port 30000 --enable-metrics`;

  const loadPlatformInfo = React.useCallback(async () => {
    const res = await api<PlatformInfo>("/api/platform/info");
    setInfo(res);
  }, []);

  const loadLiveTargets = React.useCallback(async () => {
    const res = await api<{ targets: LiveTarget[] }>("/api/monitor/live");
    const list = res.targets || [];
    setTargets(list);
    setFocusedTarget((prev) => (prev && list.some((t) => t.label === prev) ? prev : list[0]?.label || null));
    setLaunchPreset((prev) => {
      if (prev.baseUrl || !list.length) return prev;
      const first = list[0];
      return {
        onlineTarget: first.label,
        baseUrl: first.base_url.startsWith("push://") ? "" : first.base_url,
        sessionName: first.session_name || "",
        model: first.labels?.model || first.server_info?.model || "",
      };
    });
  }, []);

  const loadSessions = React.useCallback(async () => {
    const res = await api<{ sessions: ServiceSession[] }>("/api/sessions");
    const list = res.sessions || [];
    setSessions(list);
    return list;
  }, []);

  const loadJobs = React.useCallback(async () => {
    const res = await api<{ jobs: JobRecord[] }>("/api/jobs");
    const list = res.jobs || [];
    setJobs(list);
    setSelectedJobId((prev) => prev || list[0]?.job_id || null);
  }, []);

  // Reconcile baseline & candidate session IDs whenever sessions change
  React.useEffect(() => {
    if (!sessions.length) {
      setBaseSessionId(null);
      setCandSessionId(null);
      setCompareData(null);
      return;
    }
    const pinned = sessions.find((s) => s.is_baseline);
    const resolvedBase = (pinned || sessions[sessions.length - 1]).session_id;
    setBaseSessionId((prev) =>
      prev && sessions.some((s) => s.session_id === prev) ? prev : resolvedBase
    );
    setCandSessionId((prev) => {
      if (prev && sessions.some((s) => s.session_id === prev)) return prev;
      const withDataDiff = sessions.find(
        (s) => (s.bench_count > 0 || s.eval_count > 0) && s.session_id !== resolvedBase
      );
      const withDataAny = sessions.find((s) => s.bench_count > 0 || s.eval_count > 0);
      return (withDataDiff || withDataAny || sessions[0]).session_id;
    });
  }, [sessions]);

  const refreshCompare = React.useCallback(async (bId: string | null, cId: string | null) => {
    if (!bId || !cId) {
      setCompareData(null);
      return;
    }
    const ids = bId === cId ? [bId] : [bId, cId];
    const data = await api<CompareResponse>(`/api/compare?sessions=${ids.map(encodeURIComponent).join(",")}`);
    setCompareData(data);
  }, []);

  React.useEffect(() => {
    refreshCompare(baseSessionId, candSessionId).catch(() => {});
  }, [baseSessionId, candSessionId, refreshCompare]);

  // Boot & SSE live stream
  React.useEffect(() => {
    loadPlatformInfo().catch(() => {});
    Promise.all([loadLiveTargets(), loadSessions(), loadJobs()]).catch(() => {});

    const es = new EventSource("/api/events");
    es.onopen = () => setSseOnline(true);
    es.onerror = () => setSseOnline(false);

    es.addEventListener("monitor_tick", (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data);
        const d = msg.data || {};
        if (!d.target || !d.point) return;
        setTargets((prev) => {
          const idx = prev.findIndex((x) => x.label === d.target);
          if (idx < 0) {
            loadLiveTargets().catch(() => {});
            return prev;
          }
          const next = [...prev];
          const cur = next[idx];
          next[idx] = {
            ...cur,
            latest: d.point,
            series: [...(cur.series || []).slice(-59), d.point],
          };
          return next;
        });
      } catch (_) {}
    });

    const reloadAllSessions = () => {
      loadSessions()
        .then((list) => {
          if (list.length) {
            setBaseSessionId((b) => {
              setCandSessionId((c) => {
                if (b && c) refreshCompare(b, c).catch(() => {});
                return c;
              });
              return b;
            });
          }
        })
        .catch(() => {});
    };

    es.addEventListener("target_changed", () => {
      loadLiveTargets().catch(() => {});
      reloadAllSessions();
    });

    es.addEventListener("job_updated", (e: MessageEvent) => {
      try {
        const job: JobRecord = JSON.parse(e.data).data;
        if (!job) return;
        setJobs((prev) => {
          const idx = prev.findIndex((j) => j.job_id === job.job_id);
          if (idx >= 0) {
            const copy = [...prev];
            copy[idx] = job;
            return copy;
          }
          return [job, ...prev];
        });
        if (job.status === "succeeded") reloadAllSessions();
      } catch (_) {}
    });

    es.addEventListener("bench_run_added", reloadAllSessions);
    es.addEventListener("correctness_run_added", reloadAllSessions);

    return () => es.close();
  }, [loadPlatformInfo, loadLiveTargets, loadSessions, loadJobs, refreshCompare]);

  const activeJob = React.useMemo(() => {
    return jobs.find((j) => j.status === "running" || j.status === "queued") || null;
  }, [jobs]);

  const inspectedJob = React.useMemo(() => {
    return jobs.find((j) => j.job_id === selectedJobId) || jobs[0] || null;
  }, [jobs, selectedJobId]);

  React.useEffect(() => {
    if (jobConsoleOpen && logPreRef.current) {
      logPreRef.current.scrollTop = logPreRef.current.scrollHeight;
    }
  }, [jobConsoleOpen, inspectedJob?.logs_text]);

  const handleRefreshAll = async () => {
    await Promise.all([loadLiveTargets(), loadSessions(), loadJobs()]);
    await refreshCompare(baseSessionId, candSessionId);
  };

  const handleJumpToBench = (label: string) => {
    const found = targets.find((t) => t.label === label);
    if (found) {
      setFocusedTarget(label);
      setLaunchPreset({
        onlineTarget: label,
        baseUrl: found.base_url.startsWith("push://") ? "" : found.base_url,
        sessionName: found.session_name || "",
        model: found.labels?.model || found.server_info?.model || "",
      });
    }
    setView("launch");
  };

  const handleRequestDeregister = (label: string) => {
    setConfirmState({
      open: true,
      title: `注销监控目标 ${label}？`,
      description: "注销后将停止采集该实例的实时指标，已归档的历史压测与正确性记录仍会保留。",
      confirmLabel: "确认注销",
      action: async () => {
        await api("/api/deregister", { method: "POST", body: JSON.stringify({ target: label }) });
        await loadLiveTargets();
        await loadSessions();
      },
    });
  };

  const handleRequestDeleteSession = (s: ServiceSession) => {
    setConfirmState({
      open: true,
      title: `删除会话「${s.name}」？`,
      description: "此操作将永久移除该服务会话及其名下所有性能压测与正确性校验记录。",
      confirmLabel: "永久删除",
      action: async () => {
        await api(`/api/sessions/${encodeURIComponent(s.session_id)}`, { method: "DELETE" });
        await loadSessions();
      },
    });
  };

  const handleCancelJob = async (jobId: string) => {
    await api(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST", body: "{}" });
    await loadJobs();
  };

  const navItems: Array<{
    id: ViewId;
    label: string;
    icon: React.ComponentType<any>;
    count?: number;
  }> = [
    { id: "monitor", label: "实时监控", icon: Activity, count: targets.length },
    { id: "launch", label: "发起压测", icon: Play },
    { id: "compare", label: "会话对比", icon: GitCompareArrows, count: sessions.length },
    { id: "correctness", label: "正确性检视", icon: ShieldCheck },
  ];

  const renderRailContent = (inSheet = false) => (
    <div className="flex h-full flex-col justify-between bg-[var(--rail-bg)] text-[var(--rail-text)] select-none">
      <div>
        <div className="px-4 pt-4 pb-3.5 border-b border-[var(--rail-border)]">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[7px] bg-[var(--rail-accent)] text-[#06283d]">
              <Gauge className="w-4 h-4" strokeWidth={2.2} aria-hidden="true" />
            </div>
            <div className="min-w-0">
              {inSheet ? (
                <div className="text-[13.5px] font-bold tracking-tight truncate">AI Infra Bench Studio</div>
              ) : (
                <h1 className="text-[13.5px] font-bold tracking-tight m-0 truncate">AI Infra Bench Studio</h1>
              )}
              <div className="flex items-center gap-1.5 mt-0.5 text-[11px] text-[var(--rail-muted)] whitespace-nowrap">
                <span className="truncate max-w-[96px] tabular-mono">v{info.version || "aib"}</span>
                <span>·</span>
                <span className="inline-flex items-center gap-1.5 shrink-0 whitespace-nowrap">
                  <span
                    className={cn(
                      "inline-block w-1.5 h-1.5 rounded-full shrink-0",
                      sseOnline ? "bg-[#34d399] sse-live-dot" : "bg-[#94a3b8]"
                    )}
                    aria-hidden="true"
                  />
                  <span>{sseOnline ? "实时在线" : "重连中"}</span>
                </span>
              </div>
            </div>
          </div>
        </div>

        <nav aria-label="主工作区导航" className="p-2.5 flex flex-col gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = view === item.id;
            return (
              <button
                key={item.id}
                type="button"
                aria-current={active ? "page" : undefined}
                onClick={() => {
                  setView(item.id);
                  if (inSheet) setMobileNavOpen(false);
                }}
                className={cn(
                  "relative flex w-full items-center justify-between gap-2 rounded-[8px] px-3 py-2.5 text-[13px] font-semibold cursor-pointer transition-colors text-left border-0",
                  active
                    ? "bg-[var(--rail-surface)] text-white"
                    : "bg-transparent text-[var(--rail-muted)] hover:bg-[var(--rail-surface)]/60 hover:text-white"
                )}
              >
                {active && (
                  <span
                    aria-hidden="true"
                    className="absolute left-0 top-2 bottom-2 w-[3px] rounded-r bg-[var(--rail-accent)]"
                  />
                )}
                <span className="inline-flex items-center gap-2.5">
                  <Icon
                    className={cn("w-4 h-4 shrink-0", active ? "text-[var(--rail-accent)]" : "")}
                    strokeWidth={1.75}
                    aria-hidden="true"
                  />
                  <span>{item.label}</span>
                </span>
                {item.count != null && (
                  <span
                    className={cn(
                      "rounded-full px-2 py-0.5 text-[11px] tabular-mono",
                      active
                        ? "bg-[#38bdf8]/20 text-[var(--rail-accent)]"
                        : "bg-white/8 text-[var(--rail-muted)]"
                    )}
                  >
                    {item.count}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Rail Footer: Active Job Status Trigger + Theme/Refresh Controls */}
      <div className="p-3 border-t border-[var(--rail-border)] flex flex-col gap-2.5">
        {activeJob ? (
          <button
            type="button"
            onClick={() => {
              setSelectedJobId(activeJob.job_id);
              setJobConsoleOpen(true);
              if (inSheet) setMobileNavOpen(false);
            }}
            className="w-full rounded-[8px] border border-[#38bdf8]/40 bg-[#38bdf8]/10 px-3 py-2 text-left cursor-pointer hover:bg-[#38bdf8]/16 transition-colors"
          >
            <div className="flex items-center justify-between text-[11.5px] font-semibold text-[#e0f2fe]">
              <span className="inline-flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-[#38bdf8]" strokeWidth={1.75} aria-hidden="true" />
                <span>任务运行中</span>
              </span>
              <span className="tabular-mono">{activeJob.progress?.pct ?? 0}%</span>
            </div>
            <div className="mt-1.5 h-1.5 w-full rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full bg-[#38bdf8] transition-all duration-200"
                style={{ width: `${activeJob.progress?.pct ?? 0}%` }}
              />
            </div>
            <div className="mt-1 text-[11px] text-[var(--rail-muted)] tabular-mono truncate">
              {activeJob.progress?.stage || activeJob.job_type}
            </div>
          </button>
        ) : jobs.length > 0 ? (
          <button
            type="button"
            onClick={() => {
              setSelectedJobId(jobs[0].job_id);
              setJobConsoleOpen(true);
              if (inSheet) setMobileNavOpen(false);
            }}
            className="flex w-full items-center justify-between rounded-[8px] border border-[var(--rail-border)] bg-[var(--rail-surface)]/50 px-3 py-2 text-xs text-[var(--rail-muted)] hover:text-white cursor-pointer transition-colors"
          >
            <span className="inline-flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
              <span>最近任务日志</span>
            </span>
            <span className="tabular-mono text-[11px]">{jobs[0].status}</span>
          </button>
        ) : null}

        {grafanaBaseUrl && (
          <a
            href={`${grafanaBaseUrl}/d/sglang-dashboard/sglang-overview?orgId=1&refresh=5s`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between rounded-[8px] border border-[var(--rail-border)] px-3 py-1.5 text-xs font-medium text-[var(--rail-muted)] hover:text-white hover:bg-[var(--rail-surface)] no-underline transition-colors"
          >
            <span>Grafana 监控大盘</span>
            <ExternalLink className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
          </a>
        )}

        <div className="grid grid-cols-2 gap-1.5">
          <button
            type="button"
            onClick={handleRefreshAll}
            className="inline-flex items-center justify-center gap-1.5 min-h-[34px] rounded-[7px] border border-[var(--rail-border)] bg-transparent text-xs font-semibold text-[var(--rail-muted)] hover:text-white hover:bg-[var(--rail-surface)] cursor-pointer transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
            <span>刷新</span>
          </button>

          <button
            type="button"
            onClick={toggleTheme}
            className="inline-flex items-center justify-center gap-1.5 min-h-[34px] rounded-[7px] border border-[var(--rail-border)] bg-transparent text-xs font-semibold text-[var(--rail-muted)] hover:text-white hover:bg-[var(--rail-surface)] cursor-pointer transition-colors"
          >
            {renderedDark ? (
              <>
                <Sun className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                <span>浅色</span>
              </>
            ) : (
              <>
                <Moon className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                <span>深色</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen flex flex-col lg:flex-row bg-[var(--bg)] text-[var(--text)]">
      {/* Desktop Fixed Committed Rail */}
      <aside className="hidden lg:block lg:w-[228px] lg:shrink-0 lg:sticky lg:top-0 lg:h-screen border-r border-[var(--rail-border)]">
        {renderRailContent(false)}
      </aside>

      {/* Mobile Compact Header */}
      <header className="lg:hidden sticky top-0 z-30 flex items-center justify-between gap-2 px-3.5 h-13 bg-[var(--rail-bg)] text-[var(--rail-text)] border-b border-[var(--rail-border)]">
        <div className="flex items-center gap-2 min-w-0">
          <button
            type="button"
            aria-label="打开工作区导航"
            onClick={() => setMobileNavOpen(true)}
            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[7px] text-[var(--rail-text)] hover:bg-[var(--rail-surface)] cursor-pointer border-0 bg-transparent"
          >
            <Menu className="w-4.5 h-4.5" strokeWidth={1.75} aria-hidden="true" />
          </button>
          <span className="text-sm font-bold truncate">AI Infra Bench Studio</span>
          <span className="inline-flex items-center gap-1 text-[11px] text-[var(--rail-muted)] shrink-0">
            <span
              className={cn(
                "inline-block w-1.5 h-1.5 rounded-full shrink-0",
                sseOnline ? "bg-[#34d399] sse-live-dot" : "bg-[#94a3b8]"
              )}
              aria-hidden="true"
            />
            <span>{sseOnline ? "在线" : "离线"}</span>
          </span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {jobs.length > 0 && (
            <button
              type="button"
              aria-label="打开任务实时控制台"
              onClick={() => {
                setSelectedJobId((activeJob || jobs[0]).job_id);
                setJobConsoleOpen(true);
              }}
              className="inline-flex h-8 w-8 items-center justify-center rounded-[7px] text-[var(--rail-muted)] hover:text-white border border-[var(--rail-border)] bg-transparent cursor-pointer"
            >
              <Terminal className="w-4 h-4" strokeWidth={1.75} aria-hidden="true" />
            </button>
          )}
          <button
            type="button"
            aria-label="切换明暗主题"
            onClick={toggleTheme}
            className="inline-flex h-8 w-8 items-center justify-center rounded-[7px] text-[var(--rail-muted)] hover:text-white border border-[var(--rail-border)] bg-transparent cursor-pointer"
          >
            {renderedDark ? (
              <Sun className="w-4 h-4" strokeWidth={1.75} aria-hidden="true" />
            ) : (
              <Moon className="w-4 h-4" strokeWidth={1.75} aria-hidden="true" />
            )}
          </button>
        </div>
      </header>

      {/* Mobile Navigation Drawer */}
      <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
        <SheetContent side="left" className="p-0 w-[250px]">
          <SheetTitle className="sr-only">工作台导航菜单</SheetTitle>
          <SheetDescription className="sr-only">切换实时监控、发起压测、会话对比与正确性检视视图</SheetDescription>
          {renderRailContent(true)}
        </SheetContent>
      </Sheet>

      {/* Main Work Canvas */}
      <main className="flex-1 min-w-0 p-4 sm:p-6 max-w-[1480px]">
        {view === "monitor" && (
          <MonitorView
            targets={targets}
            focusedTarget={focusedTarget}
            onSelectTarget={setFocusedTarget}
            onJumpToBench={handleJumpToBench}
            onRequestDeregister={handleRequestDeregister}
            onRegisterManual={async (url, labels) => {
              await api("/api/register", { method: "POST", body: JSON.stringify({ url, labels }) });
              await loadLiveTargets();
              await loadSessions();
            }}
            onOpenCliSheet={() => setCliSheetOpen(true)}
            grafanaBaseUrl={grafanaBaseUrl}
          />
        )}

        {view === "launch" && (
          <LaunchView
            targets={targets}
            jobs={jobs}
            preset={launchPreset}
            onPresetChange={setLaunchPreset}
            onJobSubmitted={async (jobId) => {
              setSelectedJobId(jobId);
              await loadJobs();
              await loadSessions();
              setJobConsoleOpen(true);
            }}
            onOpenJobConsole={(jobId) => {
              setSelectedJobId(jobId);
              setJobConsoleOpen(true);
            }}
            onOpenSessionInCompare={async (sid) => {
              setCandSessionId(sid);
              await refreshCompare(baseSessionId, sid);
              setView("compare");
            }}
          />
        )}

        {view === "compare" && (
          <CompareView
            sessions={sessions}
            baseSessionId={baseSessionId}
            candSessionId={candSessionId}
            compareData={compareData}
            onSelectBase={setBaseSessionId}
            onSelectCand={setCandSessionId}
            onPinBaseline={async () => {
              if (!baseSessionId) return;
              await api(`/api/sessions/${encodeURIComponent(baseSessionId)}`, {
                method: "PATCH",
                body: JSON.stringify({ is_baseline: true }),
              });
              await loadSessions();
              await refreshCompare(baseSessionId, candSessionId);
            }}
            onDeleteSession={handleRequestDeleteSession}
            onJumpToLaunch={() => setView("launch")}
          />
        )}

        {view === "correctness" && (
          <CorrectnessView compareData={compareData} onJumpToLaunch={() => setView("launch")} />
        )}
      </main>

      {/* Right Slide-Over Sheet: CLI Auto-Registration & Monitoring Stack Status */}
      <Sheet open={cliSheetOpen} onOpenChange={setCliSheetOpen}>
        <SheetContent side="right" className="max-w-lg">
          <SheetHeader>
            <SheetTitle>GPU 服务启动自动注册与监控状态</SheetTitle>
            <SheetDescription>
              固定本平台 IP 后，任意集群机器启动推理服务时包裹此命令即可自动上线，进程退出自动下线归档
            </SheetDescription>
          </SheetHeader>
          <div className="p-5 flex-1 overflow-y-auto flex flex-col gap-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-[var(--text-sec)]">启动包裹命令示例</span>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    navigator.clipboard?.writeText(autoRegCommand);
                    setCopiedCmd(true);
                    setTimeout(() => setCopiedCmd(false), 1500);
                  }}
                >
                  {copiedCmd ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-[var(--emerald)]" strokeWidth={2} aria-hidden="true" />
                      <span>已复制</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                      <span>复制命令</span>
                    </>
                  )}
                </Button>
              </div>
              <pre
                tabIndex={0}
                role="region"
                aria-label="GPU 服务启动自动注册命令"
                className="m-0 rounded-[8px] border border-[#1e293b] bg-[#0b1120] p-3.5 text-xs text-[#e2e8f0] overflow-x-auto whitespace-pre-wrap break-all tabular-mono"
              >
                <code>{autoRegCommand}</code>
              </pre>
            </div>

            <div className="border-t border-[var(--border)] pt-4 flex flex-col gap-2.5 text-xs">
              <div className="font-semibold text-[var(--text-sec)]">可观测组件连接状态</div>
              <div className="flex items-center justify-between py-1.5">
                <span className="text-[var(--text-muted)]">Prometheus 时序采集</span>
                {promBaseUrl ? (
                  <a
                    href={promBaseUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 font-semibold text-[var(--brand)] tabular-mono"
                  >
                    <span>{promBaseUrl}</span>
                    <ExternalLink className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                  </a>
                ) : (
                  <Badge tone="neutral" className="tabular-mono">Disabled (--no-prometheus)</Badge>
                )}
              </div>
              <div className="flex items-center justify-between py-1.5">
                <span className="text-[var(--text-muted)]">Grafana 看板服务</span>
                {grafanaBaseUrl ? (
                  <a
                    href={`${grafanaBaseUrl}/d/sglang-dashboard/sglang-overview?orgId=1&refresh=5s`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 font-semibold text-[var(--brand)] tabular-mono"
                  >
                    <span>{grafanaBaseUrl}</span>
                    <ExternalLink className="w-3 h-3" strokeWidth={1.75} aria-hidden="true" />
                  </a>
                ) : (
                  <Badge tone="neutral" className="tabular-mono">Disabled</Badge>
                )}
              </div>
            </div>
          </div>
        </SheetContent>
      </Sheet>

      {/* Right Slide-Over Sheet: Live Job Console Logs */}
      <Sheet open={jobConsoleOpen} onOpenChange={setJobConsoleOpen}>
        <SheetContent side="right" className="max-w-xl">
          <SheetHeader>
            <SheetTitle>任务实时控制台</SheetTitle>
            <SheetDescription>通过 SSE 实时流查看压测与正确性校验任务阶段进度及标准输出日志</SheetDescription>
          </SheetHeader>
          <div className="p-5 flex-1 overflow-y-auto flex flex-col gap-4">
            {jobs.length > 0 && (
              <div className="flex items-center gap-2">
                <div className="flex-1">
                  <Select value={inspectedJob?.job_id || ""} onValueChange={setSelectedJobId}>
                    <SelectTrigger aria-label="选择任务查看日志" className="tabular-mono text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {jobs.map((j) => (
                        <SelectItem key={j.job_id} value={j.job_id} className="tabular-mono text-xs">
                          {j.job_id} · {j.job_type} ({j.status})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                {inspectedJob && (inspectedJob.status === "queued" || inspectedJob.status === "running") && (
                  <Button variant="danger" size="sm" onClick={() => handleCancelJob(inspectedJob.job_id)}>
                    <StopCircle className="w-3.5 h-3.5" strokeWidth={1.75} aria-hidden="true" />
                    <span>取消任务</span>
                  </Button>
                )}
              </div>
            )}

            {inspectedJob ? (
              <>
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <div className="flex items-center gap-2">
                    <Badge
                      tone={
                        inspectedJob.status === "succeeded"
                          ? "success"
                          : inspectedJob.status === "failed"
                          ? "danger"
                          : "brand"
                      }
                      className="tabular-mono"
                    >
                      {inspectedJob.status}
                    </Badge>
                    <span className="font-semibold text-[var(--text-sec)]">
                      阶段: {inspectedJob.progress?.stage || "-"}
                    </span>
                  </div>
                  {inspectedJob.progress?.live_tps != null && (
                    <span className="tabular-mono font-bold text-[var(--brand)]">
                      {inspectedJob.progress.live_tps} tok/s
                    </span>
                  )}
                </div>

                <div
                  role="progressbar"
                  aria-label="当前任务进度"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={inspectedJob.status === "succeeded" ? 100 : inspectedJob.progress?.pct ?? 0}
                  className="w-full h-2 rounded-full bg-[var(--surface-subtle)] overflow-hidden"
                >
                  <div
                    className="h-full bg-[var(--brand)] transition-all duration-200"
                    style={{
                      width: `${inspectedJob.status === "succeeded" ? 100 : inspectedJob.progress?.pct ?? 0}%`,
                    }}
                  />
                </div>

                <pre
                  ref={logPreRef}
                  tabIndex={0}
                  role="region"
                  aria-label="任务实时输出日志"
                  className="m-0 flex-1 min-h-[340px] rounded-[10px] border border-[#1e293b] bg-[#090d16] p-3.5 text-xs text-[#e2e8f0] overflow-auto whitespace-pre-wrap tabular-mono"
                >
                  <code>
                    {inspectedJob.logs_text ||
                      (inspectedJob.error_message ? `ERROR: ${inspectedJob.error_message}` : "等待执行输出...")}
                  </code>
                </pre>
              </>
            ) : (
              <div className="py-12 text-center text-xs text-[var(--text-muted)]">暂无任务日志</div>
            )}
          </div>
        </SheetContent>
      </Sheet>

      {/* Destructive Action Confirmation Dialog */}
      <ConfirmDialog
        open={confirmState.open}
        onOpenChange={(open) => setConfirmState((s) => ({ ...s, open }))}
        title={confirmState.title}
        description={confirmState.description}
        confirmLabel={confirmState.confirmLabel}
        onConfirm={confirmState.action}
      />
    </div>
  );
}
